"""FastAPI gateway template for an OpenAI-compatible inference backend.

The self-test path only requires pydantic and httpx, both available in the
current aigc environment. Starting the real gateway additionally requires:
    pip install fastapi uvicorn

Run self-test:
    conda run -n aigc python fastapi_gateway.py --self-test

Start gateway after installing optional deps:
    conda run -n aigc python fastapi_gateway.py --backend-url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
from collections.abc import AsyncIterator

import httpx
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant|tool)$")
    content: str


class ChatRequest(BaseModel):
    model: str = "toy-inference-model"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    stream: bool = False


def to_backend_payload(request: ChatRequest, default_model: str) -> dict:
    return {
        "model": request.model or default_model,
        "messages": [m.model_dump() for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": request.stream,
    }


async def proxy_sse(client: httpx.AsyncClient, payload: dict) -> AsyncIterator[str]:
    async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line:
                yield f"{line}\n\n"


def build_app(backend_url: str, default_model: str):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
    except ImportError as exc:  # pragma: no cover - exercised by CLI message
        raise RuntimeError("Install optional deps first: pip install fastapi uvicorn") from exc

    app = FastAPI(title="OpenAI-Compatible Gateway")
    app.state.client = httpx.AsyncClient(base_url=backend_url, timeout=300.0)

    @app.on_event("shutdown")
    async def close_client() -> None:
        await app.state.client.aclose()

    @app.get("/health")
    async def health() -> dict:
        try:
            response = await app.state.client.get("/health")
            backend_ok = response.status_code == 200
        except httpx.RequestError:
            backend_ok = False
        return {"status": "ok" if backend_ok else "degraded", "backend_ok": backend_ok}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        payload = to_backend_payload(request, default_model=default_model)
        if request.stream:
            return StreamingResponse(proxy_sse(app.state.client, payload), media_type="text/event-stream")
        try:
            response = await app.state.client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"backend error: {exc}") from exc
        return response.json()

    return app


def self_test() -> None:
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="Explain continuous batching")],
        max_tokens=32,
        stream=True,
    )
    payload = to_backend_payload(request, default_model="toy")
    assert payload["messages"][0]["role"] == "user"
    assert payload["stream"] is True
    assert payload["max_tokens"] == 32
    print("fastapi_gateway self-test passed")
    print(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="toy-inference-model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install optional deps first: pip install fastapi uvicorn") from exc

    app = build_app(args.backend_url, default_model=args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
