"""Minimal OpenAI-compatible chat server using only the Python standard library.

Run self-test:
    conda run -n aigc python openai_compatible_toy_server.py --self-test

Start server:
    conda run -n aigc python openai_compatible_toy_server.py --host 127.0.0.1 --port 8000

Then call:
    curl -N http://127.0.0.1:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"toy","messages":[{"role":"user","content":"explain KV Cache"}],"stream":true}'
"""
from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from common import build_chat_response, build_sse_event, chunk_words, toy_chat_completion


class ToyOpenAIHandler(BaseHTTPRequestHandler):
    server_version = "ToyOpenAI/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep demo output focused; remove this override when debugging requests.
        return

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path == "/health":
            self._write_json({"status": "ok", "model": "toy-inference-model"})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/v1/chat/completions":
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return

        try:
            payload = self._read_json()
            messages = payload.get("messages", [])
            model = payload.get("model", "toy-inference-model")
            max_tokens = int(payload.get("max_tokens", 64))
            temperature = float(payload.get("temperature", 0.7))
            stream = bool(payload.get("stream", False))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._write_json({"error": {"message": str(exc), "type": "bad_request"}}, status=400)
            return

        if stream:
            self._write_sse(messages, model=model, max_tokens=max_tokens, temperature=temperature)
        else:
            response = build_chat_response(messages, model=model, max_tokens=max_tokens, temperature=temperature)
            self._write_json(response)

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8")) if body else {}

    def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_sse(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        content = toy_chat_completion(messages, max_tokens=max_tokens, temperature=temperature)
        for delta in chunk_words(content):
            self.wfile.write(build_sse_event(delta, model=model).encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def run_server(host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, port), ToyOpenAIHandler)
    print(f"Serving toy OpenAI-compatible API at http://{host}:{port}")
    print("Press Ctrl+C to stop")
    httpd.serve_forever()


def self_test() -> None:
    messages = [{"role": "user", "content": "Explain KV Cache and streaming"}]
    response = build_chat_response(messages, model="toy", max_tokens=32)
    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"]["content"]
    event = build_sse_event("hello", model="toy")
    assert event.startswith("data: ") and event.endswith("\n\n")
    assert "hello" in event
    print("openai_compatible_toy_server self-test passed")
    print(json.dumps(response, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
