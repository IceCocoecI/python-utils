"""Shared helpers for module 07 inference and deployment examples.

The examples intentionally use tiny synthetic workloads so they run on CPU
without downloading model weights. They demonstrate scheduling, memory, API,
and UI mechanics that production systems implement around real models.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Iterator, Sequence


DTYPE_BYTES = {
    "fp32": 4.0,
    "float32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "float16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    arrival_step: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


def set_seed(seed: int = 42) -> None:
    random.seed(seed)


def kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    dtype_bytes: float,
) -> int:
    """Return KV cache bytes for decoder-only LLM inference.

    Formula: 2 * layers * kv_heads * head_dim * seq_len * batch * bytes.
    The leading 2 accounts for Key and Value tensors.
    """
    value = 2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * dtype_bytes
    return int(value)


def format_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def now_unix() -> int:
    return int(time.time())


def last_user_message(messages: Sequence[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def deterministic_tokens(seed_text: str, max_tokens: int) -> list[str]:
    """Generate deterministic pseudo tokens for demos and API tests."""
    vocabulary = [
        "prefill",
        "decode",
        "cache",
        "batch",
        "latency",
        "throughput",
        "scheduler",
        "stream",
        "gateway",
        "metric",
        "token",
        "service",
    ]
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    tokens: list[str] = []
    for i in range(max_tokens):
        idx = digest[i % len(digest)] % len(vocabulary)
        tokens.append(vocabulary[idx])
    return tokens


def toy_chat_completion(
    messages: Sequence[dict[str, str]],
    max_tokens: int = 64,
    temperature: float = 0.7,
) -> str:
    """A deterministic chat function with an OpenAI-like call shape.

    This is deliberately not a language model. It gives stable responses so
    protocol examples, SSE clients, and UI demos can be tested without a GPU.
    """
    user_text = last_user_message(messages).strip() or "empty prompt"
    lowered = user_text.lower()
    if "kv" in lowered or "cache" in lowered:
        prefix = "KV Cache stores per-layer Key and Value tensors so decode avoids recomputing history."
    elif "batch" in lowered or "throughput" in lowered:
        prefix = "Continuous batching schedules requests at token-step granularity to keep the accelerator busy."
    elif "diffusion" in lowered or "scheduler" in lowered:
        prefix = "Diffusion latency is dominated by repeated denoising steps; faster schedulers trade steps against quality."
    elif "stream" in lowered or "sse" in lowered:
        prefix = "SSE streams partial deltas so users see the first token before the full completion is ready."
    else:
        prefix = "This toy backend mirrors the serving protocol while replacing the real model with deterministic text."

    budget = max(0, max_tokens - len(prefix.split()))
    tail = deterministic_tokens(f"{user_text}|{temperature}", min(budget, 24))
    return " ".join([prefix, *tail]).strip()


def chunk_words(text: str, chunk_size: int = 3) -> Iterator[str]:
    words = text.split()
    for start in range(0, len(words), chunk_size):
        yield " ".join(words[start : start + chunk_size]) + (" " if start + chunk_size < len(words) else "")


def build_chat_response(
    messages: Sequence[dict[str, str]],
    model: str = "toy-inference-model",
    max_tokens: int = 64,
    temperature: float = 0.7,
) -> dict:
    content = toy_chat_completion(messages, max_tokens=max_tokens, temperature=temperature)
    prompt_tokens = sum(len(m.get("content", "").split()) for m in messages)
    completion_tokens = len(content.split())
    return {
        "id": "chatcmpl-toy-001",
        "object": "chat.completion",
        "created": now_unix(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def build_sse_event(text_delta: str, model: str = "toy-inference-model") -> str:
    payload = {
        "id": "chatcmpl-toy-001",
        "object": "chat.completion.chunk",
        "created": now_unix(),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": text_delta}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
