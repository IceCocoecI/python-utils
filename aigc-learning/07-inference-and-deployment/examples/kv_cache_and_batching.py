"""KV Cache sizing and continuous batching simulation.

Run:
    conda run -n aigc python kv_cache_and_batching.py --model llama2-7b --batch-size 4
"""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass

from common import DTYPE_BYTES, GenerationRequest, format_bytes, kv_cache_bytes


MODEL_PRESETS = {
    "llama2-7b": {"num_layers": 32, "num_kv_heads": 32, "head_dim": 128},
    "qwen2.5-7b-gqa": {"num_layers": 28, "num_kv_heads": 4, "head_dim": 128},
    "llama3-70b-gqa": {"num_layers": 80, "num_kv_heads": 8, "head_dim": 128},
}

DEFAULT_REQUESTS = [
    GenerationRequest("r1", prompt_tokens=128, output_tokens=32, arrival_step=0),
    GenerationRequest("r2", prompt_tokens=256, output_tokens=128, arrival_step=0),
    GenerationRequest("r3", prompt_tokens=64, output_tokens=24, arrival_step=0),
    GenerationRequest("r4", prompt_tokens=96, output_tokens=96, arrival_step=0),
    GenerationRequest("r5", prompt_tokens=48, output_tokens=12, arrival_step=2),
    GenerationRequest("r6", prompt_tokens=80, output_tokens=48, arrival_step=3),
    GenerationRequest("r7", prompt_tokens=160, output_tokens=80, arrival_step=4),
    GenerationRequest("r8", prompt_tokens=40, output_tokens=16, arrival_step=5),
]


@dataclass(frozen=True)
class BatchStats:
    strategy: str
    makespan_steps: int
    generated_tokens: int
    occupied_slots: int
    capacity_slots: int
    wasted_slots: int
    throughput_tokens_per_step: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="llama2-7b")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=sorted(DTYPE_BYTES), default="fp16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    return parser.parse_args()


def resolve_model_config(args: argparse.Namespace) -> dict[str, int]:
    cfg = dict(MODEL_PRESETS[args.model])
    for key, arg_name in [
        ("num_layers", "num_layers"),
        ("num_kv_heads", "num_kv_heads"),
        ("head_dim", "head_dim"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            cfg[key] = value
    return cfg


def static_batching_stats(requests: list[GenerationRequest], batch_size: int) -> BatchStats:
    generated = sum(r.output_tokens for r in requests)
    makespan = 0
    occupied = 0
    capacity = 0
    for start in range(0, len(requests), batch_size):
        group = requests[start : start + batch_size]
        group_steps = max(r.output_tokens for r in group)
        makespan += group_steps
        occupied += sum(r.output_tokens for r in group)
        capacity += group_steps * len(group)
    wasted = capacity - occupied
    return BatchStats(
        strategy="static_batching",
        makespan_steps=makespan,
        generated_tokens=generated,
        occupied_slots=occupied,
        capacity_slots=capacity,
        wasted_slots=wasted,
        throughput_tokens_per_step=generated / makespan,
    )


def continuous_batching_stats(requests: list[GenerationRequest], batch_size: int) -> BatchStats:
    pending = sorted(requests, key=lambda r: (r.arrival_step, r.request_id))
    active: list[tuple[GenerationRequest, int]] = []
    step = 0
    occupied = 0
    capacity = 0
    finished = 0
    generated = sum(r.output_tokens for r in requests)

    while finished < len(requests):
        while pending and pending[0].arrival_step <= step and len(active) < batch_size:
            active.append((pending.pop(0), 0))

        if not active and pending:
            step = pending[0].arrival_step
            continue

        next_active: list[tuple[GenerationRequest, int]] = []
        capacity += batch_size
        for req, done in active:
            done += 1
            occupied += 1
            if done >= req.output_tokens:
                finished += 1
            else:
                next_active.append((req, done))
        active = next_active
        step += 1

        while pending and pending[0].arrival_step <= step and len(active) < batch_size:
            active.append((pending.pop(0), 0))

    wasted = capacity - occupied
    return BatchStats(
        strategy="continuous_batching",
        makespan_steps=step,
        generated_tokens=generated,
        occupied_slots=occupied,
        capacity_slots=capacity,
        wasted_slots=wasted,
        throughput_tokens_per_step=generated / step,
    )


def paged_kv_waste(requests: list[GenerationRequest], max_model_len: int, block_size: int) -> dict[str, int | float]:
    reserved_tokens = len(requests) * max_model_len
    actual_tokens = sum(r.total_tokens for r in requests)
    paged_tokens = sum(math.ceil(r.total_tokens / block_size) * block_size for r in requests)
    return {
        "requests": len(requests),
        "actual_tokens": actual_tokens,
        "reserved_tokens_traditional": reserved_tokens,
        "allocated_tokens_paged": paged_tokens,
        "traditional_waste_tokens": reserved_tokens - actual_tokens,
        "paged_waste_tokens": paged_tokens - actual_tokens,
        "traditional_utilization": actual_tokens / reserved_tokens,
        "paged_utilization": actual_tokens / paged_tokens,
    }


def print_stats(stats: BatchStats) -> None:
    print(f"\n{stats.strategy}")
    for key, value in asdict(stats).items():
        if key == "strategy":
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.block_size <= 0:
        raise ValueError("block_size must be positive")

    cfg = resolve_model_config(args)
    dtype_bytes = DTYPE_BYTES[args.dtype]
    cache_bytes = kv_cache_bytes(
        num_layers=cfg["num_layers"],
        num_kv_heads=cfg["num_kv_heads"],
        head_dim=cfg["head_dim"],
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        dtype_bytes=dtype_bytes,
    )

    print("KV Cache estimate")
    print(f"  model: {args.model}")
    print(f"  config: {cfg}")
    print(f"  seq_len: {args.seq_len}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  dtype: {args.dtype} ({dtype_bytes} bytes/value)")
    print(f"  kv_cache: {format_bytes(cache_bytes)}")

    print("\nRequests")
    for req in DEFAULT_REQUESTS:
        print(f"  {req.request_id}: prompt={req.prompt_tokens}, output={req.output_tokens}, arrival={req.arrival_step}")

    print_stats(static_batching_stats(DEFAULT_REQUESTS, args.batch_size))
    print_stats(continuous_batching_stats(DEFAULT_REQUESTS, args.batch_size))

    waste = paged_kv_waste(DEFAULT_REQUESTS, args.max_model_len, args.block_size)
    print("\nPaged KV allocation simulation")
    for key, value in waste.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
