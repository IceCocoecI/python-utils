"""Memory math for DDP vs FSDP sharding strategies.

This example is runnable on CPU because it computes the memory budget instead
of launching GPU-only FSDP training.
"""
from __future__ import annotations

import argparse


def gb(num_bytes: float) -> float:
    return num_bytes / 1024**3


def estimate(params_billion: float, world_size: int) -> list[dict[str, float | str]]:
    params = params_billion * 1_000_000_000
    param_bf16 = params * 2
    grad_bf16 = params * 2
    adam_states = params * 12
    rows = []
    rows.append(
        {
            "strategy": "DDP / NO_SHARD",
            "params_gb": gb(param_bf16),
            "grads_gb": gb(grad_bf16),
            "optimizer_gb": gb(adam_states),
        }
    )
    rows.append(
        {
            "strategy": "SHARD_GRAD_OP",
            "params_gb": gb(param_bf16),
            "grads_gb": gb(grad_bf16 / world_size),
            "optimizer_gb": gb(adam_states / world_size),
        }
    )
    rows.append(
        {
            "strategy": "FULL_SHARD / ZeRO-3",
            "params_gb": gb(param_bf16 / world_size),
            "grads_gb": gb(grad_bf16 / world_size),
            "optimizer_gb": gb(adam_states / world_size),
        }
    )
    for row in rows:
        row["total_gb"] = row["params_gb"] + row["grads_gb"] + row["optimizer_gb"]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-billion", type=float, default=7.0)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()

    print(f"model={args.params_billion:g}B params, world_size={args.world_size}")
    print("strategy                 params   grads    optimizer   total")
    for row in estimate(args.params_billion, args.world_size):
        print(
            f"{row['strategy']:<24}"
            f"{row['params_gb']:>7.2f}G"
            f"{row['grads_gb']:>8.2f}G"
            f"{row['optimizer_gb']:>11.2f}G"
            f"{row['total_gb']:>8.2f}G"
        )
    print("Note: activation memory, fragmentation, and communication buffers are not included.")


if __name__ == "__main__":
    main()
