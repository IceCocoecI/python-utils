"""Tiny torch.profiler demo for CPU and optional CUDA.

The workload is intentionally small and synthetic. It demonstrates a profiling
workflow without requiring external data or model downloads.

Run:
    conda run -n aigc python aigc-learning/10-cuda-and-triton/examples/torch_profiler_demo.py --steps 5
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile as torch_profile, record_function, tensorboard_trace_handler


OUT = Path(__file__).resolve().parent / "outputs" / "profiler_demo"


class TinyBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Default is CPU so the demo stays clean on machines without NVIDIA GPUs.",
    )
    parser.add_argument("--trace", action="store_true", help="Write TensorBoard trace files")
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Collect memory stats. This may trigger CUDA device probing in some PyTorch builds.",
    )
    parser.add_argument(
        "--show-profiler-stderr",
        action="store_true",
        help="Show low-level profiler stderr messages. Hidden by default for clean CPU teaching output.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@contextlib.contextmanager
def maybe_suppress_stderr(enabled: bool):
    """Suppress fd-level stderr for noisy profiler backends in CPU-only envs."""
    if not enabled:
        yield
        return

    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


def make_batch(batch_size: int, seq_len: int, dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, seq_len, dim, device=device)
    target = torch.randn(batch_size, seq_len, dim, device=device)
    return x, target


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, target: torch.Tensor) -> float:
    optimizer.zero_grad(set_to_none=True)
    with record_function("forward"):
        out = model(x)
        loss = F.mse_loss(out, target)
    with record_function("backward"):
        loss.backward()
    with record_function("optimizer_step"):
        optimizer.step()
    return float(loss.detach().cpu())


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")

    device = resolve_device(args.device)
    model = TinyBlock(args.dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, target = make_batch(args.batch_size, args.seq_len, args.dim, device)

    for _ in range(args.warmup):
        train_step(model, optimizer, x, target)
    synchronize(device)

    start = time.perf_counter()
    losses: list[float] = []

    suppress_profiler_stderr = device.type == "cpu" and not args.show_profiler_stderr
    with maybe_suppress_stderr(suppress_profiler_stderr):
        use_legacy_cpu_profiler = device.type == "cpu" and not args.trace and not args.profile_memory
        if use_legacy_cpu_profiler:
            with torch.autograd.profiler.profile(use_cuda=False, record_shapes=True) as prof:
                for _ in range(args.steps):
                    losses.append(train_step(model, optimizer, x, target))
                    synchronize(device)
        else:
            activities = [ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(ProfilerActivity.CUDA)
            on_trace_ready = tensorboard_trace_handler(str(OUT)) if args.trace else None
            with torch_profile(
                activities=activities,
                record_shapes=True,
                profile_memory=args.profile_memory,
                with_stack=False,
                on_trace_ready=on_trace_ready,
            ) as prof:
                for _ in range(args.steps):
                    losses.append(train_step(model, optimizer, x, target))
                    synchronize(device)
                    prof.step()

    elapsed = time.perf_counter() - start
    samples_per_second = args.steps * args.batch_size / elapsed

    print("Profiler demo")
    print(f"  device: {device}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  seq_len: {args.seq_len}")
    print(f"  dim: {args.dim}")
    print(f"  steps: {args.steps}")
    print(f"  elapsed_s: {elapsed:.4f}")
    print(f"  samples_per_second: {samples_per_second:.2f}")
    print(f"  last_loss: {losses[-1]:.6f}")
    if device.type == "cuda":
        print(f"  max_cuda_memory_mb: {torch.cuda.max_memory_allocated() / 1024**2:.2f}")
    if args.trace:
        print(f"  trace_dir: {OUT}")

    print("\nTop operations by CPU time")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=8))

    if device.type == "cuda":
        print("\nTop operations by CUDA time")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))


if __name__ == "__main__":
    main()
