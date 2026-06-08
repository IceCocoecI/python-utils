"""Toy diffusion acceleration simulation.

This script does not generate images. It uses a tiny latent tensor to show how
scheduler step count affects latency and reconstruction error.

Run:
    conda run -n aigc python diffusion_acceleration_sim.py --latent-size 32
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from common import set_seed


@dataclass(frozen=True)
class SchedulerProfile:
    name: str
    steps: int
    convergence_strength: float
    relative_step_cost: float
    note: str


PROFILES = [
    SchedulerProfile("DDIM", 50, 4.0, 1.00, "baseline deterministic sampler"),
    SchedulerProfile("DPM++ 2M Karras", 25, 4.3, 1.05, "fewer steps with stronger noise schedule"),
    SchedulerProfile("LCM", 4, 2.8, 1.00, "distilled few-step sampler"),
    SchedulerProfile("Turbo", 1, 1.6, 1.00, "single-step distilled model"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-step-ms", type=float, default=35.0)
    return parser.parse_args()


def make_target(batch_size: int, channels: int, latent_size: int) -> torch.Tensor:
    axis = torch.linspace(-math.pi, math.pi, latent_size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    base = torch.stack(
        [
            torch.sin(xx) * torch.cos(yy),
            torch.cos(xx * 0.5),
            torch.sin(yy * 1.5),
            torch.cos(xx + yy),
        ],
        dim=0,
    )
    if channels != 4:
        repeats = math.ceil(channels / 4)
        base = base.repeat(repeats, 1, 1)[:channels]
    return base.unsqueeze(0).repeat(batch_size, 1, 1, 1)


def denoise_step(latent: torch.Tensor, target: torch.Tensor, alpha: float, step_idx: int) -> torch.Tensor:
    residual = target - latent
    correction = torch.tanh(residual) * alpha
    damped_noise = 0.002 * math.sin(step_idx + 1) * torch.sign(latent)
    return latent + correction - damped_noise


def run_profile(profile: SchedulerProfile, noise: torch.Tensor, target: torch.Tensor) -> dict[str, float | int | str]:
    latent = noise.clone()
    start = time.perf_counter()
    alpha = 1.0 - math.exp(-profile.convergence_strength / profile.steps)
    for step_idx in range(profile.steps):
        latent = denoise_step(latent, target, alpha, step_idx)
    wall_ms = (time.perf_counter() - start) * 1000.0
    mse = F.mse_loss(latent, target).item()
    return {
        "name": profile.name,
        "steps": profile.steps,
        "estimated_latency_ms": profile.steps * profile.relative_step_cost,
        "wall_ms_cpu_toy": wall_ms,
        "latent_mse": mse,
        "note": profile.note,
    }


def main() -> None:
    args = parse_args()
    if args.latent_size <= 0 or args.batch_size <= 0 or args.channels <= 0:
        raise ValueError("latent-size, batch-size, and channels must be positive")

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

    target = make_target(args.batch_size, args.channels, args.latent_size)
    noise = torch.randn_like(target)
    scale = args.base_step_ms * args.batch_size * (args.latent_size / 64.0) ** 2

    print("Toy diffusion acceleration simulation")
    print(f"latent_shape={tuple(target.shape)}, base_step_ms={args.base_step_ms}")
    print("name | steps | estimated_latency_ms | wall_ms_cpu_toy | latent_mse | note")
    print("--- | ---: | ---: | ---: | ---: | ---")
    for profile in PROFILES:
        result = run_profile(profile, noise, target)
        estimated = float(result["estimated_latency_ms"]) * scale
        print(
            f"{result['name']} | {result['steps']} | {estimated:.2f} | "
            f"{float(result['wall_ms_cpu_toy']):.3f} | {float(result['latent_mse']):.6f} | {result['note']}"
        )

    print("\nInterpretation")
    print("  Fewer denoising steps reduce latency almost linearly.")
    print("  Distilled few-step profiles need a model trained for that regime; changing only steps is not enough in production.")


if __name__ == "__main__":
    main()
