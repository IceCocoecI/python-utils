"""Simulate per-tensor and per-group weight quantization.

This is not a kernel-level speed demo. It shows how scale, zero point, group
size, and bit width affect reconstruction error and output drift.

Run:
    conda run -n aigc python quantization_sim.py --bits 4 --group-size 32
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from common import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def symmetric_quantize(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax().clamp_min(1e-8) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q, scale


def symmetric_dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q * scale


def group_quantize_dequantize(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    rows, cols = weight.shape
    padded_cols = math.ceil(cols / group_size) * group_size
    if padded_cols != cols:
        pad = torch.zeros(rows, padded_cols - cols, dtype=weight.dtype)
        work = torch.cat([weight, pad], dim=1)
    else:
        work = weight

    grouped = work.reshape(rows, -1, group_size)
    qmax = 2 ** (bits - 1) - 1
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(grouped / scale).clamp(-qmax, qmax)
    dequant = (q * scale).reshape(rows, padded_cols)
    return dequant[:, :cols]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    weight = torch.randn(args.rows, args.cols) * 0.4
    outlier_cols = torch.arange(0, args.cols, max(1, args.cols // 8))
    weight[:, outlier_cols] *= 4.0

    x = torch.randn(256, args.cols)
    y_ref = F.linear(x, weight)

    q_tensor, tensor_scale = symmetric_quantize(weight, args.bits)
    tensor_dequant = symmetric_dequantize(q_tensor, tensor_scale)
    group_dequant = group_quantize_dequantize(weight, args.bits, args.group_size)

    tensor_weight_mse = F.mse_loss(tensor_dequant, weight).item()
    group_weight_mse = F.mse_loss(group_dequant, weight).item()
    tensor_output_mse = F.mse_loss(F.linear(x, tensor_dequant), y_ref).item()
    group_output_mse = F.mse_loss(F.linear(x, group_dequant), y_ref).item()

    fp16_bytes = weight.numel() * 2
    packed_bytes = weight.numel() * args.bits / 8
    scale_bytes = args.rows * math.ceil(args.cols / args.group_size) * 2

    print("Quantization simulation")
    print(f"shape={tuple(weight.shape)}, bits={args.bits}, group_size={args.group_size}")
    print(f"fp16_weight_bytes={fp16_bytes:.0f}")
    print(f"packed_weight_bytes_ideal={packed_bytes:.0f}")
    print(f"group_scale_bytes_fp16={scale_bytes:.0f}")
    print(f"per_tensor_weight_mse={tensor_weight_mse:.8f}")
    print(f"per_group_weight_mse={group_weight_mse:.8f}")
    print(f"per_tensor_output_mse={tensor_output_mse:.8f}")
    print(f"per_group_output_mse={group_output_mse:.8f}")


if __name__ == "__main__":
    main()
