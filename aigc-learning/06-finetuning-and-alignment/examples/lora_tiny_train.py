"""Train a tiny LoRA adapter and compare it with full fine-tuning.

Run:
    conda run -n aigc python lora_tiny_train.py --epochs 6 --rank 2
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from common import (
    FullLinear,
    LoRALinear,
    TinyRegressionConfig,
    count_parameters,
    make_linear_shift_dataset,
    train_regression_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TinyRegressionConfig(
        seed=args.seed,
        n_train=1024,
        n_val=256,
        input_dim=64,
        output_dim=64,
        batch_size=128,
        epochs=args.epochs,
        lr=args.lr,
    )
    train_loader, val_loader, base_weight, target_weight = make_linear_shift_dataset(cfg)

    full_model = FullLinear(cfg.input_dim, cfg.output_dim, base_weight)
    lora_model = LoRALinear(
        cfg.input_dim,
        cfg.output_dim,
        rank=args.rank,
        alpha=args.alpha,
        base_weight=base_weight,
    )

    full_metrics = train_regression_model(full_model, train_loader, val_loader, cfg.epochs, cfg.lr)
    lora_metrics = train_regression_model(lora_model, train_loader, val_loader, cfg.epochs, cfg.lr)

    full_total, full_trainable = count_parameters(full_model)
    lora_total, lora_trainable = count_parameters(lora_model)

    merged_weight = lora_model.merged_weight()
    with torch.no_grad():
        merge_error = F.mse_loss(merged_weight, target_weight).item()

    print("LoRA tiny training demo")
    print(f"full_finetune: total_params={full_total}, trainable_params={full_trainable}")
    print(f"lora_adapter: total_params={lora_total}, trainable_params={lora_trainable}, rank={args.rank}")
    print(f"trainable_reduction={full_trainable / lora_trainable:.2f}x")
    print(f"full_val_mse={full_metrics['val_mse']:.6f}")
    print(f"lora_val_mse={lora_metrics['val_mse']:.6f}")
    print(f"merged_weight_mse_vs_target={merge_error:.6f}")


if __name__ == "__main__":
    main()
