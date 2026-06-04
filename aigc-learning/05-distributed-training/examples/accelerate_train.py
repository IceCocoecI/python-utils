"""Accelerate training example.

Single process:
  python accelerate_train.py --epochs 1

Two CPU processes:
  accelerate launch --cpu --num_processes=2 accelerate_train.py --epochs 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
from torch.utils.data import DataLoader

from common import (
    TrainConfig,
    accuracy,
    build_model,
    config_dict,
    environment_snapshot,
    make_dataset,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/accelerate"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=2)
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    accelerate_set_seed(cfg.seed)

    train_ds = make_dataset(cfg.n_samples, cfg.input_dim, cfg.seed)
    val_ds = make_dataset(cfg.n_val, cfg.input_dim, cfg.seed + 10_000)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
    )

    final_metrics: dict[str, float] = {}
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        steps = 0
        for x, y in train_loader:
            with accelerator.accumulate(model):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_loss += loss.detach().float().item()
            train_acc += accuracy(logits.detach(), y)
            steps += 1

        model.eval()
        val_loss_total = torch.tensor(0.0, device=accelerator.device)
        val_correct_total = torch.tensor(0.0, device=accelerator.device)
        val_count_total = torch.tensor(0.0, device=accelerator.device)
        for x, y in val_loader:
            with torch.no_grad():
                logits = model(x)
                loss = F.cross_entropy(logits, y, reduction="sum")
            val_loss_total += loss.detach()
            val_correct_total += (logits.argmax(dim=-1) == y).sum()
            val_count_total += y.numel()

        val_loss_total = accelerator.gather(val_loss_total).sum()
        val_correct_total = accelerator.gather(val_correct_total).sum()
        val_count_total = accelerator.gather(val_count_total).sum()
        final_metrics = {
            "epoch": float(epoch),
            "train/loss": train_loss / max(steps, 1),
            "train/acc": train_acc / max(steps, 1),
            "val/loss": (val_loss_total / val_count_total).item(),
            "val/acc": (val_correct_total / val_count_total).item(),
        }
        accelerator.print(f"epoch={epoch} metrics={final_metrics}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), args.output_dir / "model.pt")
        write_json(
            args.output_dir / "run.json",
            {
                "config": config_dict(cfg),
                "environment": environment_snapshot(),
                "num_processes": accelerator.num_processes,
                "metrics": final_metrics,
            },
        )
        accelerator.print(f"saved: {args.output_dir / 'run.json'}")


if __name__ == "__main__":
    main()
