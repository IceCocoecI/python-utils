"""Experiment tracking example with TensorBoard and optional W&B.

Runs offline by default:
  python wandb_train.py --epochs 2

Enable W&B after installing/login:
  WANDB_MODE=offline python wandb_train.py --use-wandb
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from common import TrainConfig, environment_snapshot, run_training, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/tiny-tracking"))
    parser.add_argument("--use-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.log_dir / "config.json", asdict(cfg))
    write_json(args.log_dir / "environment.json", environment_snapshot())

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit(
                "wandb is not installed. Install it with: pip install wandb"
            ) from exc

        wandb_run = wandb.init(
            project="aigc-training-engineering",
            name=f"tiny-lr{cfg.lr}-seed{cfg.seed}",
            config={**asdict(cfg), **environment_snapshot()},
        )

    writer = SummaryWriter(log_dir=str(args.log_dir))

    def log_metrics(metrics: dict[str, float], step: int) -> None:
        for key, value in metrics.items():
            writer.add_scalar(key, value, step)
        if wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step)

    metrics = run_training(cfg, log_fn=log_metrics, output_dir=args.log_dir)
    writer.close()
    if wandb_run is not None:
        wandb_run.finish()

    write_json(args.log_dir / "metrics.json", metrics)
    print(f"final metrics: {metrics}")
    print(f"tensorboard: tensorboard --logdir {args.log_dir.parent}")


if __name__ == "__main__":
    main()
