"""Configuration-driven training example.

Uses Hydra when installed, and falls back to a small stdlib parser otherwise.

Hydra mode:
  python hydra_train.py training.epochs=2 optimizer.lr=0.001

Fallback mode:
  python hydra_train.py --epochs 2 --lr 0.001
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from common import TrainConfig, environment_snapshot, run_training, write_json


def train_from_mapping(cfg: dict[str, Any], output_dir: Path) -> dict[str, float]:
    train_cfg = TrainConfig(
        seed=int(cfg["training"]["seed"]),
        epochs=int(cfg["training"]["epochs"]),
        batch_size=int(cfg["training"]["batch_size"]),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        deterministic=bool(cfg["training"].get("deterministic", False)),
    )
    write_json(output_dir / "resolved_config.json", cfg)
    write_json(output_dir / "environment.json", environment_snapshot())
    metrics = run_training(train_cfg, output_dir=output_dir)
    write_json(output_dir / "metrics.json", metrics)
    return metrics


try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    hydra = None
    DictConfig = object
    OmegaConf = None


if hydra is not None:

    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def main(cfg: DictConfig) -> None:
        output_dir = Path.cwd()
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        metrics = train_from_mapping(cfg_dict, output_dir)
        print(f"final metrics: {metrics}")
        print(f"hydra output dir: {output_dir}")

else:

    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs", type=int, default=4)
        parser.add_argument("--lr", type=float, default=3e-3)
        parser.add_argument("--weight-decay", type=float, default=1e-4)
        parser.add_argument("--hidden-dim", type=int, default=32)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--output-dir", type=Path, default=Path("outputs/hydra-fallback"))
        args = parser.parse_args()
        cfg = {
            "model": {
                "name": "tiny-mlp",
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
            },
            "optimizer": {
                "name": "adamw",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "deterministic": False,
            },
        }
        metrics = train_from_mapping(cfg, args.output_dir)
        print("Hydra is not installed, used fallback argparse mode.")
        print("Install Hydra with: pip install hydra-core")
        print(f"final metrics: {metrics}")
        print(f"output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
