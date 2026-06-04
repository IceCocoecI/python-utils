"""Reproducible training template.

Runs the same training twice and checks whether the final metrics match.

  python reproducible_train.py --epochs 2
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from common import (
    TrainConfig,
    dataset_checksum,
    environment_snapshot,
    run_training,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reproducible"))
    return parser.parse_args()


def close_enough(left: dict[str, float], right: dict[str, float], tol: float = 1e-9) -> bool:
    keys = sorted(k for k in left if isinstance(left[k], float))
    return all(abs(left[k] - right[k]) <= tol for k in keys)


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        epochs=args.epochs,
        seed=args.seed,
        deterministic=args.deterministic,
    )

    run_a = run_training(cfg, output_dir=args.output_dir / "run_a")
    run_b = run_training(cfg, output_dir=args.output_dir / "run_b")
    reproducible = close_enough(run_a, run_b)

    payload = {
        "config": asdict(cfg),
        "environment": environment_snapshot(),
        "dataset_sha256": dataset_checksum(cfg),
        "run_a": run_a,
        "run_b": run_b,
        "metrics_match": reproducible,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    write_json(args.output_dir / "reproducibility_report.json", payload)

    print(f"run A: {run_a}")
    print(f"run B: {run_b}")
    print(f"metrics match: {reproducible}")
    print(f"report: {args.output_dir / 'reproducibility_report.json'}")


if __name__ == "__main__":
    main()
