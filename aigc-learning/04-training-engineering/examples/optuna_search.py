"""Hyperparameter search example.

Uses Optuna when installed, and falls back to deterministic random search.

Optuna mode:
  python optuna_search.py --trials 8

Fallback mode works with only PyTorch:
  python optuna_search.py --trials 8
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from common import TrainConfig, environment_snapshot, run_training, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/optuna-search"))
    return parser.parse_args()


def evaluate_config(cfg: TrainConfig) -> dict[str, float]:
    return run_training(cfg)


def run_optuna(args: argparse.Namespace) -> dict[str, Any]:
    import optuna

    def objective(trial: optuna.Trial) -> float:
        cfg = TrainConfig(
            seed=args.seed,
            epochs=args.epochs,
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            hidden_dim=trial.suggest_categorical("hidden_dim", [16, 32, 64]),
            dropout=trial.suggest_float("dropout", 0.0, 0.35),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        )

        def report(metrics: dict[str, float], step: int) -> None:
            epoch = int(metrics["epoch"])
            trial.report(metrics["val/loss"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        metrics = run_training(cfg, log_fn=report)
        trial.set_user_attr("metrics", metrics)
        return metrics["val/loss"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=args.trials)
    return {
        "mode": "optuna",
        "best_value": study.best_value,
        "best_params": study.best_trial.params,
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }


def run_random_search(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    trials = []
    best = None
    for trial_id in range(args.trials):
        lr = 10 ** rng.uniform(math.log10(1e-4), math.log10(1e-2))
        weight_decay = 10 ** rng.uniform(math.log10(1e-6), math.log10(1e-2))
        cfg = TrainConfig(
            seed=args.seed,
            epochs=args.epochs,
            lr=lr,
            hidden_dim=rng.choice([16, 32, 64]),
            dropout=rng.uniform(0.0, 0.35),
            weight_decay=weight_decay,
        )
        metrics = evaluate_config(cfg)
        record = {
            "number": trial_id,
            "value": metrics["val/loss"],
            "params": asdict(cfg),
            "metrics": metrics,
        }
        trials.append(record)
        if best is None or record["value"] < best["value"]:
            best = record

    return {
        "mode": "random_search_fallback",
        "note": "Install optuna for TPE sampling, pruning, storage, and visualizations.",
        "best_value": best["value"],
        "best_params": best["params"],
        "trials": trials,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "environment.json", environment_snapshot())

    try:
        result = run_optuna(args)
    except ImportError:
        result = run_random_search(args)

    write_json(args.output_dir / "search_results.json", result)
    print(f"mode: {result['mode']}")
    print(f"best value: {result['best_value']:.4f}")
    print(f"best params: {result['best_params']}")
    print(f"results: {args.output_dir / 'search_results.json'}")


if __name__ == "__main__":
    main()
