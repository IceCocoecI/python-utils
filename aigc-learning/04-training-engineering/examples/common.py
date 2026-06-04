"""Shared tiny training utilities for module 04 examples.

The dataset is synthetic and deterministic, so every example runs quickly on
CPU without downloading data or requiring external services.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    seed: int = 42
    n_train: int = 512
    n_val: int = 256
    input_dim: int = 16
    hidden_dim: int = 32
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 4
    lr: float = 3e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    deterministic: bool = False


class TinyClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataset(n_samples: int, input_dim: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, input_dim, generator=generator)
    weights = torch.linspace(-1.0, 1.0, input_dim)
    nonlinear = 0.35 * torch.sin(x[:, 0] * 2.0) + 0.25 * x[:, 1].square()
    logits = x @ weights + nonlinear
    y = (logits > logits.median()).long()
    return TensorDataset(x, y)


def build_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    train_ds = make_dataset(cfg.n_train, cfg.input_dim, cfg.seed)
    val_ds = make_dataset(cfg.n_val, cfg.input_dim, cfg.seed + 10_000)
    generator = torch.Generator().manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, val_loader


def build_model(cfg: TrainConfig, device: torch.device) -> TinyClassifier:
    return TinyClassifier(cfg.input_dim, cfg.hidden_dim, cfg.dropout).to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)

    return {
        "train/loss": total_loss / total,
        "train/acc": total_correct / total,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)

    return {
        "val/loss": total_loss / total,
        "val/acc": total_correct / total,
    }


def run_training(
    cfg: TrainConfig,
    log_fn=None,
    output_dir: Path | None = None,
) -> dict[str, float]:
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    last_metrics: dict[str, float] = {}
    global_step = 0
    for epoch in range(cfg.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        last_metrics = {"epoch": float(epoch), **train_metrics, **val_metrics}

        if log_fn is not None:
            log_fn(last_metrics, global_step)

        global_step += len(train_loader)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": asdict(cfg),
                "metrics": last_metrics,
                "rng_state": capture_rng_state(),
            },
            output_dir / "last.pt",
        )

    return last_metrics


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def dataset_checksum(cfg: TrainConfig) -> str:
    train_ds = make_dataset(cfg.n_train, cfg.input_dim, cfg.seed)
    x, y = train_ds.tensors
    digest = hashlib.sha256()
    digest.update(x.numpy().tobytes())
    digest.update(y.numpy().tobytes())
    return digest.hexdigest()


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "git": get_git_hash(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
