"""Shared utilities for module 05 distributed training examples.

The examples intentionally use synthetic data and small models so they can run
on CPU with the gloo backend. GPU/NCCL-specific topics are covered in the docs.
"""
from __future__ import annotations

import json
import math
import os
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset


@dataclass
class TrainConfig:
    seed: int = 42
    n_samples: int = 512
    n_val: int = 256
    input_dim: int = 16
    hidden_dim: int = 32
    num_classes: int = 2
    batch_size: int = 64
    epochs: int = 2
    lr: float = 3e-3
    weight_decay: float = 1e-4


class TinyClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_dataset(n_samples: int, input_dim: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, input_dim, generator=generator)
    weights = torch.linspace(-1.0, 1.0, input_dim)
    logits = x @ weights + 0.3 * torch.sin(x[:, 0] * math.pi)
    y = (logits > logits.median()).long()
    return TensorDataset(x, y)


def build_model(cfg: TrainConfig) -> TinyClassifier:
    return TinyClassifier(cfg.input_dim, cfg.hidden_dim, cfg.num_classes)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def config_dict(cfg: TrainConfig) -> dict[str, Any]:
    return asdict(cfg)
