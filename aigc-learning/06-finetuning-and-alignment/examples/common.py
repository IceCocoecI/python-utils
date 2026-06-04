"""Shared utilities for module 06 fine-tuning and alignment examples.

The examples use tiny synthetic tensors so they can run on CPU without
downloading model weights or datasets. They demonstrate the mechanics behind
LoRA, quantization, SFT data processing, and preference optimization.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass
class TinyRegressionConfig:
    seed: int = 42
    n_train: int = 512
    n_val: int = 128
    input_dim: int = 16
    output_dim: int = 8
    batch_size: int = 64
    epochs: int = 6
    lr: float = 5e-2


class LoRALinear(nn.Module):
    """A minimal LoRA layer for a frozen linear projection.

    This mirrors W' = W + (alpha / rank) * B @ A. The base weight is frozen,
    while lora_A and lora_B are trainable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 8.0,
        base_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        if base_weight is None:
            base_weight = torch.empty(out_features, in_features)
            nn.init.kaiming_uniform_(base_weight, a=math.sqrt(5))
        self.weight = nn.Parameter(base_weight.clone(), requires_grad=False)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base_out + self.scaling * lora_out

    def merged_weight(self) -> torch.Tensor:
        delta = self.scaling * (self.lora_B @ self.lora_A)
        return self.weight.detach() + delta.detach()


class FullLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, base_weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(base_weight.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


def make_linear_shift_dataset(cfg: TinyRegressionConfig) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """Create a task where the target differs from the base model by a low-rank shift."""
    set_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    base_weight = torch.randn(cfg.output_dim, cfg.input_dim, generator=generator) * 0.5

    true_rank = 2
    true_a = torch.randn(true_rank, cfg.input_dim, generator=generator) * 0.6
    true_b = torch.randn(cfg.output_dim, true_rank, generator=generator) * 0.6
    target_weight = base_weight + true_b @ true_a

    def build_split(n_samples: int, split_seed: int) -> TensorDataset:
        split_gen = torch.Generator().manual_seed(split_seed)
        x = torch.randn(n_samples, cfg.input_dim, generator=split_gen)
        y = F.linear(x, target_weight)
        y = y + 0.02 * torch.randn(y.shape, generator=split_gen)
        return TensorDataset(x, y)

    train_ds = build_split(cfg.n_train, cfg.seed + 1)
    val_ds = build_split(cfg.n_val, cfg.seed + 2)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    return train_loader, val_loader, base_weight, target_weight


def train_regression_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
) -> dict[str, float]:
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    train_loss = _regression_loss(model, train_loader)
    val_loss = _regression_loss(model, val_loader)
    return {"train_mse": train_loss, "val_mse": val_loss}


@torch.no_grad()
def _regression_loss(model: nn.Module, loader: DataLoader) -> float:
    total_loss = 0.0
    total = 0
    for x, y in loader:
        loss = F.mse_loss(model(x), y, reduction="sum")
        total_loss += loss.item()
        total += y.numel()
    return total_loss / total


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
