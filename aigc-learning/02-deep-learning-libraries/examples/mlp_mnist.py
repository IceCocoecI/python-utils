"""完整 PyTorch 训练循环示例：MLP on MNIST。
功能：Dataset + DataLoader + Model + Loss + Optim + Scheduler + AMP + Checkpoint + Eval

运行：python mlp_mnist.py
需要联网下载 MNIST (~11MB)；也可以把 data_dir 指向已下载的目录。
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


@dataclass
class Config:
    data_dir: Path = Path("./data")
    ckpt_dir: Path = Path("./checkpoints")
    batch_size: int = 128
    eval_batch_size: int = 256
    num_epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 0.01
    num_workers: int = 2
    seed: int = 42
    use_amp: bool = True


class MLP(nn.Module):
    def __init__(self, in_dim: int = 784, hidden: int = 256, out_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(str(cfg.data_dir), train=True, download=True, transform=tf)
    val_ds = datasets.MNIST(str(cfg.data_dir), train=False, download=True, transform=tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss, total_n = 0.0, 0
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
        if step % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info("epoch %d step %d  loss=%.4f  lr=%.2e", epoch, step, loss.item(), lr)

    return {"train_loss": total_loss / total_n}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += loss.item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)
    return {"val_loss": total_loss / total, "val_acc": correct / total}


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, epoch: int, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics": metrics,
    }, path)


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("./data"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("./checkpoints"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-amp", action="store_true")
    ns = p.parse_args()
    return Config(
        data_dir=ns.data_dir,
        ckpt_dir=ns.ckpt_dir,
        num_epochs=ns.epochs,
        batch_size=ns.batch_size,
        lr=ns.lr,
        use_amp=not ns.no_amp,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = parse_args()
    logger.info("config: %s", asdict(cfg))

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    train_loader, val_loader = build_loaders(cfg)
    total_steps = len(train_loader) * cfg.num_epochs

    model = MLP().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model params: %d", n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=total_steps, pct_start=0.1
    )

    scaler = torch.amp.GradScaler(device.type) if (cfg.use_amp and device.type == "cuda") else None

    best_acc = 0.0
    for epoch in range(cfg.num_epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch)
        val_stats = evaluate(model, val_loader, device)
        logger.info("epoch %d  %s  %s", epoch, train_stats, val_stats)

        if val_stats["val_acc"] > best_acc:
            best_acc = val_stats["val_acc"]
            save_checkpoint(cfg.ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, val_stats)
            logger.info("saved best model with val_acc=%.4f", best_acc)

    logger.info("done. best val_acc=%.4f", best_acc)


if __name__ == "__main__":
    main()
