"""完整 PyTorch 训练循环示例：MLP on MNIST。
功能：Dataset + DataLoader + Model + Loss + Optim + Scheduler + AMP + Checkpoint + Eval

运行：python mlp_mnist.py
需要联网下载 MNIST (~11MB)；也可以把 data_dir 指向已下载的目录。
离线 smoke test：python mlp_mnist.py --synthetic --epochs 1 --max-train-batches 3 --max-val-batches 2
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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
    download: bool = True
    synthetic: bool = False
    synthetic_train_size: int = 1024
    synthetic_val_size: int = 256
    max_train_batches: int | None = None
    max_val_batches: int | None = None


class SyntheticMNIST(Dataset):
    """可离线跑通训练循环的合成 MNIST。

    每个类别对应一个固定的横线/竖线组合，叠加少量噪声。它不是 MNIST 的替代品，
    但足够验证 Dataset、DataLoader、反传、评估、checkpoint 这条链路。
    """

    def __init__(self, size: int, seed: int = 42):
        generator = torch.Generator().manual_seed(seed)
        labels = torch.arange(size, dtype=torch.long) % 10
        images = 0.05 * torch.randn(size, 1, 28, 28, generator=generator)
        for idx, label in enumerate(labels.tolist()):
            row = 2 + (label // 5) * 12 + (label % 5) * 2
            col = 2 + (label % 5) * 5
            images[idx, 0, row:row + 3, 3:25] += 0.8
            images[idx, 0, 3:25, col:col + 2] += 0.6
        images = images.clamp(0.0, 1.0)
        self.images = (images - 0.1307) / 0.3081
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


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
    if cfg.synthetic:
        train_ds = SyntheticMNIST(cfg.synthetic_train_size, seed=cfg.seed)
        val_ds = SyntheticMNIST(cfg.synthetic_val_size, seed=cfg.seed + 1)
    else:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(str(cfg.data_dir), train=True, download=cfg.download, transform=tf)
        val_ds = datasets.MNIST(str(cfg.data_dir), train=False, download=cfg.download, transform=tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
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
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    total_loss, total_n = 0.0, 0
    for step, (x, y) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
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

    return {"train_loss": total_loss / max(total_n, 1)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    for step, (x, y) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += loss.item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)
    return {"val_loss": total_loss / max(total, 1), "val_acc": correct / max(total, 1)}


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
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="Use a generated MNIST-like dataset; no network needed.")
    p.add_argument("--synthetic-train-size", type=int, default=1024)
    p.add_argument("--synthetic-val-size", type=int, default=256)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    ns = p.parse_args()
    return Config(
        data_dir=ns.data_dir,
        ckpt_dir=ns.ckpt_dir,
        num_epochs=ns.epochs,
        batch_size=ns.batch_size,
        eval_batch_size=ns.eval_batch_size,
        lr=ns.lr,
        num_workers=ns.workers,
        use_amp=not ns.no_amp,
        download=not ns.no_download,
        synthetic=ns.synthetic,
        synthetic_train_size=ns.synthetic_train_size,
        synthetic_val_size=ns.synthetic_val_size,
        max_train_batches=ns.max_train_batches,
        max_val_batches=ns.max_val_batches,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = parse_args()
    logger.info("config: %s", asdict(cfg))

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    train_loader, val_loader = build_loaders(cfg)
    steps_per_epoch = len(train_loader)
    if cfg.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, cfg.max_train_batches)
    total_steps = max(steps_per_epoch * cfg.num_epochs, 1)

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
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch, cfg.max_train_batches
        )
        val_stats = evaluate(model, val_loader, device, cfg.max_val_batches)
        logger.info("epoch %d  %s  %s", epoch, train_stats, val_stats)

        if val_stats["val_acc"] > best_acc:
            best_acc = val_stats["val_acc"]
            save_checkpoint(cfg.ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, val_stats)
            logger.info("saved best model with val_acc=%.4f", best_acc)

    logger.info("done. best val_acc=%.4f", best_acc)


if __name__ == "__main__":
    main()
