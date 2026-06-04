"""CPU DDP training example using gloo.

This is intentionally small and runs without GPUs:
  torchrun --standalone --nproc_per_node=2 ddp_cpu_train.py --epochs 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from common import (
    TrainConfig,
    accuracy,
    build_model,
    config_dict,
    environment_snapshot,
    make_dataset,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ddp-cpu"))
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: torch.nn.Module, cfg: TrainConfig) -> dict[str, float]:
    dataset = make_dataset(cfg.n_val, cfg.input_dim, cfg.seed + 10_000)
    loader = DataLoader(dataset, batch_size=cfg.batch_size)
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total = 0
    for x, y in loader:
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)
    return {"val/loss": total_loss / total, "val/acc": total_correct / total}


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    set_seed(cfg.seed + rank)

    dataset = make_dataset(cfg.n_samples, cfg.input_dim, cfg.seed)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=cfg.seed,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler)

    model = DDP(build_model(cfg))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    metrics: dict[str, float] = {}
    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        steps = 0
        for x, y in loader:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += accuracy(logits.detach(), y)
            steps += 1

        loss_tensor = torch.tensor([total_loss / steps, total_acc / steps])
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= world_size
        metrics = {
            "epoch": float(epoch),
            "train/loss": loss_tensor[0].item(),
            "train/acc": loss_tensor[1].item(),
        }
        if rank == 0:
            metrics.update(evaluate(model.module, cfg))
            print(f"epoch={epoch} metrics={metrics}")

    dist.barrier()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.module.state_dict(), args.output_dir / "model.pt")
        write_json(
            args.output_dir / "run.json",
            {
                "config": config_dict(cfg),
                "environment": environment_snapshot(),
                "world_size": world_size,
                "metrics": metrics,
            },
        )
        print(f"saved: {args.output_dir / 'run.json'}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
