"""Minimal torch.distributed collective communication demo.

Run with two CPU processes:
  torchrun --standalone --nproc_per_node=2 collectives_demo.py
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    backend = os.environ.get("DIST_BACKEND", "gloo")
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    value = torch.tensor([float(rank + 1)])
    dist.all_reduce(value, op=dist.ReduceOp.SUM)

    source = torch.tensor([123.0 if rank == 0 else 0.0])
    dist.broadcast(source, src=0)

    gathered = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(gathered, torch.tensor([float(rank)]))

    if rank == 0:
        print(f"world_size={world_size}")
        print(f"all_reduce_sum={value.item():.1f}")
        print(f"broadcast_value={source.item():.1f}")
        print(f"all_gather={[x.item() for x in gathered]}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
