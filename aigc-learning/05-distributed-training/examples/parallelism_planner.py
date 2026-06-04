"""Simple planner for DP/TP/PP/EP factorization.

The planner is not a replacement for benchmark-driven design. It gives a quick
sanity check for whether a proposed parallelism layout fits the GPU count.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass


@dataclass
class Plan:
    total_gpus: int
    tensor_parallel: int
    pipeline_parallel: int
    expert_parallel: int
    data_parallel: int
    valid: bool
    note: str


def make_plan(total_gpus: int, tp: int, pp: int, ep: int) -> Plan:
    used = tp * pp * ep
    if used <= 0 or total_gpus % used != 0:
        return Plan(
            total_gpus,
            tp,
            pp,
            ep,
            0,
            False,
            "total_gpus must be divisible by tp * pp * ep",
        )
    dp = total_gpus // used
    notes = []
    if tp > 1:
        notes.append("keep TP inside one NVLink/NVSwitch node when possible")
    if pp > 1:
        notes.append("increase micro-batches to reduce pipeline bubbles")
    if ep > 1:
        notes.append("watch all-to-all bandwidth and expert load balancing")
    if dp > 1:
        notes.append("scale global batch and learning rate deliberately")
    return Plan(total_gpus, tp, pp, ep, dp, True, "; ".join(notes) or "single-process plan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-gpus", type=int, default=128)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--pp", type=int, default=4)
    parser.add_argument("--ep", type=int, default=1)
    args = parser.parse_args()
    plan = make_plan(args.total_gpus, args.tp, args.pp, args.ep)
    print(asdict(plan))


if __name__ == "__main__":
    main()
