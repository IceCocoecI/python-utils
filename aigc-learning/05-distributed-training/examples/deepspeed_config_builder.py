"""Generate small DeepSpeed ZeRO config files.

This is useful even when DeepSpeed is not installed because it validates the
configuration shape described in the docs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_config(stage: int, offload: bool) -> dict:
    zero = {
        "stage": stage,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
    }
    if stage >= 2:
        zero["reduce_scatter"] = True
    if stage == 3:
        zero.update(
            {
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_gather_16bit_weights_on_model_save": True,
            }
        )
    if offload:
        zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        if stage == 3:
            zero["offload_param"] = {"device": "cpu", "pin_memory": True}

    return {
        "bf16": {"enabled": True},
        "zero_optimization": zero,
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/deepspeed/ds_config.json"))
    args = parser.parse_args()

    cfg = build_config(args.stage, args.offload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(cfg, indent=2, sort_keys=True))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
