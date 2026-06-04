# Module 05 Examples

The examples are designed for the current `aigc` environment. They use synthetic
data and can be run without downloading datasets.

CPU-distributed examples:

```bash
conda run -n aigc torchrun --standalone --nproc_per_node=2 collectives_demo.py
conda run -n aigc torchrun --standalone --nproc_per_node=2 ddp_cpu_train.py --epochs 1
conda run -n aigc accelerate launch --cpu --num_processes=2 accelerate_train.py --epochs 1
```

Runnable analysis/config examples:

```bash
conda run -n aigc python fsdp_memory_math.py --params-billion 7 --world-size 4
conda run -n aigc python deepspeed_config_builder.py --stage 3 --offload
conda run -n aigc python parallelism_planner.py --total-gpus 128 --tp 8 --pp 4
```

Current environment notes:

- PyTorch distributed is available.
- CUDA build exists, but `torch.cuda.is_available()` is false on this machine.
- `accelerate` is installed.
- `deepspeed` is not installed, so the DeepSpeed example generates config only.
