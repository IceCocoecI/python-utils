# 01 · 分布式训练基础与 DDP

> 目标：理解分布式训练的核心概念，掌握 DDP——最常用、最成熟的数据并行方案。
> DDP 是所有更高级方案（FSDP、DeepSpeed）的基础，必须先吃透。

---

## 1. 为什么单卡不够？

### 1.1 模型越来越大

```
2018  BERT-Large       340M     →  单卡可训
2020  GPT-3            175B     →  需要几百张 GPU
2023  LLaMA-2 70B      70B      →  需要几十张 GPU
2024  LLaMA-3 405B     405B     →  需要几千张 GPU
2024  DeepSeek-V3      671B MoE →  2048 × H800
```

### 1.2 显存都花在哪了？

以 7B 模型 (bf16) 为例，全量训练的显存预算：

| 组成部分 | 计算方式 | 显存 |
|---|---|---|
| 模型参数 (bf16) | 7B × 2 bytes | 14 GB |
| 梯度 (bf16) | 7B × 2 bytes | 14 GB |
| 优化器状态 (AdamW, fp32) | 7B × (4+4+4) bytes | 84 GB |
| 激活值 (估算) | 取决于 batch/seq_len | 10–50 GB |
| **合计** | | **122–162 GB** |

一张 A100 80GB 都放不下。所以需要分布式。

本章对应可运行示例：

```bash
cd aigc-learning/05-distributed-training/examples
conda run -n aigc torchrun --standalone --nproc_per_node=2 collectives_demo.py
conda run -n aigc torchrun --standalone --nproc_per_node=2 ddp_cpu_train.py --epochs 1
```

这两个示例使用 CPU + `gloo`，用于验证 rank/world size、集合通信、`DistributedSampler`、DDP 梯度同步和 rank 0 checkpoint。文档里的 `nccl` / CUDA 示例需要真实 NVIDIA GPU 环境。

在受限沙箱中，`torchrun --standalone` 可能因为本地 TCPStore 被拦截而报 `Operation not permitted`；真实终端通常不受影响。

---

## 2. 核心概念

```
┌────────────────────────────────────────────┐
│  Node 0 (机器 0)          Node 1 (机器 1)  │
│  ┌──────┐ ┌──────┐      ┌──────┐ ┌──────┐ │
│  │GPU 0 │ │GPU 1 │      │GPU 0 │ │GPU 1 │ │
│  │rank=0│ │rank=1│      │rank=2│ │rank=3│ │
│  │local │ │local │      │local │ │local │ │
│  │rank=0│ │rank=1│      │rank=0│ │rank=1│ │
│  └──────┘ └──────┘      └──────┘ └──────┘ │
│       world_size = 4                       │
└────────────────────────────────────────────┘
```

| 术语 | 含义 |
|---|---|
| `world_size` | 参与训练的总进程数（通常 = 总 GPU 数） |
| `rank` | 当前进程的全局编号 (0 ~ world_size-1) |
| `local_rank` | 当前进程在本机的编号 (0 ~ 本机 GPU 数-1) |
| Process Group | 一组进程的通信域，支持集合通信操作 |
| Backend | 通信后端：`nccl`（GPU 首选）/ `gloo`（CPU / 备选） |

### 2.1 通信后端选择

| Backend | 适用设备 | 特点 |
|---|---|---|
| `nccl` | NVIDIA GPU | 性能最好，GPU 训练必用 |
| `gloo` | CPU / GPU | 跨平台，性能略差，调试时有用 |
| `mpi` | CPU / GPU | 需额外安装，HPC 场景用 |

> **经验法则**：GPU 训练永远用 `nccl`，只有在 CPU 环境或调试时才考虑 `gloo`。

---

## 3. DP vs DDP

### 3.1 `nn.DataParallel`（DP）—— 不推荐

```python
model = nn.DataParallel(model)
```

DP 的问题：
- **单进程多线程**，受 GIL 限制
- **GPU 0 是瓶颈**：forward scatter、loss gather 都过 GPU 0
- 负载不均衡：GPU 0 显存远大于其他卡

### 3.2 `DistributedDataParallel`（DDP）—— 推荐

```python
model = DistributedDataParallel(model, device_ids=[local_rank])
```

DDP 的优势：
- **多进程**：每个 GPU 一个独立进程，无 GIL
- **Ring All-Reduce**：梯度同步负载均匀，无瓶颈 GPU
- 通信与计算**重叠**：反向传播时边算梯度边同步

### 3.3 对比表

| | DP | DDP |
|---|---|---|
| 进程模型 | 单进程多线程 | 多进程 |
| GIL 限制 | 有 | 无 |
| 通信模式 | Scatter/Gather (GPU 0 瓶颈) | Ring All-Reduce (均衡) |
| 多机支持 | ❌ | ✅ |
| 性能 | 较差 | 接近线性加速 |
| 推荐度 | ❌ 已淘汰 | ✅ 标准方案 |

---

## 4. `torch.distributed` 基础 API

### 4.1 初始化

```python
import torch.distributed as dist

dist.init_process_group(
    backend="nccl",
    init_method="env://",   # 从环境变量读取 MASTER_ADDR, MASTER_PORT
)

rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)
```

### 4.2 常用集合通信

```python
tensor = torch.tensor([rank], dtype=torch.float32, device="cuda")

dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

dist.broadcast(tensor, src=0)

dist.barrier()

gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
dist.all_gather(gather_list, tensor)
```

| 操作 | 说明 | 典型用途 |
|---|---|---|
| `all_reduce` | 所有进程的张量求和/求平均 | 梯度同步、loss 聚合 |
| `broadcast` | 从一个进程广播到所有进程 | 参数初始化同步 |
| `all_gather` | 每个进程收集所有进程的张量 | 评估时聚合预测结果 |
| `reduce_scatter` | 规约后分片 | FSDP 内部使用 |
| `barrier` | 等待所有进程到达此处 | 同步点（如保存 checkpoint 前） |

### 4.3 清理

```python
dist.destroy_process_group()
```

---

## 5. DDP 完整示例

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def main():
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # ---- 模型 ----
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(local_rank)

    model = DDP(model, device_ids=[local_rank])

    # ---- 数据 ----
    dataset = torch.utils.data.TensorDataset(
        torch.randn(10000, 784),
        torch.randint(0, 10, (10000,)),
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # ---- 训练 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        sampler.set_epoch(epoch)    # 每个 epoch 打乱数据
        model.train()
        for x, y in loader:
            x, y = x.to(local_rank), y.to(local_rank)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()         # DDP 在这里自动同步梯度
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch} done")

    # ---- 保存（仅 rank 0）----
    if rank == 0:
        torch.save(model.module.state_dict(), "model.pt")

    cleanup()


if __name__ == "__main__":
    main()
```

### 5.1 关键细节

**① `DistributedSampler`**：保证每个 GPU 拿到不同的数据子集。

```python
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
sampler.set_epoch(epoch)   # 必须调用！否则每个 epoch 数据分配相同
```

**② `model.module`**：DDP 包裹后，原模型在 `.module` 属性下。保存/加载时用 `model.module.state_dict()`。

**③ 梯度同步是自动的**：DDP 在 `loss.backward()` 期间通过 hook 自动触发 All-Reduce。

**④ 只在 rank 0 做 I/O**：日志、checkpoint、wandb 等只在 rank 0 执行，避免重复写入和冲突。

---

## 6. 启动方式

### 6.1 `torchrun`（推荐）

```bash
# 单机 4 卡
torchrun --nproc_per_node=4 train.py

# 多机（2 台，每台 8 卡）
# 在 Node 0 上执行：
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train.py

# 在 Node 1 上执行：
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train.py
```

`torchrun` 会自动设置以下环境变量：

| 环境变量 | 说明 |
|---|---|
| `RANK` | 全局 rank |
| `LOCAL_RANK` | 本机 rank |
| `WORLD_SIZE` | 总进程数 |
| `MASTER_ADDR` | 主节点 IP |
| `MASTER_PORT` | 主节点端口 |

### 6.2 `torch.distributed.launch`（旧版，不推荐）

```bash
# 已弃用，但很多老代码还在用
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

**区别**：`torch.distributed.launch` 把 `local_rank` 作为命令行参数传递（需要 `argparse`），
而 `torchrun` 通过环境变量 `LOCAL_RANK` 传递——更简洁，不需要改代码。

### 6.3 SLURM 集群

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8

srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    train.py
```

---

## 7. 混合精度 + DDP

DDP 与 AMP（Automatic Mixed Precision）配合非常自然：

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

for x, y in loader:
    x, y = x.to(local_rank), y.to(local_rank)
    optimizer.zero_grad()

    with autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = loss_fn(model(x), y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

> **bf16 vs fp16**：如果你的 GPU 支持 bf16（A100/H100/4090），优先用 `torch.bfloat16`，
> 它的动态范围和 fp32 一样大，不需要 loss scaling。此时 `GradScaler` 可以省略。

```python
with autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = loss_fn(model(x), y)
loss.backward()
optimizer.step()
```

---

## 8. 多机训练实战

### 8.1 网络配置清单

| 检查项 | 说明 |
|---|---|
| 节点间网络连通 | `ping` / `ssh` 确认可达 |
| 防火墙端口 | 放通 `MASTER_PORT`（默认 29500） |
| NCCL 网络接口 | `export NCCL_SOCKET_IFNAME=eth0`（指定网卡） |
| InfiniBand / RoCE | 高带宽互联，大规模训练必备 |
| NFS / 共享存储 | 代码和数据需要所有节点可见 |

### 8.2 常用环境变量

```bash
# 指定 NCCL 使用的网络接口
export NCCL_SOCKET_IFNAME=eth0

# 调试 NCCL 通信
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# InfiniBand 相关
export NCCL_IB_DISABLE=0            # 启用 IB
export NCCL_IB_GID_INDEX=3          # 根据实际网络配置

# 超时设置（秒）
export NCCL_TIMEOUT=1800            # 默认 30 分钟
```

### 8.3 多机启动脚本模板

```bash
#!/bin/bash
MASTER_ADDR=192.168.1.100
MASTER_PORT=29500
NNODES=2
NODE_RANK=$1   # 启动时传入：bash run.sh 0 / bash run.sh 1

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py --lr 1e-4 --epochs 10
```

---

## 9. DDP 调试指南

### 9.1 常见错误及排查

| 错误信息 | 原因 | 解决方法 |
|---|---|---|
| `NCCL timeout` | 节点间网络不通 / 进程卡住 | 检查防火墙、网卡配置；设 `NCCL_DEBUG=INFO` |
| `Address already in use` | 端口被占用 | 换 `MASTER_PORT` 或 `kill` 旧进程 |
| `RuntimeError: Expected all tensors to be on the same device` | 数据没移到正确 GPU | 检查 `x.to(local_rank)` |
| 进程 hang（不报错但不动） | All-Reduce 等待超时、数据量不均 | 检查 `DistributedSampler`、是否有分支代码只在部分 rank 执行 |
| `Sizes of tensors must match` | 不同 rank 的 batch size 不同 | 使用 `DistributedSampler` 的 `drop_last=True` |

### 9.2 调试技巧

**① 先用 `gloo` 后端在 CPU 上跑通逻辑：**

```python
dist.init_process_group(backend="gloo")
```

**② 用 `NCCL_DEBUG=INFO` 追踪通信：**

```bash
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 train.py
```

**③ 只打 rank 0 的日志：**

```python
import logging

logger = logging.getLogger(__name__)
if dist.get_rank() != 0:
    logging.disable(logging.CRITICAL)
```

**④ 确保所有 rank 走同样的代码路径：**

```python
# 错误示范——只有 rank 0 进入 if，其他 rank 在 barrier 处等不到
if rank == 0:
    dist.barrier()

# 正确示范
dist.barrier()   # 所有 rank 都要执行
if rank == 0:
    save_checkpoint()
dist.barrier()   # 等 rank 0 存完再继续
```

### 9.3 梯度一致性检查

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_sum = param.grad.sum()
        dist.all_reduce(grad_sum, op=dist.ReduceOp.SUM)
        if rank == 0:
            expected = grad_sum / world_size
            actual = param.grad.sum()
            assert torch.allclose(actual, expected, atol=1e-5), \
                f"Gradient mismatch in {name}"
```

---

## 10. 性能优化

### 10.1 梯度分桶（Gradient Bucketing）

DDP 不会等所有梯度算完再一次性通信，而是把参数分成多个 bucket，**边算边同步**：

```
反向传播时间线：
Layer4 梯度完成 → 打包到 Bucket 0 → 开始 All-Reduce ─────→ 完成
Layer3 梯度完成 → 打包到 Bucket 1 → ··· 开始 All-Reduce ──→ 完成
Layer2 梯度完成 → 打包到 Bucket 2 → ···     ···
Layer1 梯度完成 → ...
```

```python
model = DDP(
    model,
    device_ids=[local_rank],
    bucket_cap_mb=25,          # 每个 bucket 的大小上限 (MB)
)
```

> `bucket_cap_mb` 默认 25MB。更大的 bucket 减少通信次数但增加延迟，
> 更小的 bucket 更容易重叠但增加开销。通常默认值就够了。

### 10.2 `find_unused_parameters`

如果模型的某些参数在 forward 中没用到（如多任务学习），需要开启：

```python
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
```

**注意**：这会增加开销（需要额外遍历计算图）。如果不需要，保持默认 `False`。

### 10.3 `static_graph`

如果每次 forward 的计算图结构相同（大多数情况），开启可以优化：

```python
model = DDP(model, device_ids=[local_rank], static_graph=True)
```

### 10.4 通信/计算重叠

DDP 默认就会重叠——这是它快的核心原因。但如果你在 backward 后加了 `torch.cuda.synchronize()`
或其他阻塞操作，重叠就失效了。

### 10.5 学习率线性缩放

多 GPU 时，全局 batch size = 单卡 batch × world_size。学习率通常需要线性缩放：

```python
base_lr = 1e-4
effective_lr = base_lr * world_size
```

> 配合 warmup 使用效果更好：前 N 步从小学习率线性增到 `effective_lr`。

---

## 11. DDP + 梯度累积

梯度累积时，不是每个 micro-batch 都需要同步：

```python
for i, (x, y) in enumerate(loader):
    x, y = x.to(local_rank), y.to(local_rank)

    # 只在最后一个 micro-batch 同步梯度
    is_last_step = (i + 1) % accum_steps == 0
    context = model.no_sync() if not is_last_step else nullcontext()

    with context:
        loss = loss_fn(model(x), y) / accum_steps
        loss.backward()

    if is_last_step:
        optimizer.step()
        optimizer.zero_grad()
```

`model.no_sync()` 禁止 DDP 在这次 backward 时同步梯度，省去无谓通信。

---

## 12. 常见坑总结

| 坑 | 后果 | 正确做法 |
|---|---|---|
| 忘记 `sampler.set_epoch(epoch)` | 每个 epoch 数据分配相同 | 每个 epoch 开头调用 |
| 保存 `model.state_dict()` 而非 `model.module.state_dict()` | key 带 `module.` 前缀，加载报错 | 用 `model.module.state_dict()` |
| 不同 rank 的随机种子相同 | 数据增强对每张卡完全一样 | 设 `seed + rank` |
| 所有 rank 都写日志 | 日志翻 N 倍，文件冲突 | 只在 `rank == 0` 写 |
| 在梯度累积时每步都同步 | 通信开销翻 N 倍 | 用 `model.no_sync()` |
| `find_unused_parameters=True` 滥用 | 性能下降 | 确认需要再开 |

---

## 小结

- **DDP = 每卡一个完整模型副本 + Ring All-Reduce 同步梯度**。
- 用 `torchrun` 启动，代码通过环境变量获取 `RANK` / `LOCAL_RANK` / `WORLD_SIZE`。
- `DistributedSampler` + `set_epoch()` 保证数据正确分发。
- DDP 在 `backward()` 时自动同步梯度，与计算重叠。
- 只在 `rank == 0` 做日志 / checkpoint / wandb。
- 当模型太大单卡放不下时，DDP 不够用——下一节我们学 FSDP。
