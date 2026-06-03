# 02 · FSDP：全分片数据并行

> 目标：掌握 FSDP——当模型太大单卡放不下时，PyTorch 原生的显存优化方案。
> FSDP 是 DDP 的进化版：不再每卡存完整模型，而是把参数、梯度、优化器状态都分片。

---

## 1. DDP 的显存瓶颈

DDP 的核心假设：**每张卡上有一份完整的模型副本**。

```
DDP（4 张卡）：
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ GPU 0  │ │ GPU 1  │ │ GPU 2  │ │ GPU 3  │
│ 全量参数│ │ 全量参数│ │ 全量参数│ │ 全量参数│
│ 全量梯度│ │ 全量梯度│ │ 全量梯度│ │ 全量梯度│
│ 全量优化│ │ 全量优化│ │ 全量优化│ │ 全量优化│
└────────┘ └────────┘ └────────┘ └────────┘
  每卡都一样 → 显存浪费 N 倍
```

对 7B 模型（bf16 训练，AdamW）：每卡需要 ~122GB。4 卡 DDP = 4 × 122GB = 488GB 总显存，但信息完全冗余。

---

## 2. FSDP 核心思想

FSDP = **Fully Sharded Data Parallel**：把参数、梯度、优化器状态在 N 张卡间均匀分片。

```
FSDP（4 张卡，FULL_SHARD）：
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ GPU 0  │ │ GPU 1  │ │ GPU 2  │ │ GPU 3  │
│ 参数 1/4│ │ 参数 2/4│ │ 参数 3/4│ │ 参数 4/4│
│ 梯度 1/4│ │ 梯度 2/4│ │ 梯度 3/4│ │ 梯度 4/4│
│ 优化 1/4│ │ 优化 2/4│ │ 优化 3/4│ │ 优化 4/4│
└────────┘ └────────┘ └────────┘ └────────┘
  每卡只存 1/N → 显存降 N 倍
```

**代价**：forward / backward 时需要通信来临时聚合完整参数。

### 2.1 FSDP 执行流程

```
Forward:
  1. all-gather: 从其他卡收集完整参数
  2. 执行该层的 forward
  3. 释放非本卡的参数分片（省显存）

Backward:
  1. all-gather: 再次收集完整参数
  2. 执行该层的 backward，得到完整梯度
  3. reduce-scatter: 每卡只保留自己那 1/N 梯度
  4. 释放非本卡的参数和梯度

Optimizer step:
  每卡只更新自己的 1/N 参数
```

---

## 3. 显存对比

以 7B 模型、4 × A100 80GB、bf16 训练、AdamW 为例：

| 组件 | DDP（每卡） | FSDP FULL_SHARD（每卡） |
|---|---|---|
| 参数 (bf16) | 14 GB | 3.5 GB |
| 梯度 (bf16) | 14 GB | 3.5 GB |
| 优化器状态 (fp32) | 84 GB | 21 GB |
| **合计（不含激活）** | **112 GB** ❌ | **28 GB** ✅ |

FSDP 让 4 张 A100 轻松训练 7B 模型，而 DDP 连单卡都放不下。

---

## 4. 分片策略

| 策略 | 分片内容 | 显存节省 | 通信量 | 适用场景 |
|---|---|---|---|---|
| `FULL_SHARD` | 参数 + 梯度 + 优化器 | 最大 | 最大 | 模型大、显存紧张 |
| `SHARD_GRAD_OP` | 梯度 + 优化器 | 中等 | 中等 | 模型能放下但优化器放不下 |
| `NO_SHARD` | 无（等价于 DDP） | 无 | 最小 | 调试、对比基线 |

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 默认
)
```

> **经验法则**：先试 `FULL_SHARD`；如果通信是瓶颈（如 PCIe 互联），
> 降级到 `SHARD_GRAD_OP` 可能更快。

---

## 5. FSDP API 详解

### 5.1 基本使用

```python
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = MyLargeModel()
model = FSDP(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in loader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 5.2 自动包裹策略（Auto Wrap Policy）

FSDP 需要知道在哪些层级做分片。手动包裹每一层太麻烦，推荐用自动策略：

```python
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
import functools

# 方式 1：按参数量——超过阈值的模块单独分片
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=1_000_000,   # 超过 1M 参数的子模块单独 FSDP wrap
)

# 方式 2：按层类型——Transformer 模型推荐
from my_model import TransformerBlock

auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)

model = FSDP(model, auto_wrap_policy=auto_wrap_policy)
```

> **推荐**：LLM / Diffusion Model 用 `transformer_auto_wrap_policy`，
> 把每个 Transformer Block 作为一个 FSDP 单元。

### 5.3 混合精度

```python
from torch.distributed.fsdp import MixedPrecision

bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,    # 通信精度
    buffer_dtype=torch.bfloat16,
)

model = FSDP(model, mixed_precision=bf16_policy)
```

| 字段 | 说明 |
|---|---|
| `param_dtype` | forward/backward 时参数的精度 |
| `reduce_dtype` | 梯度 All-Reduce 时的精度 |
| `buffer_dtype` | Buffer（如 BatchNorm 的 running_mean）的精度 |

---

## 6. FSDP2（PyTorch 2.4+）

PyTorch 正在推出新一代 FSDP，称为 FSDP2（`torch.distributed._composable.fsdp`）。

### 6.1 FSDP2 vs FSDP1 的区别

| 特性 | FSDP1 | FSDP2 |
|---|---|---|
| API 风格 | Module 包裹 (`FSDP(model)`) | 可组合 (`fully_shard(model)`) |
| 分片粒度 | 按模块 | 按参数 |
| 与其他并行组合 | 复杂 | 原生支持（TP + FSDP 等） |
| 状态字典 | 需特殊 API | 直接用 `model.state_dict()` |
| 成熟度 | 稳定 | 快速迭代中 |

### 6.2 FSDP2 基本用法

```python
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)

for layer in model.layers:
    fully_shard(layer, mp_policy=mp_policy)

fully_shard(model, mp_policy=mp_policy)
```

**优势**：

- 不需要 `auto_wrap_policy`——直接对需要分片的子模块调用 `fully_shard`
- 与 Tensor Parallel 等其他并行方案可以自然组合
- `model.state_dict()` 直接返回完整的 state dict，不需要特殊 API

> **建议**：新项目可以尝试 FSDP2；已有项目如果 FSDP1 跑得好，暂时不需要迁移。

---

## 7. 激活检查点（Activation Checkpointing）

FSDP 省了参数 / 梯度 / 优化器的显存，但激活值仍然占大头。
激活检查点用"时间换空间"：前向时丢弃中间激活，反向时重新计算。

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing,
)
import functools

check_fn = lambda submodule: isinstance(submodule, TransformerBlock)

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=check_fn,
)
```

或者用 PyTorch 2.4+ 的新 API：

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x):
        return checkpoint(self._forward_impl, x, use_reentrant=False)

    def _forward_impl(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

> **经验**：激活检查点一般让显存减半、训练速度下降 20–30%。大模型训练几乎都开启。

---

## 8. Checkpoint 保存与加载

### 8.1 Full State Dict（推荐简单场景）

```python
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    state_dict = model.state_dict()
    if dist.get_rank() == 0:
        torch.save(state_dict, "model.pt")
```

**特点**：rank 0 收集完整模型，保存为普通 state_dict。加载时和单卡一样。

### 8.2 Sharded State Dict（推荐大模型）

```python
from torch.distributed.checkpoint import save, load
from torch.distributed.checkpoint import FileSystemWriter, FileSystemReader

writer = FileSystemWriter("checkpoint_dir")
save(model.state_dict(), writer)

reader = FileSystemReader("checkpoint_dir")
state_dict = model.state_dict()
load(state_dict, reader)
model.load_state_dict(state_dict)
```

**特点**：每个 rank 保存自己的分片，保存/加载速度快，适合 70B+ 模型。
注意：加载时的 GPU 数量可以和保存时不同——`torch.distributed.checkpoint` 会自动 re-shard。

### 8.3 FSDP2 的 Checkpoint

FSDP2 的优势之一是 checkpoint 更简单：

```python
import torch.distributed.checkpoint as dcp

dcp.save(model.state_dict(), checkpoint_id="checkpoint_dir")
dcp.load(model.state_dict(), checkpoint_id="checkpoint_dir")
```

---

## 9. 完整训练示例

```python
import os
import functools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    model = SimpleTransformer(vocab_size=32000, d_model=1024, n_layers=24)

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    bf16_mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=wrap_policy,
        mixed_precision=bf16_mp,
        device_id=local_rank,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    dataset = torch.utils.data.TensorDataset(
        torch.randint(0, 32000, (5000, 512)),
    )
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=4, sampler=sampler)

    for epoch in range(3):
        sampler.set_epoch(epoch)
        model.train()
        for (tokens,) in loader:
            tokens = tokens.to(local_rank)
            logits = model(tokens)
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                tokens[:, 1:].reshape(-1),
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if rank == 0:
            print(f"Epoch {epoch} complete")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

```bash
torchrun --nproc_per_node=4 train_fsdp.py
```

---

## 10. 常见坑与调试

### 10.1 常见错误

| 问题 | 原因 | 解决方案 |
|---|---|---|
| OOM 但已用 FSDP | 激活值太大 | 加 activation checkpointing、减 batch size |
| `RuntimeError: inconsistent tensor size` | wrap policy 不一致 | 确保所有 rank 的模型结构相同 |
| 保存的 checkpoint 加载报错 | state_dict 类型不匹配 | 保存和加载用同一种 `StateDictType` |
| 训练 loss 不收敛 | 混合精度溢出 | 检查 `reduce_dtype`，尝试用 bf16 代替 fp16 |
| 速度比 DDP 慢 | 通信开销大于省显存收益 | 模型不大时直接用 DDP |

### 10.2 调试技巧

```python
# 打印 FSDP 包裹结构
if dist.get_rank() == 0:
    print(model)

# 检查每卡显存
torch.cuda.reset_peak_memory_stats()
# ... 跑一个 step ...
peak_mem = torch.cuda.max_memory_allocated() / 1024**3
print(f"Rank {dist.get_rank()} peak memory: {peak_mem:.2f} GB")
```

---

## 11. FSDP vs DeepSpeed：如何选择

| 维度 | FSDP | DeepSpeed ZeRO |
|---|---|---|
| 生态 | PyTorch 原生 | 微软维护，独立库 |
| 集成难度 | 简单（PyTorch 原生 API） | 需要配置 JSON + launcher |
| CPU Offload | 支持（但文档较少） | 成熟且高效 |
| NVMe Offload | 不支持 | 支持（ZeRO-Infinity） |
| HuggingFace 集成 | Trainer + Accelerate 均支持 | Trainer + Accelerate 均支持 |
| 推理优化 | 不关注 | DeepSpeed Inference |
| 社区活跃度 | 高（PyTorch 核心团队） | 高（微软团队） |

> **简单选择**：
> - PyTorch 原生主义者 / 新项目 → FSDP
> - 需要 CPU/NVMe Offload / 已有 DeepSpeed 经验 → DeepSpeed
> - 用 HuggingFace 生态 → 两个都方便，按需切换

---

## 小结

- **FSDP = 把参数 + 梯度 + 优化器状态在 N 张卡间分片**。
- 三种策略：`FULL_SHARD`（省最多）、`SHARD_GRAD_OP`（折中）、`NO_SHARD`（= DDP）。
- 用 `auto_wrap_policy` 自动决定分片粒度，Transformer 模型推荐按 block 分。
- 激活检查点是大模型训练的标配，和 FSDP 配合使用。
- Checkpoint 可以用 full state dict（简单）或 sharded state dict（高效）。
- FSDP2 是新一代 API，更简洁、可组合性更好。

下一节我们学 HuggingFace Accelerate——用最少的代码改动实现分布式训练。
