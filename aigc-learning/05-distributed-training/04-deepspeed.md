# 04 · DeepSpeed：ZeRO 优化与大模型训练

> 目标：掌握 DeepSpeed 的 ZeRO 优化——从 ZeRO-1 到 ZeRO-3 + CPU Offload，
> 理解何时选择 DeepSpeed，以及如何通过 HuggingFace 生态快速集成。

---

## 1. DeepSpeed 概览

DeepSpeed 是微软开源的大模型训练优化库，核心卖点是 **ZeRO（Zero Redundancy Optimizer）**。

核心思想：**DDP 的每张卡上都存着完全相同的优化器状态、梯度、参数——这是巨大的浪费。
ZeRO 把这些冗余分摊到多卡上。**

```bash
pip install deepspeed
```

验证安装：

```bash
ds_report
```

`ds_report` 会显示 DeepSpeed 版本、CUDA 版本、可用的 ops 等——**出问题时第一步就跑它**。

---

## 2. ZeRO 三个阶段

### 2.1 图解

```
DDP（基线）：
每卡存储：[参数 P] [梯度 G] [优化器状态 OS]   全部冗余

ZeRO-1：分片优化器状态
每卡存储：[参数 P] [梯度 G] [OS 1/N]          省 OS 显存

ZeRO-2：分片优化器 + 梯度
每卡存储：[参数 P] [G 1/N] [OS 1/N]           省 OS + G 显存

ZeRO-3：分片一切
每卡存储：[P 1/N] [G 1/N] [OS 1/N]            省全部冗余
```

### 2.2 量化对比

以 7B 模型、4 × A100 80GB、bf16 + AdamW 为例：

| 组件 | 每参数字节 | DDP（每卡） | ZeRO-1（每卡） | ZeRO-2（每卡） | ZeRO-3（每卡） |
|---|---|---|---|---|---|
| 参数 (bf16) | 2B | 14 GB | 14 GB | 14 GB | **3.5 GB** |
| 梯度 (bf16) | 2B | 14 GB | 14 GB | **3.5 GB** | **3.5 GB** |
| 优化器 (fp32 momentum + variance + master weight) | 12B | 84 GB | **21 GB** | **21 GB** | **21 GB** |
| **合计（不含激活）** | | **112 GB** | **49 GB** | **38.5 GB** | **28 GB** |
| 单卡 80GB 是否放得下 | | ❌ | ✅ (含激活可能紧) | ✅ | ✅ |

### 2.3 如何选择

| 场景 | 推荐 |
|---|---|
| 模型本身放得下，优化器放不下 | ZeRO-1 |
| 模型 + 梯度放得下，加优化器放不下 | ZeRO-2 |
| 模型太大单卡放不下 | ZeRO-3 |
| 想尽量少通信 | ZeRO-1（通信量 ≈ DDP） |
| 极端场景（训 70B+ 用几张卡） | ZeRO-3 + CPU Offload |

> **通信量递增**：ZeRO-1 ≈ DDP < ZeRO-2 < ZeRO-3。
> 除非显存逼你用高阶 ZeRO，否则优先用低阶。

---

## 3. DeepSpeed 配置详解

DeepSpeed 通过一个 JSON 配置文件驱动：

### 3.1 ZeRO-2 配置

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

### 3.2 ZeRO-3 配置

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

### 3.3 关键字段说明

| 字段 | 说明 |
|---|---|
| `stage` | ZeRO 阶段：1 / 2 / 3 |
| `overlap_comm` | 通信与计算重叠（推荐 true） |
| `allgather_bucket_size` | All-Gather 的 bucket 大小（字节） |
| `reduce_bucket_size` | Reduce-Scatter 的 bucket 大小 |
| `stage3_prefetch_bucket_size` | ZeRO-3 参数预取大小 |
| `stage3_param_persistence_threshold` | 小于此大小的参数不分片（减通信） |
| `stage3_gather_16bit_weights_on_model_save` | 保存时收集完整 fp16 权重 |
| `contiguous_gradients` | 梯度存在连续内存中（减碎片） |

---

## 4. CPU Offload

显存还是不够？把优化器状态甚至参数卸载到 CPU 内存：

### 4.1 Offload 优化器（常用）

```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

### 4.2 Offload 参数（ZeRO-3 专属）

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

### 4.3 NVMe Offload（ZeRO-Infinity）

CPU 内存也不够时，可以卸载到 NVMe SSD：

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        }
    }
}
```

### 4.4 Offload 的代价

| 方案 | 显存节省 | 速度影响 |
|---|---|---|
| 无 Offload | 基线 | 基线 |
| Offload 优化器到 CPU | 显著 | 慢 10–30% |
| Offload 参数到 CPU | 极大 | 慢 50–80% |
| Offload 到 NVMe | 极大 | 慢 2–5× |

> **经验法则**：Offload 优化器是性价比最高的。参数 Offload 只在"非它不可"时才用。

---

## 5. 集成方式

### 5.1 原生 DeepSpeed

```python
import deepspeed
import torch
import torch.nn as nn

model = nn.Linear(784, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config="ds_config.json",
)

for batch in loader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

```bash
deepspeed --num_gpus=4 train.py
```

### 5.2 通过 HuggingFace Trainer

**最简单的方式**——零代码改动，只加配置：

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="./output",
    deepspeed="ds_config.json",     # 指定 DeepSpeed 配置
    bf16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

### 5.3 通过 Accelerate

```python
from accelerate import Accelerator, DeepSpeedPlugin

ds_plugin = DeepSpeedPlugin(
    zero_stage=2,
    gradient_accumulation_steps=4,
    offload_optimizer_device="none",
)

accelerator = Accelerator(deepspeed_plugin=ds_plugin)

model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

for batch in loader:
    loss = model(batch)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

```bash
accelerate launch --use_deepspeed train.py
```

---

## 6. 激活检查点

```json
{
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": true,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false
    }
}
```

也可以在代码里配置：

```python
deepspeed.checkpointing.configure(
    mpu_=None,
    partition_activations=True,
    contiguous_checkpointing=True,
)
```

或者用 PyTorch 原生的 `torch.utils.checkpoint`，两者都兼容。

---

## 7. DeepSpeed Inference

DeepSpeed 不仅用于训练，还有推理优化：

```python
import deepspeed

model = deepspeed.init_inference(
    model,
    mp_size=2,             # 张量并行度
    dtype=torch.bfloat16,
    replace_with_kernel_inject=True,
)

output = model(input_ids)
```

**功能**：
- 自动张量并行
- 优化的 Transformer kernel
- 支持 INT8 / FP16 / BF16 推理
- 适合中等规模模型（7B–70B）的推理加速

> **注意**：对于更大规模或生产级 LLM 推理，通常用 vLLM / TensorRT-LLM 等专用引擎。

---

## 8. 调试

### 8.1 `ds_report`

```bash
ds_report
```

输出 DeepSpeed 版本、CUDA 信息、已编译的 ops。**安装出问题时第一步看它。**

### 8.2 常见错误

| 错误 | 原因 | 解决方案 |
|---|---|---|
| `ModuleNotFoundError: No module named 'deepspeed'` | 未安装或环境不对 | `pip install deepspeed` |
| `CUDA_HOME is not set` | 找不到 CUDA toolkit | `export CUDA_HOME=/usr/local/cuda` |
| `RuntimeError: NCCL timeout` | 网络或同步问题 | 见 DDP 章节调试方法 |
| `AssertionError: train_batch_size` | 全局 batch 算不对 | 用 `"auto"` 或手动算对 |
| ZeRO-3 下 `model.parameters()` 返回空张量 | 参数被分片了 | 用 `deepspeed.zero.GatheredParameters` |
| OOM despite ZeRO-3 | 激活值太大 | 加 activation checkpointing，减 seq_len |

### 8.3 ZeRO-3 参数访问

ZeRO-3 下，参数分散在多卡上。直接访问会拿到空的 placeholder：

```python
# 错误：ZeRO-3 下 weight 可能是空的
print(model.embed.weight)

# 正确：用 GatheredParameters 临时收集
with deepspeed.zero.GatheredParameters(model.embed.weight):
    print(model.embed.weight)
```

### 8.4 查看显存使用

```python
import deepspeed

deepspeed.runtime.utils.see_memory_usage("Before training", force=True)
# ... 训练 ...
deepspeed.runtime.utils.see_memory_usage("After one step", force=True)
```

---

## 9. 不同规模的配置参考

### 9.1 7B 模型（4 × A100 80GB）

```json
{
    "bf16": { "enabled": true },
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 4,
    "train_batch_size": 64
}
```

> 7B 在 4 × A100 上 ZeRO-2 足够。

### 9.2 13B 模型（4 × A100 80GB）

```json
{
    "bf16": { "enabled": true },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto"
    },
    "gradient_accumulation_steps": 8,
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 2,
    "train_batch_size": 64
}
```

> 13B 在 4 × A100 上需要 ZeRO-3，batch size 要小一些。

### 9.3 70B 模型（8 × A100 80GB）

```json
{
    "bf16": { "enabled": true },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "activation_checkpointing": {
        "partition_activations": true,
        "contiguous_memory_optimization": true
    },
    "gradient_accumulation_steps": 16,
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 1,
    "train_batch_size": 128
}
```

> 70B 在 8 × A100 上需要 ZeRO-3 + CPU Offload + Activation Checkpointing。
> 速度会比较慢——更好的方案是用更多 GPU 或 3D 并行。

---

## 10. FSDP vs DeepSpeed：决策树

```
你的模型放得下单卡吗？
├── 是 → 用 DDP，不需要 ZeRO/FSDP
└── 否 → 你需要 ZeRO/FSDP
         │
         ├── 需要 CPU/NVMe Offload？
         │   ├── 是 → DeepSpeed（更成熟）
         │   └── 否 ──┐
         │             │
         ├── 需要和 TP/PP 组合？
         │   ├── 是 → FSDP2（可组合性更好）
         │   └── 否 ──┐
         │             │
         └── 都行 → 两者功能类似，选你更熟悉的
                    PyTorch 原生偏好 → FSDP
                    HuggingFace 生态 → 两者都方便
                    已有 DeepSpeed 基础设施 → DeepSpeed
```

---

## 11. 性能调优 Tips

### 11.1 Bucket Size 调整

```json
{
    "zero_optimization": {
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    }
}
```

更大的 bucket → 更少通信次数但更高延迟，更小的 bucket → 更多通信次数但更低延迟。
一般 `5e8`（500MB）是个不错的起点。

### 11.2 通信与计算重叠

```json
{
    "zero_optimization": {
        "overlap_comm": true
    }
}
```

**始终开启**——让 All-Reduce/All-Gather 和反向传播并行执行。

### 11.3 Pin Memory

使用 CPU Offload 时，`pin_memory: true` 可以加速 CPU↔GPU 数据传输：

```json
{
    "zero_optimization": {
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

### 11.4 梯度累积

增大梯度累积步数可以减少通信次数（每 N 步才同步一次）：

```json
{
    "gradient_accumulation_steps": 8
}
```

但要注意全局 batch size 的变化对学习率的影响。

---

## 12. 完整原生训练示例

```python
import os
import argparse
import torch
import torch.nn as nn
import deepspeed
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, labels=None):
        logits = self.net(x)
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            return loss
        return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    deepspeed.init_distributed()
    rank = int(os.environ.get("RANK", 0))

    model = SimpleMLP(784, 512, 10)

    dataset = TensorDataset(
        torch.randn(10000, 784),
        torch.randint(0, 10, (10000,)),
    )
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    for epoch in range(5):
        sampler.set_epoch(epoch)
        for x, y in loader:
            x = x.to(model_engine.device)
            y = y.to(model_engine.device)
            loss = model_engine(x, labels=y)
            model_engine.backward(loss)
            model_engine.step()

        if rank == 0:
            print(f"Epoch {epoch} done")

    model_engine.save_checkpoint("./checkpoints", tag=f"epoch_{epoch}")


if __name__ == "__main__":
    main()
```

```bash
deepspeed --num_gpus=4 train_ds.py --deepspeed_config ds_config.json
```

---

## 小结

- **ZeRO-1** 分片优化器，通信量 ≈ DDP，性能影响最小。
- **ZeRO-2** 追加分片梯度，进一步省显存。
- **ZeRO-3** 连参数也分片——最省显存但通信最多。
- **CPU Offload** 是最后的杀招——用时间（速度）换空间（显存）。
- **集成方式**：原生 DeepSpeed / HuggingFace Trainer / Accelerate，三种任选。
- **配置驱动**：改 JSON 就能切换策略，不需要改训练代码。
- **选择建议**：能用低阶 ZeRO 就不用高阶，能不 Offload 就不 Offload。

下一节我们跳出"数据并行"的框架，学习更多并行策略——TP / PP / EP / SP / 3D 并行。
