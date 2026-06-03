# 03 · HuggingFace Accelerate：一行代码切换分布式

> 目标：用 Accelerate 以最小的代码改动，让你的单卡训练脚本跑在多卡、多机、
> 混合精度、DeepSpeed、FSDP 上——"write once, run everywhere"。

---

## 1. 为什么用 Accelerate？

直接用 `torch.distributed` 写分布式训练代码，需要：

- 手动 `init_process_group`
- 手动 `DistributedSampler`
- 手动 `DDP(model)` / `FSDP(model)`
- 手动处理设备放置
- 手动混合精度
- 手动梯度累积中的 `no_sync`
- 不同后端（DDP / FSDP / DeepSpeed）切换要大改代码

**Accelerate 的哲学**：你只需写一份普通的 PyTorch 训练脚本，
Accelerate 帮你处理所有分布式细节。切换后端只需改配置，不改代码。

```
                你的训练脚本
                    │
           ┌────────┴────────┐
           │   Accelerator   │
           └────────┬────────┘
        ┌───────┬───┴───┬────────┐
        │       │       │        │
      单卡    DDP    FSDP   DeepSpeed
```

---

## 2. 安装

```bash
pip install accelerate
```

---

## 3. 快速入门：从单卡到分布式

### 3.1 原始单卡脚本

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

model = nn.Linear(784, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loader = DataLoader(dataset, batch_size=64)

device = "cuda"
model = model.to(device)

for epoch in range(10):
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = nn.functional.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 3.2 用 Accelerate 改写（只需改 5 行）

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator                        # 1. 导入

accelerator = Accelerator()                               # 2. 创建

model = nn.Linear(784, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loader = DataLoader(dataset, batch_size=64)

model, optimizer, loader = accelerator.prepare(           # 3. prepare
    model, optimizer, loader
)

for epoch in range(10):
    for x, y in loader:                                   # 4. 不需要手动 .to(device)
        loss = nn.functional.cross_entropy(model(x), y)
        optimizer.zero_grad()
        accelerator.backward(loss)                        # 5. 用 accelerator.backward
        optimizer.step()
```

**就这些。** 这段代码可以跑在单卡、多卡 DDP、FSDP、DeepSpeed 上——不改任何代码。

---

## 4. `accelerate config`：交互式配置

```bash
accelerate config
```

会引导你完成一系列选择：

```
In which compute environment are you running? [0] This machine
Which type of machine are you using? [0] No distributed training / [1] multi-GPU / [2] multi-node
How many GPUs? 4
Do you want to use DeepSpeed? [yes/NO]
Do you want to use FSDP? [yes/NO]
Do you want to use mixed precision? [NO/fp16/bf16]
```

配置保存在 `~/.cache/huggingface/accelerate/default_config.yaml`：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1
num_processes: 4
mixed_precision: bf16
```

也可以手动创建 YAML 文件：

```bash
accelerate config --config_file my_config.yaml
```

---

## 5. 核心 API 详解

### 5.1 `Accelerator`

```python
accelerator = Accelerator(
    mixed_precision="bf16",          # "no", "fp16", "bf16"
    gradient_accumulation_steps=4,   # 梯度累积步数
    log_with="wandb",                # 日志集成
    project_dir="./outputs",
)
```

### 5.2 `prepare()`

```python
model, optimizer, train_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, scheduler
)
```

`prepare` 做了什么：

| 对象 | 处理 |
|---|---|
| Model | 包裹为 DDP / FSDP / DeepSpeed 模型，移到正确设备 |
| Optimizer | 适配分布式（DeepSpeed 会替换为 fused optimizer） |
| DataLoader | 自动加 `DistributedSampler`，处理 batch 分发 |
| Scheduler | 自动适配梯度累积步数 |

### 5.3 `backward()`

```python
accelerator.backward(loss)
```

对比直接 `loss.backward()`：
- 自动处理混合精度的 loss scaling
- 自动处理梯度累积（只在累积完成时同步梯度）
- DeepSpeed 需要用 `accelerator.backward` 而不是 `loss.backward`

### 5.4 `unwrap_model()`

```python
unwrapped = accelerator.unwrap_model(model)
state_dict = unwrapped.state_dict()
```

`prepare` 之后模型被包裹了，保存时需要先解包。

### 5.5 设备相关

```python
accelerator.device              # 当前设备
accelerator.is_main_process     # 是否主进程（用于日志/保存）
accelerator.is_local_main_process  # 本机主进程
accelerator.num_processes       # 总进程数
accelerator.process_index       # 当前进程号
```

---

## 6. 梯度累积

不需要手动写 `no_sync` 逻辑：

```python
accelerator = Accelerator(gradient_accumulation_steps=4)

for batch in loader:
    with accelerator.accumulate(model):    # 自动处理 no_sync
        loss = model(batch)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

`accumulate` 上下文管理器自动在中间步骤跳过梯度同步，只在第 4 步（累积完成时）同步。

---

## 7. 混合精度

```python
accelerator = Accelerator(mixed_precision="bf16")
```

配合 `prepare` 后自动生效——不需要手动写 `autocast` 或 `GradScaler`。

如果你需要手动 autocast（如在自定义 loss 函数中）：

```python
with accelerator.autocast():
    loss = custom_loss_fn(model_output, targets)
```

---

## 8. 梯度裁剪

```python
accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
# 或
accelerator.clip_grad_value_(model.parameters(), clip_value=0.5)
```

**注意**：必须在 `optimizer.step()` 之前、`accelerator.backward()` 之后调用。

---

## 9. 多 GPU / 多机启动

```bash
# 单机多卡（使用 default_config.yaml）
accelerate launch train.py

# 单机多卡（命令行指定）
accelerate launch --num_processes=4 --mixed_precision=bf16 train.py

# 多机（2 台，每台 8 卡）
# Node 0:
accelerate launch \
    --num_machines=2 \
    --num_processes=16 \
    --machine_rank=0 \
    --main_process_ip=192.168.1.100 \
    --main_process_port=29500 \
    train.py

# Node 1:
accelerate launch \
    --num_machines=2 \
    --num_processes=16 \
    --machine_rank=1 \
    --main_process_ip=192.168.1.100 \
    --main_process_port=29500 \
    train.py
```

也可以直接用 `torchrun`（Accelerate 兼容 torch 的环境变量）：

```bash
torchrun --nproc_per_node=4 train.py
```

---

## 10. DeepSpeed 集成

### 10.1 通过配置文件

创建 `ds_config.json`：

```json
{
    "bf16": { "enabled": true },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": { "device": "none" },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto"
}
```

```bash
accelerate launch --use_deepspeed --deepspeed_config_file ds_config.json train.py
```

### 10.2 通过 Accelerator 参数

```python
from accelerate import Accelerator, DeepSpeedPlugin

ds_plugin = DeepSpeedPlugin(
    zero_stage=2,
    gradient_accumulation_steps=4,
    gradient_clipping=1.0,
    offload_optimizer_device="none",
    offload_param_device="none",
)

accelerator = Accelerator(deepspeed_plugin=ds_plugin)
```

### 10.3 `"auto"` 字段

在 DeepSpeed 配置 JSON 中，设为 `"auto"` 的字段会由 Accelerate 自动填充：

| 字段 | 来源 |
|---|---|
| `train_batch_size` | `batch_size × num_processes × grad_accum_steps` |
| `train_micro_batch_size_per_gpu` | DataLoader 的 `batch_size` |
| `gradient_accumulation_steps` | Accelerator 构造参数 |

这样你不需要手动计算这些值。

---

## 11. FSDP 集成

```python
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import ShardingStrategy

fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy="transformer_based_wrap",
    transformer_cls_names_to_wrap=["TransformerBlock"],
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
```

或通过配置：

```bash
accelerate launch --use_fsdp \
    --fsdp_sharding_strategy=FULL_SHARD \
    --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=TransformerBlock \
    train.py
```

---

## 12. 与 HuggingFace Trainer 的关系

```
Trainer
  └── 内部使用 Accelerate
        └── 内部使用 torch.distributed / DeepSpeed / FSDP

Accelerate = "中间层"
Trainer = "最高层抽象"
```

- **Trainer** 用户：不需要直接用 Accelerate，Trainer 内部会自动处理。
  只需在 `TrainingArguments` 中配置 `deepspeed` / `fsdp` 字段即可。
- **自定义训练循环** 用户：用 Accelerate 是最佳选择——保留训练循环的灵活性，同时获得分布式支持。

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    bf16=True,
    fsdp="full_shard",
    fsdp_config={
        "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
    },
    # 或者用 DeepSpeed：
    # deepspeed="ds_config.json",
)
```

---

## 13. 日志与追踪集成

```python
accelerator = Accelerator(log_with="wandb")

accelerator.init_trackers(
    project_name="my-project",
    config={"lr": 1e-4, "epochs": 10},
)

for step, batch in enumerate(loader):
    loss = train_step(batch)
    accelerator.log({"loss": loss.item(), "step": step}, step=step)

accelerator.end_training()
```

支持的 tracker：`wandb`、`tensorboard`、`comet_ml`、`mlflow`。

> 只在主进程记录，Accelerate 自动处理——你不需要手写 `if rank == 0`。

---

## 14. 完整训练脚本

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from accelerate.utils import set_seed


def main():
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=4,
        log_with="tensorboard",
        project_dir="./logs",
    )

    set_seed(42)

    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    dataset = TensorDataset(
        torch.randn(10000, 784),
        torch.randint(0, 10, (10000,)),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )

    accelerator.init_trackers("example", config={"lr": 1e-3})

    for epoch in range(10):
        model.train()
        total_loss = 0.0

        for step, (x, y) in enumerate(loader):
            with accelerator.accumulate(model):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            if step % 50 == 0:
                accelerator.log({"train_loss": loss.item()}, step=epoch * len(loader) + step)

        scheduler.step()
        avg_loss = total_loss / len(loader)
        accelerator.print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), "model.pt")

    accelerator.end_training()


if __name__ == "__main__":
    main()
```

```bash
# 单卡
python train.py

# 多卡
accelerate launch --num_processes=4 --mixed_precision=bf16 train.py
```

---

## 15. 常见坑

### 15.1 `prepare` 之后才能访问 `accelerator.device`

```python
# 错误：prepare 前把数据搬到设备
x = x.to(accelerator.device)       # 此时 device 可能还没确定
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

# 正确：prepare 后 loader 自动搬数据到正确设备
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
for x, y in loader:                 # x, y 已经在正确设备上
    ...
```

### 15.2 保存时忘记 `unwrap_model`

```python
# 错误
torch.save(model.state_dict(), "model.pt")  # 包含 DDP/FSDP 前缀

# 正确
unwrapped = accelerator.unwrap_model(model)
torch.save(unwrapped.state_dict(), "model.pt")

# 或者用 accelerator 的 save 方法
accelerator.save_model(model, "model_dir")
```

### 15.3 评估时收集所有进程的预测

```python
all_predictions = []
all_labels = []

for batch in eval_loader:
    with torch.no_grad():
        outputs = model(batch["input_ids"])
    predictions = outputs.argmax(dim=-1)

    predictions, labels = accelerator.gather_for_metrics(
        (predictions, batch["labels"])
    )
    all_predictions.append(predictions)
    all_labels.append(labels)
```

`gather_for_metrics` 自动处理 padding（`DistributedSampler` 可能补齐最后一个 batch）。

### 15.4 DeepSpeed ZeRO-3 下不能直接访问模型参数

```python
# ZeRO-3 下参数被分片，直接访问是空的
print(model.linear.weight)  # 可能全是 0

# 正确：用 gather 上下文
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
```

### 15.5 `accelerator.print` vs `print`

```python
print("这会在每个进程打一次——4 卡就打 4 次")
accelerator.print("这只在主进程打一次")
```

---

## 小结

| API | 用途 |
|---|---|
| `Accelerator()` | 创建分布式上下文 |
| `accelerator.prepare(...)` | 包裹模型/优化器/数据 |
| `accelerator.backward(loss)` | 替代 `loss.backward()` |
| `accelerator.accumulate(model)` | 梯度累积上下文 |
| `accelerator.unwrap_model(model)` | 解包模型（保存时用） |
| `accelerator.is_main_process` | 判断主进程 |
| `accelerator.print()` | 只在主进程打印 |
| `accelerate launch` | 启动分布式训练 |
| `accelerate config` | 交互式配置 |

**一句话**：**Accelerate 是 PyTorch 分布式训练的最佳"胶水层"**——
如果你不用 Trainer，用 Accelerate 几乎是必选项。

下一节我们学 DeepSpeed——显存优化的终极武器。
