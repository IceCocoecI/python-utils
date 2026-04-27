# 02 · PyTorch 完整训练循环

> 一旦你写过一个完整的训练循环，再复杂的训练脚本你都能看懂。
> 本文围绕一个最小示例（MLP on MNIST），展开所有必备模块。

---

## 1. 训练循环的七大部件

```
┌─────────────────────────────────────────────────────────┐
│ 1. Dataset       —— 定义怎么取一条样本                    │
│ 2. DataLoader    —— 把样本打 batch、多进程加载           │
│ 3. Model         —— 继承 nn.Module                       │
│ 4. Loss          —— 衡量预测 vs 真值                      │
│ 5. Optimizer     —— 更新参数（AdamW/SGD）                 │
│ 6. LR Scheduler  —— 学习率调度（warmup + cosine）         │
│ 7. Loop          —— for epoch: for batch: ...            │
└─────────────────────────────────────────────────────────┘
```

外加可选组件：AMP（混合精度）、Checkpointing、Logging、Evaluation。

---

## 2. `Dataset` 与 `DataLoader`

### 2.1 自定义 Dataset

```python
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "text": self.texts[idx],
            "label": self.labels[idx],
        }
```

三个必要方法：`__init__` / `__len__` / `__getitem__`。

### 2.2 DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

for batch in loader:
    ...
```

**关键参数**：
- `num_workers`：多进程数据加载（CPU 密集时必开）。
- `pin_memory=True`：把数据锁页到显存，加速 CPU→GPU 传输。
- `drop_last=True`：避免最后一个不完整 batch 导致 BatchNorm 报错。

### 2.3 自定义 `collate_fn`（处理变长序列）

```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    return {
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=0),
        "labels": labels,
    }

loader = DataLoader(ds, batch_size=32, collate_fn=collate_fn)
```

---

## 3. 损失函数与优化器

### 3.1 常用损失函数

| 任务 | 损失 |
|---|---|
| 分类 | `nn.CrossEntropyLoss` |
| 回归 | `nn.MSELoss` / `nn.L1Loss` / `nn.HuberLoss` |
| LLM 语言建模 | `nn.CrossEntropyLoss`（对 logits 和 labels 算） |
| 扩散模型 | `nn.MSELoss`（预测噪声） |
| 对比学习 | 自定义（InfoNCE） |

### 3.2 优化器

```python
import torch.optim as optim

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.95),
)
```

AIGC 里 99% 场景用 `AdamW`。`weight_decay` 通常 0.01～0.1。

### 3.3 LR Scheduler

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=500)
cosine = CosineAnnealingLR(optimizer, T_max=10_000, eta_min=1e-6)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[500])
```

**经验**：warmup（前 5%–10% 步数线性升温）+ cosine 衰减是业内默认配方。

---

## 4. 完整训练循环模板

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss, total_n = 0.0, 0
    for step, batch in enumerate(loader):
        x, y = batch[0].to(device), batch[1].to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
        if step % 50 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"epoch {epoch} step {step} loss={loss.item():.4f} lr={lr:.2e}")

    return {"train_loss": total_loss / total_n}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)
    return {"val_acc": correct / total}
```

### 4.1 小技巧

- `optimizer.zero_grad(set_to_none=True)` 比 `zero_grad()` 快、省显存。
- `clip_grad_norm_` 可防止梯度爆炸——LLM 训练必开（通常 1.0）。
- `scheduler.step()` 每一步都调；epoch 级 scheduler 才一个 epoch 调一次。

---

## 5. 混合精度训练（AMP）

**几乎免费的 2 倍加速 + 省一半显存。**

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler("cuda")

for x, y in loader:
    x, y = x.to(device), y.to(device)

    with autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

**规则**：
- 有 Ampere 及以上的 GPU（A100/4090/H100）—— 用 `bfloat16`，不需要 `GradScaler`。
- 老显卡（V100/T4）—— 用 `float16` + `GradScaler`。

---

## 6. 模型保存与加载

### 6.1 推荐方式：只保存 state_dict

```python
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict(),
    "config": cfg,
}, "checkpoint.pt")

ckpt = torch.load("checkpoint.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state"])
optimizer.load_state_dict(ckpt["optimizer_state"])
```

**不要**保存整个 `model` 对象——会绑定代码路径，换环境就失效。

### 6.2 safetensors（更安全、更快）

```python
from safetensors.torch import save_file, load_file

save_file(model.state_dict(), "model.safetensors")
state = load_file("model.safetensors")
model.load_state_dict(state)
```

HuggingFace 生态全面拥抱 safetensors——它不允许执行任意代码（`torch.load` 的反序列化有安全隐患）。

---

## 7. `torch.compile`：一行代码提速 20–50%

PyTorch 2.x 的明星特性：

```python
model = torch.compile(model)

model = torch.compile(model, mode="reduce-overhead")
model = torch.compile(model, mode="max-autotune")
```

首次调用会编译（几十秒），后续每次前向/反向都会被加速。
**注意**：模型里有 `print`、`if` 依赖 Tensor 值等动态控制流时会"graph break"，效果打折。

---

## 8. 分布式训练（一瞥）

### 8.1 DDP 基础模板

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

启动：`torchrun --nproc_per_node=4 train.py`。

### 8.2 Accelerate：零配置分布式

```python
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16")
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

for batch in loader:
    out = model(batch["x"])
    loss = F.cross_entropy(out, batch["y"])
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

启动：`accelerate launch train.py`（按提示回答几个问题即可）。

---

## 9. 显存预算：训练一个 N 参数模型到底要多少 GPU？

这是算法工程师**必须**会算的账。否则遇到 OOM 时完全没有头绪。

### 9.1 训练时显存的四大组成部分

训练 N 参数模型（单位：字节）：

```
总显存 ≈ 参数 + 梯度 + 优化器状态 + 激活
```

| 组件 | 占用（fp32 参数） | 占用（bf16 参数） | 说明 |
|---|---|---|---|
| 参数 | `4N` | `2N` | 模型权重本体 |
| 梯度 | `4N` | `2N` | 反向传播累积 |
| Adam 优化器状态 | `8N` | `8N`（主状态用 fp32） | m + v 两个状态 |
| Adam Master Weights（AMP） | 0 | `4N` | bf16 训练时保留的 fp32 副本 |
| **小计（静态）** | **16N** | **16N** | |
| 激活（activation） | 视模型结构而定 | | 可以是 params 的几倍 |

**7B 模型纯静态显存**：`16 × 7e9 = 112 GB`。这就是为什么你需要 A100-80G × 2 + DeepSpeed，或者 LoRA + QLoRA。

### 9.2 推理时显存

```
参数 + KV cache + 激活
= 2N (bf16) + 2 × B × T × L × H × D (bf16)
```

LLaMA-7B 推理（bs=1, seq=4096, fp16）：
- 参数：14 GB
- KV cache：~2 GB
- 激活：几百 MB
- **总共 ~16 GB**——单张 24G 卡够用。

### 9.3 省显存的武器库

| 技术 | 省显存比例 | 代价 |
|---|---|---|
| **Mixed Precision (bf16)** | ~30% | 几乎无 |
| **Gradient Checkpointing** | 60–80%（激活） | 慢 20–30% |
| **Gradient Accumulation** | 间接（小 batch 等效大 batch） | 慢 |
| **LoRA / QLoRA** | 极大（只训 <1% 参数） | 表达力略降 |
| **8-bit / 4-bit 量化** | 50% / 75% | 精度略降 |
| **FSDP / DeepSpeed ZeRO** | 线性划分到多卡 | 通信开销 |
| **CPU / NVMe Offload** | 无限（但慢） | 速度大幅下降 |

### 9.4 梯度检查点（Gradient Checkpointing）

训练时，前向计算的中间激活会被缓存以供反向传播——长序列下这是显存主要消耗。

```python
model.gradient_checkpointing_enable()

from torch.utils.checkpoint import checkpoint

def forward(self, x):
    for block in self.blocks:
        x = checkpoint(block, x, use_reentrant=False)
    return x
```

原理：只保存 block 边界的激活，block 内部反向时**重新算一次前向**。
时间换空间，长序列/大模型训练标配。

---

## 10. 调试训练的"救命技巧"

### 10.1 Loss 一直是 NaN？

按以下顺序排查：

1. **输入有 NaN / Inf**：
   ```python
   assert not torch.isnan(x).any(), f"input has NaN at batch {step}"
   ```

2. **学习率太大**：试着除以 10。

3. **softmax 输入过大**（特别是手写 attention）：忘记 `/√d_k`。

4. **log(0) / 除 0**：loss 里有 `log(prob)`，prob 需要加 `1e-8`。

5. **混合精度 fp16 溢出**：换成 `bf16` 或加 `GradScaler`。

6. **归一化层输入全相同**：`LayerNorm` 会产生 NaN（std=0）。

**终极调试代码**：

```python
def check_nan(name, tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError(f"NaN/Inf detected at {name}, range=[{tensor.min()}, {tensor.max()}]")

for name, param in model.named_parameters():
    if param.grad is not None:
        check_nan(name, param.grad)
```

### 10.2 Loss 下不去 / 不收敛？

- **先用 1 个 batch 过拟合**：训练 100 步，loss 应该接近 0。不行就是代码 bug，不是超参问题。
- **检查 `optimizer.zero_grad()` 是否漏了**。
- **检查学习率 warmup** 是否太慢——冷启 500 步还在 0 附近。
- **检查数据**：label 是不是全是 0？shape 有没有错位？
- **梯度消失/爆炸**：打印 `grad_norm`，正常范围 0.1–10。持续 < 1e-5 = 消失；> 100 = 爆炸。

### 10.3 训练慢？

1. **GPU 利用率低（`nvidia-smi` 看）**：
   - `num_workers` 不够 → 加大到 CPU 核数。
   - `pin_memory=False` → 开启。
   - 预处理太重 → 提前离线处理好存 `.arrow`。

2. **单步太慢**：
   - `torch.compile(model)` 一行加速。
   - 开混合精度 `bf16`。
   - 检查有没有 CPU-GPU 同步（`.item()` / `.cpu()` 放在循环里）。

3. **显存瓶颈**：先开梯度检查点，再考虑分布式。

### 10.4 训练诊断的"必看仪表板"

用 wandb / tensorboard 追踪以下指标，有问题一眼看出来：

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        writer.add_scalar(f"grad_norm/{name}", param.grad.norm(), step)
        writer.add_scalar(f"param_norm/{name}", param.norm(), step)

writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)
writer.add_scalar("grad_norm/total", total_norm, step)

writer.add_scalar("mem/peak_gb", torch.cuda.max_memory_allocated() / 1024**3, step)
writer.add_scalar("throughput/samples_per_sec", batch_size / step_time, step)
```

**经验**：grad_norm 曲线突然飙升 100 倍——多半是某个脏数据混进来了。

### 10.5 常见错误信息速查

| 错误 | 多半是 |
|---|---|
| `CUDA out of memory` | 显存不够——减 batch、开 checkpoint、量化 |
| `Expected all tensors to be on the same device` | 忘了 `.to(device)` |
| `RuntimeError: grad can be implicitly created only for scalar outputs` | `loss.backward()` 要求 loss 是标量 |
| `element 0 of tensors does not require grad` | 模型被 `torch.no_grad()` 包住了 |
| `CUDA error: device-side assert triggered` | **跑 CPU 模式**看真实错误（常常是 label 越界） |
| `size mismatch for ...: copying a param with shape X from checkpoint, the shape in current model is Y` | checkpoint 与模型不匹配（头部不同） |
| `Trying to backward through the graph a second time` | 忘了 `zero_grad` 或 `retain_graph=True` 用错 |

---

## 11. 工程最佳实践清单

- [ ] 随机种子统一设置：`torch.manual_seed(seed)` + `numpy.random.seed` + `random.seed`。
- [ ] 配置用 `dataclass` / `hydra` 管理，不要硬编码。
- [ ] 训练前先 **用 1 个 batch 反复过拟合**——如果连这个都拟合不到 0，说明代码有 bug。
- [ ] 打印模型参数量：`sum(p.numel() for p in model.parameters())`。
- [ ] 使用 `tensorboard` / `wandb` 跟踪 loss、grad norm、lr。
- [ ] 定期保存 checkpoint（每 N 步或每 epoch）。
- [ ] 在 validation set 上监控指标，用 best model 保存。

---

## 小结

PyTorch 训练循环的"通用骨架"：

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # forward
        loss = loss_fn(model(batch), batch.y)
        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        optimizer.step()
        scheduler.step()
    # eval
    metrics = evaluate(model, val_loader)
    save_checkpoint(...)
```

所有复杂项目（nanoGPT / transformers.Trainer / diffusers 训练脚本）本质都是这个骨架的扩展。配套示例见 `examples/mlp_mnist.py`。

下一节进入 **HuggingFace Transformers**。
