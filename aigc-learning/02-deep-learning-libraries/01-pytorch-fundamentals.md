# 01 · PyTorch 基础

> 目标：吃透 Tensor、autograd、nn.Module、device 这四件事。
> 它们占了 PyTorch 所有用法的 80%。

---

## 1. Tensor：一切的起点

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
randn = torch.randn(3, 4)
arange = torch.arange(10)
linspace = torch.linspace(0, 1, steps=5)

a = torch.tensor([1, 2, 3])
b = torch.tensor([1.0, 2.0, 3.0])
print(a.dtype, b.dtype)
```

### 1.1 形状、数据类型、设备：Tensor 的三要素

```python
x = torch.randn(2, 3, 4)
print(x.shape, x.dtype, x.device)

x = x.float()
x = x.to(torch.bfloat16)
x = x.to("cuda")
```

### 1.2 基础运算

```python
x = torch.arange(12).reshape(3, 4).float()

y = x + 1
y = x * 2
y = x.sum(dim=0)
y = x.mean(dim=1, keepdim=True)
y = x.max(dim=1).values

a = torch.randn(3, 4)
b = torch.randn(4, 5)
c = a @ b
```

### 1.3 形状操作（AIGC 代码中极其频繁）

```python
x = torch.randn(2, 3, 4, 5)

x.reshape(2, 60)
x.view(2, 60)
x.permute(0, 2, 1, 3)
x.transpose(-2, -1)

x.squeeze()
x.unsqueeze(0)

a = torch.randn(3, 4)
b = torch.randn(3, 4)
torch.cat([a, b], dim=0)
torch.stack([a, b], dim=0)
```

> **强烈推荐**：这些操作用 `einops` 会清晰 10 倍——见模块 03。

### 1.4 广播（Broadcasting）

广播是 PyTorch 最核心的机制之一：**自动扩展形状较小的张量来匹配较大的**。

```python
x = torch.randn(5, 3)
m = torch.randn(3)
y = x - m
```

规则（从右往左对齐）：
1. 每一维要么相等，要么有一个为 1，要么缺失。
2. 如果某维为 1，会被"虚拟扩展"到另一边的尺寸。

### 1.5 NumPy 互操作

```python
import numpy as np

t = torch.randn(2, 3)
arr = t.numpy()

t2 = torch.from_numpy(arr)
```

注意：`numpy()` 和 `from_numpy()` **共享底层内存**，修改一个会影响另一个。GPU Tensor 不能直接转 numpy，需先 `.cpu()`。

---

## 2. Autograd：自动微分

这是 PyTorch 的灵魂——你只需写前向计算，反向传播是自动的。

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1

y.backward()

print(x.grad)
```

### 2.1 计算图是动态构建的

```python
w = torch.randn(3, 4, requires_grad=True)
b = torch.randn(4, requires_grad=True)
x = torch.randn(5, 3)

y = x @ w + b
loss = y.sum()

loss.backward()

print(w.grad.shape)
print(b.grad.shape)
```

### 2.2 禁用梯度（推理时必备）

```python
with torch.no_grad():
    predictions = model(x)

@torch.no_grad()
def infer(model, x):
    return model(x)
```

**为什么重要**：
- 推理时省一半显存（不用保存中间激活）。
- 加速 30%+（不记录计算图）。

### 2.3 梯度累积（小显存训练大 batch）

```python
optimizer.zero_grad()
for i, (x, y) in enumerate(loader):
    out = model(x)
    loss = loss_fn(out, y) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 3. `nn.Module`：模型的基本单位

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MLP(784, 256, 10)
x = torch.randn(32, 784)
out = model(x)
```

### 3.1 `nn.Module` 帮你做了什么

- 自动收集所有子模块的参数（`.parameters()`）
- 自动管理设备（`.to(device)` 递归生效）
- 自动管理 train/eval 模式（`.train()` / `.eval()`）
- 支持 state_dict 保存加载

### 3.2 常用层

| 层 | 用途 |
|---|---|
| `nn.Linear` | 全连接层 |
| `nn.Conv2d` / `nn.Conv3d` | 卷积层（视觉/视频） |
| `nn.LayerNorm` / `nn.BatchNorm2d` / `nn.RMSNorm` | 归一化层 |
| `nn.Embedding` | 词/token 嵌入 |
| `nn.MultiheadAttention` | 多头注意力 |
| `nn.TransformerEncoderLayer` | Transformer 块 |
| `nn.Dropout` | 随机失活 |

### 3.3 自己写一个简化版 Self-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in qkv]
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(attn)
```

`F.scaled_dot_product_attention` 是 PyTorch 2.x 内置的 FlashAttention 实现——**直接用它，不要手写 softmax**。

---

## 4. Device：CPU / CUDA / MPS

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)
x = x.to(device)

torch.cuda.synchronize()
torch.cuda.empty_cache()
print(torch.cuda.max_memory_allocated() / 1024**3, "GB")
```

### 4.1 推荐姿势：一次定义，到处使用

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

model = MLP(784, 256, 10).to(device=DEVICE, dtype=DTYPE)
```

### 4.2 GPU 选择

```python
torch.cuda.device_count()
torch.cuda.set_device(0)
torch.cuda.current_device()

x = x.to("cuda:0")
y = y.to("cuda:1")
```

---

## 5. 常见坑

### 5.1 忘记 `.to(device)`

错误：`RuntimeError: Expected all tensors to be on the same device`。
**原则**：模型和数据必须都在同一个设备上。

### 5.2 原地操作破坏计算图

```python
x = torch.randn(3, requires_grad=True)
y = x + 1
y.add_(1)
```

AIGC 代码里常见的是 `x.detach_()`、`x.zero_()`。慎用。

### 5.3 梯度没清零

```python
for batch in loader:
    loss = loss_fn(model(batch.x), batch.y)
    loss.backward()
    optimizer.step()
```

梯度会累加到上一次。必须在 `loss.backward()` 前 `optimizer.zero_grad()`。

### 5.4 shape 不一致（最常见的报错）

训练时把 `print(x.shape)` 写在各个关键节点，能帮你快速定位。

---

## 6. 推荐练手

1. 手动实现 Linear 层（不用 `nn.Linear`，只用 `torch.Tensor`）。
2. 手动实现 LayerNorm、RMSNorm。
3. 手动实现 Multi-Head Attention（先用 for 循环写清楚，再用 reshape 提速）。
4. 跟着 [nanoGPT](https://github.com/karpathy/nanoGPT) 的 `model.py` 从头读一遍——~300 行。

---

## 小结

- Tensor = 形状 + dtype + device。
- `requires_grad=True` 让 PyTorch 记录计算图，`.backward()` 触发反传。
- `with torch.no_grad():` 推理必备。
- 继承 `nn.Module`，在 `__init__` 注册子模块，`forward` 写前向逻辑。
- Flash Attention 用 `F.scaled_dot_product_attention`。

下一节进入**完整训练循环**——真正跑起来一个模型。
