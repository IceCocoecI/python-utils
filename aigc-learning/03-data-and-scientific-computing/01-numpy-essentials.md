# 01 · NumPy 精要

> PyTorch 的 Tensor API 几乎和 NumPy 一一对应，吃透 NumPy 就等于吃透了一半 PyTorch。
> 本节只讲 AIGC 工程师最常用的 20% 功能——它们覆盖 80% 的场景。

---

## 1. ndarray：核心数据结构

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([[1, 2, 3], [4, 5, 6]])

print(a.shape, a.dtype, a.ndim)
print(b.shape, b.dtype, b.ndim)
```

### 1.1 常用创建方式

```python
np.zeros((3, 4))
np.ones((2, 3), dtype=np.float32)
np.full((2, 3), fill_value=7)
np.arange(10)
np.linspace(0, 1, 5)
np.eye(3)
np.random.randn(3, 4)
np.random.rand(3, 4)
np.random.randint(0, 10, size=(3,))
```

### 1.2 dtype 尤其重要

```python
arr = np.array([1, 2, 3])
arr.dtype

arr = np.array([1, 2, 3], dtype=np.float32)

arr2 = arr.astype(np.int64)
```

AIGC 里典型的 dtype：
- 图像：`np.uint8` [0, 255]
- 深度学习输入：`np.float32` [0, 1]
- 标签：`np.int64`

---

## 2. 切片与索引

### 2.1 基础切片

```python
a = np.arange(24).reshape(2, 3, 4)

a[0]
a[0, 1]
a[0, 1, 2]
a[:, :, 0]
a[..., 0]
a[0:1, :, 1:3]
```

### 2.2 布尔索引

```python
a = np.array([1, -2, 3, -4, 5])
a[a > 0]
a[a < 0] = 0
```

### 2.3 整数数组索引（fancy indexing）

```python
a = np.array([10, 20, 30, 40, 50])
idx = np.array([0, 2, 4])
print(a[idx])

one_hot = np.eye(5)[np.array([0, 1, 2, 3])]
```

### 2.4 **关键**：切片是视图，fancy 索引是复制

```python
a = np.arange(10)
b = a[2:5]
b[0] = 999
print(a)

c = a[[2, 3, 4]]
c[0] = 0
print(a)
```

这个差异跟 PyTorch Tensor 完全一致，不理解会写出灾难性的 bug。

---

## 3. 形状操作

```python
a = np.arange(24)

a.reshape(2, 3, 4)
a.reshape(2, -1)

b = a.reshape(2, 3, 4)
b.transpose(0, 2, 1)
b.T

x = np.arange(10)
x.reshape(-1, 1)
x[:, None]
x[None, :]

y = np.array([[[1]], [[2]]])
y.squeeze()
```

> **推荐**：对复杂形状操作一律用 einops（见下一节），代码更清晰。

---

## 4. 广播（Broadcasting）

**规则**（从右往左对齐）：
1. 每一维要么相等，要么有一个为 1，要么缺失。
2. 缺失的维度视作 1。

```python
x = np.random.randn(32, 3, 224, 224)
mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

x_norm = (x - mean) / std
```

这就是 ImageNet 归一化的标准写法。

```python
a = np.arange(12).reshape(3, 4)
b = np.arange(4)
print((a + b).shape)

c = np.arange(3).reshape(3, 1)
print((a + c).shape)
```

---

## 5. 向量化：永远不要手写 for 循环

### 5.1 反例 vs 正例

```python
result = []
for i in range(len(x)):
    result.append(x[i] ** 2 + 3 * x[i])

result = x ** 2 + 3 * x
```

**NumPy 用 C 实现，向量化版本比 Python for 循环快 10–100 倍。**

### 5.2 常用归约操作

```python
a = np.random.randn(3, 4, 5)

a.sum()
a.sum(axis=0)
a.sum(axis=(0, 1))

a.mean(axis=-1, keepdims=True)
a.max(axis=1)
a.argmax(axis=1)

x = np.array([1, 2, 3, 4])
np.cumsum(x)
```

### 5.3 条件/掩码

```python
a = np.random.randn(10)
np.where(a > 0, a, 0)
np.clip(a, -1, 1)

(a > 0).all()
(a > 0).any()
np.logical_and(a > -1, a < 1)
```

### 5.4 数学函数

```python
np.exp(x)
np.log(x)
np.sqrt(x)
np.sin(x)
np.abs(x)

softmax = np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)
```

---

## 6. 矩阵运算

```python
a = np.random.randn(3, 4)
b = np.random.randn(4, 5)

c = a @ b
c = np.matmul(a, b)

a = np.random.randn(8, 3, 4)
b = np.random.randn(8, 4, 5)
c = a @ b

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
np.dot(x, y)

np.linalg.inv(mat)
U, S, Vt = np.linalg.svd(mat)
eigvals, eigvecs = np.linalg.eig(mat)
```

---

## 7. 随机数：训练要可复现

```python
np.random.seed(42)
rng = np.random.default_rng(42)
x = rng.standard_normal((3, 4))
idx = rng.choice(10, size=3, replace=False)
```

**工程标准**：所有可能涉及随机的函数，接受一个 `rng` 参数或 `seed`，不要用全局状态。

---

## 8. NumPy ↔ PyTorch 互转

```python
import numpy as np
import torch

arr = np.array([1, 2, 3], dtype=np.float32)
t = torch.from_numpy(arr)

arr2 = t.numpy()

arr2 = t.cpu().numpy()

arr3 = arr.copy()
t2 = torch.from_numpy(arr3)
```

**关键**：`torch.from_numpy` 共享内存——修改 Tensor 会影响原 array。

---

## 9. 常见陷阱

### 9.1 忘记 `axis` 参数

```python
a = np.array([[1, 2], [3, 4]])
a.sum()
a.sum(axis=0)
a.sum(axis=1)
```

### 9.2 整数除法 / 类型转换

```python
a = np.array([1, 2, 3])
b = np.array([2.0, 2.0, 2.0])
c = a / b
print(c.dtype)

a = np.array([0, 1], dtype=np.int32)
b = a.mean()
print(b.dtype)
```

### 9.3 原地 vs 复制

```python
a = np.arange(10)
b = a.reshape(2, 5)
b[0, 0] = 999
print(a[0])

c = a.copy()
```

---

## 10. 速查：常用 API

| 操作 | NumPy | PyTorch 对应 |
|---|---|---|
| 创建 | `np.zeros` / `np.ones` / `np.arange` | `torch.zeros` / `torch.ones` / `torch.arange` |
| 随机 | `np.random.randn` | `torch.randn` |
| 形状 | `reshape` / `transpose` | `reshape` / `permute` |
| 矩阵乘 | `a @ b` | `a @ b` |
| 归约 | `sum(axis=)` / `mean(axis=)` | `sum(dim=)` / `mean(dim=)` |
| 索引 | `a[mask]` / `a[idx]` | 同 |
| 堆叠 | `np.stack` / `np.concatenate` | `torch.stack` / `torch.cat` |

**一句话**：NumPy 吃透了，PyTorch 张量操作基本免费学。

下一节进入 **einops**，让你的形状操作代码彻底脱胎换骨。
