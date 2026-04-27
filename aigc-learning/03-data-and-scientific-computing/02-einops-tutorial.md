# 02 · einops：张量操作的 DSL

> 推荐等级：**★★★★★**
> 一旦用过 einops，你就再也不想回去写 `view` / `permute` / `reshape` 了。

---

## 1. 为什么用 einops？

看两段代码：

```python
y = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
```

```python
y = rearrange(x, "b c h w -> b (h w) c")
```

第二段：
- **一眼看懂维度含义**（不用数 0,1,2,3）。
- **自动检查**：形状对不上立即报错。
- **不易出 bug**：维度顺序由名字决定，不是位置。

einops 支持 NumPy / PyTorch / JAX / TensorFlow，API 完全一致。

---

## 2. 三大核心操作

| 操作 | 等价于 | 示例 |
|---|---|---|
| `rearrange` | reshape + transpose + stack + squeeze | `"b c h w -> b (h w) c"` |
| `reduce` | sum / mean / max + reshape | `"b c h w -> b c", "mean"` |
| `repeat` | tile + expand | `"b c -> b c n", n=4` |

```python
from einops import rearrange, reduce, repeat
```

---

## 3. `rearrange`：形状重整

### 3.1 基础：转置

```python
x = torch.randn(2, 3, 224, 224)

y = rearrange(x, "b c h w -> b h w c")
```

### 3.2 合并轴

```python
y = rearrange(x, "b c h w -> b c (h w)")
y = rearrange(x, "b c h w -> b (h w) c")

patches = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=16, p2=16)
```

### 3.3 分解轴

```python
y = rearrange(x, "b (h d) -> b h d", h=8)

y = rearrange(x, "b (p1 p2 c) (h) (w) -> b c (h p1) (w p2)", p1=2, p2=2)
```

### 3.4 维度增删

```python
y = rearrange(x, "h w c -> 1 h w c")
y = rearrange(x, "1 c h w -> c h w")
```

### 3.5 AIGC 实战：Multi-Head Attention 的 QKV 拆分

**普通写法：**

```python
B, T, C = x.shape
qkv = self.qkv(x)
q, k, v = qkv.chunk(3, dim=-1)
q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
```

**einops 写法：**

```python
qkv = self.qkv(x)
q, k, v = rearrange(qkv, "b t (three h d) -> three b h t d", three=3, h=self.n_heads)
```

一行搞定，形状明明白白。

---

## 4. `reduce`：归约

```python
y = reduce(x, "b c h w -> b c", "mean")

y = reduce(x, "b c (h 2) (w 2) -> b c h w", "max")

y = reduce(x, "b c h w -> b c 1 1", "mean")
```

**好处**：一行代码既做了 pooling 又做了形状变换。

---

## 5. `repeat`：复制

```python
y = repeat(x, "c h w -> b c h w", b=8)

y = repeat(x, "h w c -> h (w 3) c")

mask = repeat(mask, "t -> b t", b=B)
```

---

## 6. `einops.layers.torch`：作为 PyTorch Module 使用

```python
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce

model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.ReLU(),
    Reduce("b c (h 2) (w 2) -> b c h w", "max"),
    Rearrange("b c h w -> b (c h w)"),
    nn.Linear(64 * 56 * 56, 10),
)
```

它们是正常的 Module，可以 `.to(device)`、参与 `state_dict`、一切都如常。

---

## 7. `einsum`：最通用的张量运算

```python
import torch

a = torch.randn(4, 3, 5)
b = torch.randn(4, 5, 7)
c = torch.einsum("bij,bjk->bik", a, b)

attn = torch.einsum("bhid,bhjd->bhij", q, k)

y = torch.einsum("bhts,bhsd->bhtd", attn, v)

a = torch.randn(3)
b = torch.randn(3)
c = torch.einsum("i,i->", a, b)
```

如果 `einsum` 的字符串对你太抽象，用 `einops.einsum` 版本：

```python
from einops import einsum

attn = einsum(q, k, "b h t d, b h s d -> b h t s")
```

更易读——因为维度有名字。

---

## 8. AIGC 常见一行式小抄

### 8.1 CHW ↔ HWC

```python
img = rearrange(img, "h w c -> c h w")
img = rearrange(img, "c h w -> h w c")
```

### 8.2 ViT 切 patch

```python
patches = rearrange(img, "c (h p1) (w p2) -> (h w) (p1 p2 c)", p1=16, p2=16)
```

### 8.3 把 batch 展平成"样本级"

```python
y = rearrange(x, "b c h w -> (b h w) c")
```

### 8.4 batch 叠加 & 拆分（常用于 classifier-free guidance）

```python
both = torch.cat([cond, uncond], dim=0)
out = model(both)
cond_out, uncond_out = rearrange(out, "(two b) ... -> two b ...", two=2)
```

### 8.5 卷积 → 全局池化 → 分类头

```python
class Classifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        return self.head(self.backbone(x))
```

---

## 9. 常见坑

### 9.1 轴名冲突

```python
rearrange(x, "b b -> b b")
```

### 9.2 数字轴要带上下文

```python
rearrange(x, "b 10 h w -> ...")
rearrange(x, "b c h w -> b (c h w)")
```

### 9.3 带括号的合并要保证每一组只出现一次

```python
"b c (h 2) (w 2) -> b c h w"
"b (h w) c -> b h w c"
```

---

## 10. 学习建议

1. **从替换入手**：找到你代码里所有 `view` / `permute` / `reshape`，尝试用 einops 改写。
2. **读一遍 [nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py)**：它的 Attention 模块是 einops 思想的绝佳范例（虽然用了原生写法，你可以试着改写）。
3. **研究 [ViT / DiT 源码](https://github.com/facebookresearch/DiT/blob/main/models.py)**：里面大量用 einops。

---

## 小结

| 操作 | 记忆口诀 |
|---|---|
| `rearrange` | "不改数量只换形" |
| `reduce` | "减一个维度同时重排" |
| `repeat` | "增加一个维度并重复" |
| `Rearrange` / `Reduce` (Layer) | "塞进 `nn.Sequential` 就行" |
| `einsum` | "字符串下标表示的万能张量运算" |

**一句话**：**能用 einops 就用 einops**。你的未来同事/reviewer 会感谢你。
