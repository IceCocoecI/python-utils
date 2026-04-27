# 05 · Transformer 架构深度剖析

> 所有现代 AIGC 模型（GPT / LLaMA / Qwen / Flux / DiT / Sora）都建立在 Transformer 之上。
> **不真正理解 Transformer，后面学什么都会半懂不懂。**
> 本节不教你调 API，而是带你把这些"看上去很玄"的概念拆到最底。

---

## 1. 为什么 Transformer 统治了 AIGC？

简单对比：

| 架构 | 序列建模方式 | 并行度 | 长程依赖 |
|---|---|---|---|
| RNN / LSTM | 逐步递推 | 差（串行） | 弱（信息逐层稀释） |
| CNN | 局部感受野 | 好 | 弱（需多层堆叠） |
| **Transformer** | **全局自注意力** | **极好** | **强（直接相连）** |

**一句话**：Transformer 用"矩阵乘法 + Softmax"取代了"逐步递推"，让 GPU 吃得饱、训得动、拓展性好。

---

## 2. 数据流：一条 prompt 在 Transformer 里走过的路径

```
┌──────────────────────────────────────────────────────────────┐
│ "hello world"                                                │
│    │                                                         │
│    ▼  Tokenizer                                              │
│ [15339, 1917]              (list[int], seq_len=2)            │
│    │                                                         │
│    ▼  Embedding + Position                                   │
│ X  ∈ ℝ^(B, T, d_model)    (例如 B=1, T=2, d=4096)             │
│    │                                                         │
│    ▼  N × Transformer Block                                  │
│    ┌─── LayerNorm → Multi-Head Attention → residual          │
│    └─── LayerNorm → FFN (SwiGLU / GELU)    → residual        │
│    │                                                         │
│    ▼  LayerNorm                                              │
│    ▼  lm_head (Linear to vocab_size)                         │
│ logits ∈ ℝ^(B, T, vocab)                                     │
│    │                                                         │
│    ▼  Softmax → next token                                   │
└──────────────────────────────────────────────────────────────┘
```

记住这张图。后面所有细节都是在填这张图的细节。

---

## 3. Self-Attention：一行公式解决一切

$$
\text{Attention}(Q, K, V) = \text{Softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

用人话说：

1. 每个 token 发出一个"问题" Q；
2. 每个 token 都准备一个"可被查询的 key" K、一个"携带的值" V；
3. `Q·Kᵀ`：所有 token 的问题和所有 token 的 key 相乘 → 得到"相似度矩阵"；
4. `Softmax` 把每一行归一化 → 注意力权重；
5. 加权 V → 输出：每个 token 都根据"别的 token 对它的贡献"得到新的表示。

### 3.1 Multi-Head：多视角

把 `d_model` 切成 `h` 组独立的 attention，各算各的，最后拼回来：

```
d_model = 4096, n_heads = 32, d_head = 128
Q, K, V: (B, T, 4096) → reshape → (B, 32, T, 128)
```

每个 head 可以关注不同"子空间"——有的看语法、有的看语义、有的看指代。

### 3.2 最小实现（从零手写）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b t (three h d) -> three b h t d",
                             three=3, h=self.n_heads)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)

        y = rearrange(y, "b h t d -> b t (h d)")
        return self.out(y)
```

`F.scaled_dot_product_attention` 是 PyTorch 2.x 内置的 FlashAttention 实现——直接用，不要手写 softmax 算 `QKᵀ`。

### 3.3 如果你一定要手写一遍（教学用）

```python
def naive_attention(q, k, v, mask=None):
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / d ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = scores.softmax(dim=-1)
    return attn @ v
```

**问题**：`scores` 的形状是 `(B, H, T, T)`——序列长 8k 时就是 `8k × 8k = 64M` 浮点数/head，显存炸掉。FlashAttention 就是来救这个问题的。

---

## 4. 位置编码（Positional Encoding）

self-attention 本身**不认位置**（交换两个 token 输入，输出也交换）。必须显式注入位置信息。

### 4.1 绝对位置编码：原版 Transformer

```python
pe = torch.zeros(max_len, d_model)
position = torch.arange(max_len).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)

x = x + pe[:x.size(1)]
```

### 4.2 学习式位置嵌入：GPT-2

```python
pos_emb = nn.Embedding(max_len, d_model)
positions = torch.arange(0, x.size(1))
x = tok_emb + pos_emb(positions)
```

**问题**：这两种都在"输入端加一次"，且**长度外推能力差**——训练时 2k，推理时 4k 就退化。

### 4.3 RoPE（Rotary Position Embedding）：现代 LLM 标准

核心思想：**不加到输入上，而是在每一层的 Q/K 上"旋转"**。

```python
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
```

`cos` / `sin` 由位置和频率计算，形状 `(T, d_head)`。

**优势**：
- 内积 `Q·Kᵀ` 只依赖**相对位置**，天生适合序列外推。
- LLaMA / Qwen / Mistral / Mixtral 全系列都用它。

### 4.4 ALiBi：简化版位置偏置

不修改 Q/K，而是给 attention score 加一个**和距离成正比的负偏置**：

```
score[i, j] += -|i - j| * slope
```

极简、外推好，Bloom、Baichuan 用过。

---

## 5. 注意力掩码（Attention Mask）：三种形态

### 5.1 因果掩码（Causal Mask）：LLM 必备

```
        k_0   k_1   k_2   k_3
q_0     1     0     0     0
q_1     1     1     0     0
q_2     1     1     1     0
q_3     1     1     1     1
```

意思：`q_i` 只能看到 `k_0...k_i`——**未来是不可见的**，这保证了自回归生成的合法性。

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### 5.2 填充掩码（Padding Mask）：处理变长 batch

Batch 里句子长度不一，pad 到同一长度后，pad 位置不能参与计算：

```python
attn_mask = (input_ids != pad_token_id)
y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask[:, None, None, :])
```

### 5.3 组合掩码：LLM 推理时实际用的

```
final_mask = causal_mask & padding_mask
```

HuggingFace `transformers` 内部统一叫 `attention_mask`，把两件事都处理了——但作为工程师你必须知道为什么要两件。

---

## 6. KV Cache：推理加速的命脉

### 6.1 问题：naive 生成的复杂度

生成 token `t_n` 需要重新算 `t_0..t_{n-1}` 的 Attention——总复杂度 `O(N²)` 每步，整个序列 `O(N³)`。

### 6.2 解决：缓存 K、V

Key / Value 一旦算出来就**不会变**（因为它们只依赖历史 token）。把它们缓存下来：

```
第 t 步推理：
- 只有 1 个新 query q_t
- K = [K_cache; k_t]  (concat)
- V = [V_cache; v_t]
- attn = softmax(q_t @ K^T / √d) @ V
```

复杂度降到 `O(N)` 每步。**大模型推理慢不慢，主要看这个实现好不好**。

### 6.3 最小实现骨架

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        ...
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)

    def forward(self, x, use_cache=False):
        q, k, v = self.compute_qkv(x)
        if use_cache and self.k_cache is not None:
            k = torch.cat([self.k_cache, k], dim=-2)
            v = torch.cat([self.v_cache, v], dim=-2)
        if use_cache:
            self.k_cache, self.v_cache = k, v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(self.k_cache is None))
        ...
```

### 6.4 KV Cache 的显存成本

```
kv_cache_bytes = 2 × batch × seq_len × n_layers × n_heads × d_head × dtype_bytes
                 ↑
                 K 和 V
```

举例（LLaMA-7B，fp16，bs=1，seq=4096）：
`2 × 1 × 4096 × 32 × 32 × 128 × 2 ≈ 2.1 GB`

**这是为什么推理时 seq_len 越长，显存增长越快**——bs 小了 KV cache 还是会爆。

### 6.5 进阶优化（了解即可）

- **GQA（Grouped-Query Attention）**：多个 query head 共享一组 KV（LLaMA-2 70B、LLaMA-3 全系列用）。
- **MQA（Multi-Query Attention）**：所有 query head 共享 **1 组** KV（更激进）。
- **PagedAttention**：vLLM 的核心——把 KV cache 像虚拟内存一样分页管理，支持 batch 内不同长度高效共存。

---

## 7. FFN 与归一化

### 7.1 FFN 的演化

原版 Transformer：

```python
y = W_2 @ GELU(W_1 @ x)
```

现代 LLM（LLaMA 系列）：**SwiGLU**

```python
y = W_3 @ (SiLU(W_1 @ x) * (W_2 @ x))
```

参数从 2 个 Linear 变成 3 个，但效果提升显著。

### 7.2 归一化层

| 名字 | 公式 | 使用 |
|---|---|---|
| LayerNorm | `(x - μ) / σ * γ + β` | 原版 Transformer |
| **RMSNorm** | `x / √(mean(x²)) * γ` | LLaMA / Qwen / Mistral |
| GroupNorm | 分组做 LN | 扩散模型常用 |

**RMSNorm** 省去均值中心化，速度快、效果相当。

### 7.3 Pre-Norm vs Post-Norm

```
Post-Norm (原版):  y = LN(x + SubLayer(x))
Pre-Norm (现代):   y = x + SubLayer(LN(x))
```

现代大模型全用 **Pre-Norm**——深度堆叠更稳定，训练不容易发散。

---

## 8. 采样策略（文本生成的灵魂）

生成时怎么从 logits 选 next token？

### 8.1 基本操作

```python
logits = logits / temperature

probs = F.softmax(logits, dim=-1)

next_token = torch.argmax(probs, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

### 8.2 Top-K / Top-P

```python
def top_k_sampling(logits, k):
    top_logits, top_idx = logits.topk(k)
    probs = F.softmax(top_logits, dim=-1)
    sampled = torch.multinomial(probs, 1)
    return top_idx.gather(-1, sampled)


def top_p_sampling(logits, p):
    sorted_logits, sorted_idx = logits.sort(descending=True)
    cumprobs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    mask = cumprobs > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits[mask] = float("-inf")
    probs = F.softmax(sorted_logits, dim=-1)
    sampled = torch.multinomial(probs, 1)
    return sorted_idx.gather(-1, sampled)
```

### 8.3 推荐配置

| 任务 | temperature | top_p | top_k |
|---|---|---|---|
| 抽取 / 分类 | 0（贪心） | - | - |
| 代码生成 | 0.2–0.4 | 0.95 | - |
| 对话 / 问答 | 0.6–0.8 | 0.9 | - |
| 创意写作 | 0.8–1.2 | 0.95 | - |

---

## 9. 现代 LLM 的完整 Block（LLaMA 风格）

```python
class LlamaBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, causal=True)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ffn)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

整个 LLaMA-7B 就是 32 个这样的 Block 堆起来的。

---

## 10. FlashAttention 简要

朴素 attention 的痛点：`QKᵀ` 矩阵大到装不下 SRAM，反复读写 HBM 成本极高。

FlashAttention 的两招：
1. **Tile 分块**：把 Q/K 切成小块，每次只算一块，结果在 SRAM 里用在线 softmax 累加。
2. **重计算**：反向传播时不保存 attention 矩阵，而是重新算一次（省显存）。

结果：
- **速度快 2–4 倍**（减少 HBM 读写）
- **显存降 10×**（`O(N²) → O(N)`）

作为工程师，你需要知道的是：

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**PyTorch 2.x 的 SDPA 已经内置 FlashAttention**——背后会自动选最优内核。你不需要手动调用 `flash-attn` 库，除非有特殊需求（如 `flash-attn-3`、自定义 bias）。

---

## 11. 推荐的"读源码路径"

想真正吃透 Transformer，按以下顺序读源码（每份代码都不超过 300 行）：

1. **Karpathy: `nanoGPT/model.py`** — GPT 最简实现
   https://github.com/karpathy/nanoGPT/blob/master/model.py
2. **Karpathy: `nanochat/gpt.py`** — 带 RoPE、GQA 的现代版
   https://github.com/karpathy/nanochat
3. **Meta: `llama/model.py`** — 工业级 LLaMA 实现
   https://github.com/meta-llama/llama/blob/main/llama/model.py
4. **HF: `modeling_llama.py`** — HuggingFace 工程化实现（含 KV cache 细节）
   https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

每读一份，你的认知都会上一个台阶。

---

## 12. 自检问答

读完本节，你应该能回答：

- [ ] 为什么 Attention 要除以 `√d_k`？（softmax 饱和）
- [ ] Multi-Head 相比 Single-Head 增加了参数量吗？（没有，`d_model` 是总的）
- [ ] Pre-Norm 为什么比 Post-Norm 更稳？（梯度路径直通）
- [ ] RoPE 凭什么能外推？（内积只依赖相对位置）
- [ ] KV Cache 只能缓存 K/V，不能缓存 Q 吗？（因为新 token 的 Q 每步都变）
- [ ] FlashAttention 为什么能省显存？（分块 + 重计算）
- [ ] GQA 和 MQA 的区别？（共享组的粒度不同）
- [ ] seq_len = 8192 的 KV cache 有多大？（按第 6.4 节公式算）

能答 6 题以上，你在 Transformer 上已经超越 80% 的同行。

---

## 小结

```
Attention is All You Need
  → Multi-Head + Scaled Dot-Product
  → 位置编码：RoPE 为主
  → 掩码：causal + padding 组合
  → 推理：KV Cache → GQA → PagedAttention
  → 内核：FlashAttention (已在 SDPA 里)
  → 归一化：RMSNorm + Pre-Norm
  → FFN：SwiGLU
```

这些元素构成了**现代所有 LLM / DiT / Flux 模型**的骨架。理解它们后，再去读最新论文、最新开源项目，会发现它们只是在某一个子模块上做创新。

下一步可以去读 [nanoGPT](https://github.com/karpathy/nanoGPT) 的 `model.py`——约 300 行，你会发现自己"居然能看懂每一行"。
