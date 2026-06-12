# 01 · LLM 架构全解

> 目标：搞懂主流大语言模型的架构设计——从 Transformer 原论文到 2025 年的 DeepSeek-V3。
> 你不需要从零训练这些模型，但你必须能看懂 `config.json`、理解每个设计选择背后的动机。
> 如果你还不熟悉基础 Transformer，先读模块 02 的
> [Transformer 原理白话速览](../02-deep-learning-libraries/06-transformer-principles-overview.md)
> 和 [Transformer 架构深度剖析](../02-deep-learning-libraries/05-transformer-from-scratch.md)；
> 本文重点放在 LLM 架构演进和现代变体。

---

## 1. Transformer 家族树

2017 年 "Attention Is All You Need" 提出 Transformer，此后衍生出三大流派：

```
                         Transformer (2017)
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   Encoder-only        Encoder-Decoder       Decoder-only
   (双向注意力)         (交叉注意力)          (因果注意力)
          │                   │                   │
       BERT              T5 / BART          GPT / LLaMA / Qwen
       RoBERTa           mBART              Mistral / DeepSeek
       CLIP-text         Flan-T5            Gemma / Phi
```

### 为什么 LLM 主流是 Decoder-only？

| 因素 | Encoder-Decoder | Decoder-only |
|---|---|---|
| 训练目标 | 需要设计 span corruption 等 | 简单的 next-token prediction |
| 扩展性 | 参数分散在 encoder 和 decoder | 参数集中，scaling 更清晰 |
| 推理效率 | 需要两次前向 | 一次前向 + KV Cache |
| 多任务统一 | 需要不同的 prompt 格式 | 所有任务都是"续写" |
| 实际验证 | T5-11B 后很少有更大的 | GPT-3 → GPT-4，一路 scale up |

**结论**：Decoder-only 在 Scaling Laws 下表现最好，工程上最简单，已成为绝对主流。

---

## 2. GPT 系列演进

### 2.1 从 GPT-1 到 GPT-4

| 模型 | 年份 | 参数量 | 关键创新 |
|---|---|---|---|
| GPT-1 | 2018 | 117M | 首次证明"预训练 + 微调"在 NLP 的有效性 |
| GPT-2 | 2019 | 1.5B | 证明 zero-shot 能力，"语言模型即多任务学习者" |
| GPT-3 | 2020 | 175B | In-context learning，few-shot prompting |
| InstructGPT | 2022 | ~175B | RLHF 对齐，让模型"听话" |
| GPT-4 | 2023 | 未公开（传闻 MoE ~1.8T） | 多模态、推理能力飞跃 |

### 2.2 GPT 的核心架构

```python
# GPT 的 Transformer block（简化版）
class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, 4 * d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # Pre-Norm + 残差
        x = x + self.mlp(self.ln2(x))    # Pre-Norm + 残差
        return x
```

GPT-2 开始采用 **Pre-Norm**（LayerNorm 放在注意力之前而非之后），
训练更稳定，之后几乎所有 LLM 都沿用这个设计。

---

## 3. LLaMA 系列：开源 LLM 的基石

LLaMA 系列是 Meta 开源的 LLM 家族，几乎定义了 2023–2025 年开源 LLM 的架构标准。

### 3.1 LLaMA 1 (2023.02) 的架构创新

| 组件 | 传统 GPT | LLaMA 1 |
|---|---|---|
| 归一化 | LayerNorm | **RMSNorm**（去掉 mean，快 10–15%） |
| 激活函数 | GELU | **SwiGLU**（Swish + GLU，效果更好） |
| 位置编码 | 可学习绝对位置 | **RoPE**（旋转位置编码，天然支持外推） |
| 词表 | BPE (GPT-2 tokenizer) | SentencePiece (32K) |

```python
# RMSNorm vs LayerNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

# SwiGLU MLP
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

### 3.2 LLaMA 2 (2023.07)

- **Grouped Query Attention (GQA)**：介于 MHA 和 MQA 之间，KV 头数少于 Q 头数
- 训练数据量从 1T → 2T tokens
- 上下文从 2K → 4K
- 首次包含 Chat 版本（RLHF 对齐）

### 3.3 LLaMA 3 / 3.1 (2024)

- 词表扩展到 128K（支持更多语言）
- 上下文扩展到 128K tokens
- 训练数据 15T+ tokens
- 最大模型 405B 参数

### 3.4 GQA 详解

```
MHA (Multi-Head Attention):        每个 Q 头有独立的 K、V 头
  Q: [h1, h2, h3, h4, h5, h6, h7, h8]
  K: [h1, h2, h3, h4, h5, h6, h7, h8]  ← 8 个 KV 头
  V: [h1, h2, h3, h4, h5, h6, h7, h8]

MQA (Multi-Query Attention):       所有 Q 头共享 1 个 K、V 头
  Q: [h1, h2, h3, h4, h5, h6, h7, h8]
  K: [h1]                                ← 1 个 KV 头
  V: [h1]

GQA (Grouped Query Attention):    每组 Q 头共享 1 个 K、V 头
  Q: [h1, h2, h3, h4 | h5, h6, h7, h8]
  K: [h1            |  h2            ]   ← 2 个 KV 头
  V: [h1            |  h2            ]
```

**GQA 的好处**：
- KV Cache 显存减少 4–8x（相比 MHA）
- 推理速度提升（KV Cache 更小 → 访存更少）
- 效果损失很小（比 MQA 好，接近 MHA）

```python
# GQA 的核心：KV 头数 < Q 头数
class GQAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # 每个 KV 头被几个 Q 头共享
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # 扩展 KV 头以匹配 Q 头数
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)
```

---

## 4. Qwen 系列

Qwen（通义千问）是阿里巴巴开源的 LLM 系列，在中文和多语言任务上表现优异。

### 4.1 Qwen 演进

| 版本 | 时间 | 参数规格 | 关键特点 |
|---|---|---|---|
| Qwen 1 | 2023.08 | 7B / 14B | 基础架构接近 LLaMA |
| Qwen 1.5 | 2024.02 | 0.5B–110B | GQA、更多训练数据、改进对齐 |
| Qwen 2 | 2024.06 | 0.5B–72B | 128K 上下文、多语言增强 |
| Qwen 2.5 | 2024.09 | 0.5B–72B | 代码和数学能力增强、18T tokens |
| Qwen 3 | 2025 | 0.6B–235B (MoE) | 混合思维模式、MoE 架构 |

### 4.2 Qwen 架构特点

```python
# 读懂 Qwen2 的 config.json
{
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 4096,           # d_model
    "intermediate_size": 22016,    # FFN 隐藏层（SwiGLU）
    "num_attention_heads": 32,     # Q 头数
    "num_key_value_heads": 8,      # KV 头数（GQA，4:1）
    "num_hidden_layers": 32,       # Transformer 层数
    "vocab_size": 152064,          # 大词表支持多语言
    "max_position_embeddings": 131072,  # 128K 上下文
    "rms_norm_eps": 1e-6,          # RMSNorm
    "rope_theta": 1000000.0,       # RoPE base frequency
    "sliding_window": null,
    "tie_word_embeddings": false
}
```

Qwen 系列的重要设计选择：
- **大词表**（152K）：覆盖中、英、日、韩等多语言，减少 tokenization 碎片化
- **GQA**（4:1 比例）：32 个 Q 头共享 8 个 KV 头
- **RoPE base 增大**：`rope_theta=1M`，有利于长上下文外推
- **不绑定 embedding 权重**：`tie_word_embeddings=false`，输入和输出 embedding 独立

---

## 5. Mistral / Mixtral：窗口注意力与 MoE

### 5.1 Mistral 7B (2023.09)

Mistral 7B 以精巧的架构设计击败了 LLaMA 2 13B：

| 特性 | 说明 |
|---|---|
| **Sliding Window Attention (SWA)** | 每层只看最近 4096 个 token，但通过层堆叠覆盖更远 |
| GQA | 8 个 KV 头 vs 32 个 Q 头 |
| 无 bias | 所有 Linear 层去掉 bias |

```
Sliding Window Attention 原理：

Layer 1: token_i 看 [i-4096, i]
Layer 2: token_i 的表示已经融合了 Layer 1 中 [i-4096, i] 的信息
         → 现在它看 [i-8192, i]（通过信息传递）
...
Layer 32: 理论感受野 = 32 × 4096 = 131,072 tokens

但注意：这种"间接"覆盖远不如直接注意力。
```

### 5.2 Mixtral 8x7B (2024.01)：开源 MoE 的标杆

Mixtral 是第一个真正好用的开源 MoE 模型：

```
Mixtral 架构：
  - 总参数：46.7B
  - 激活参数：12.9B（每个 token 只用 2 个 expert）
  - 8 个 expert，每个 expert 是一个 SwiGLU FFN
  - 路由策略：Top-2

┌─────────────────────────────────────┐
│         Transformer Block           │
│                                     │
│  ┌─────────────────────────────┐    │
│  │    Self-Attention (GQA)     │    │
│  └──────────────┬──────────────┘    │
│                 ▼                   │
│  ┌─────────────────────────────┐    │
│  │        MoE FFN Layer        │    │
│  │  ┌───┐ ┌───┐     ┌───┐     │    │
│  │  │ E1│ │ E2│ ... │ E8│     │    │
│  │  └─┬─┘ └─┬─┘     └─┬─┘     │    │
│  │    │     │         │        │    │
│  │  ┌─┴─────┴─────────┴─┐     │    │
│  │  │   Router (Top-2)   │     │    │
│  │  └────────────────────┘     │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

---

## 6. Mixture of Experts (MoE) 深入

### 6.1 MoE 核心思想

用多个"专家"（Expert）替换 Transformer 中的 FFN 层，
每个 token 只激活其中 K 个专家——**总参数大但激活参数少**。

```python
class MoELayer(nn.Module):
    def __init__(self, dim, hidden_dim, n_experts, top_k):
        super().__init__()
        self.experts = nn.ModuleList([
            SwiGLU(dim, hidden_dim) for _ in range(n_experts)
        ])
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)

        logits = self.gate(x_flat)                     # (B*T, n_experts)
        weights, indices = logits.topk(self.top_k)     # (B*T, top_k)
        weights = F.softmax(weights, dim=-1)

        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                expert_input = x_flat[mask]
                expert_weight = weights[indices == i]
                output[mask] += expert_weight.unsqueeze(-1) * expert(expert_input)

        return output.view(B, T, D)
```

### 6.2 路由策略

| 策略 | 说明 | 代表模型 |
|---|---|---|
| Top-K | 每个 token 选 K 个得分最高的 expert | Mixtral (K=2) |
| Expert Choice | 每个 expert 选 top-K 个 token | Switch Transformer |
| Shared Expert | 固定一部分 expert 始终激活 | DeepSeek-MoE |

### 6.3 Load Balancing Loss

MoE 的致命问题：**路由崩塌**——所有 token 都涌向少数几个 expert，其余 expert "饿死"。

```python
# 辅助损失：鼓励 expert 负载均衡
def load_balancing_loss(gate_logits, n_experts):
    probs = F.softmax(gate_logits, dim=-1)          # (B*T, n_experts)
    # 每个 expert 被选中的频率
    freq = probs.mean(dim=0)                         # (n_experts,)
    # 理想分布：均匀
    target = torch.ones_like(freq) / n_experts
    return n_experts * (freq * target).sum()
```

---

## 7. DeepSeek 系列：国产 LLM 的架构创新

### 7.1 DeepSeek-V2 (2024.05)：MLA + DeepSeekMoE

DeepSeek-V2 提出了两个重要创新：

**Multi-head Latent Attention (MLA)**——用低秩压缩替代标准 KV Cache：

```
标准 MHA 的 KV Cache（每层需缓存的数据量）:
  KV Cache 大小 = 2 × n_heads × head_dim × seq_len
  例如 n_heads=128, head_dim=128, seq_len=32K
  → 2 × 128 × 128 × 32768 = 1GB / 层 (fp16)

MLA 的做法：
  把 KV 压缩到低维潜在空间（latent），推理时只缓存 latent
  KV Cache 大小 = d_latent × seq_len
  例如 d_latent=512
  → 512 × 32768 = 32MB / 层 (fp16)

  压缩比 ≈ 32x!
```

```python
class MLAttention(nn.Module):
    """Multi-head Latent Attention (MLA) 简化示意"""
    def __init__(self, d_model, n_heads, d_latent):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Q: 先压缩到 latent，再解压到多头
        self.q_down = nn.Linear(d_model, d_latent, bias=False)
        self.q_up = nn.Linear(d_latent, n_heads * self.head_dim, bias=False)

        # KV: 压缩到共享 latent
        self.kv_down = nn.Linear(d_model, d_latent, bias=False)
        self.k_up = nn.Linear(d_latent, n_heads * self.head_dim, bias=False)
        self.v_up = nn.Linear(d_latent, n_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape

        # 推理时只需缓存 kv_latent（维度远小于完整 KV）
        q_latent = self.q_down(x)
        kv_latent = self.kv_down(x)

        q = self.q_up(q_latent).view(B, T, self.n_heads, self.head_dim)
        k = self.k_up(kv_latent).view(B, T, self.n_heads, self.head_dim)
        v = self.v_up(kv_latent).view(B, T, self.n_heads, self.head_dim)

        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)
```

**DeepSeekMoE**——细粒度 expert + 共享 expert：

| 特性 | Mixtral | DeepSeekMoE |
|---|---|---|
| Expert 数量 | 8 | 160 (V2) / 256 (V3) |
| Expert 粒度 | 粗（每个很大） | 细（每个很小） |
| 激活 expert 数 | 2 | 6 out of 160 |
| 共享 expert | 无 | 有（2 个始终激活） |

### 7.2 DeepSeek-V3 (2024.12)

- 671B 总参数，37B 激活参数
- 在 14.8T tokens 上训练
- 训练成本仅 ~$5.5M（FP8 训练 + 高效工程）
- Multi-Token Prediction (MTP) 作为辅助训练目标
- 性能比肩 GPT-4o 和 Claude 3.5 Sonnet

---

## 8. 架构创新时间线

```
2017  Transformer        原始架构 (Post-Norm, Sinusoidal PE, GELU)
  │
2018  GPT-1              Decoder-only 预训练
  │
2019  GPT-2              Pre-Norm (LayerNorm 放在前面)
  │
2020  GPT-3              175B, in-context learning
  │
2021  RoPE               旋转位置编码 (Su et al.)
  │
2022  PaLM               SwiGLU 激活函数普及
  │   Chinchilla          Scaling Law: 数据和模型同等重要
  │   FlashAttention      IO-aware 注意力加速 (Dao et al.)
  │
2023  LLaMA 1            RMSNorm + SwiGLU + RoPE = 开源标配
  │   LLaMA 2            GQA 成为主流
  │   Mistral 7B         Sliding Window Attention
  │   Mixtral 8x7B       开源 MoE
  │   FlashAttention-2   进一步优化
  │
2024  LLaMA 3            128K 上下文，405B 参数
  │   Qwen 2 / 2.5       大词表 + 多语言
  │   DeepSeek-V2        MLA (Multi-head Latent Attention)
  │   DeepSeek-V3        671B MoE，成本仅 $5.5M
  │   FlashAttention-3   Hopper GPU 优化
  │
2025  Qwen 3             混合思维 MoE
      DeepSeek-R1        推理增强
```

---

## 9. 长上下文技术

LLM 上下文长度从 2K → 128K → 1M+，靠的是一系列技术：

### 9.1 RoPE 及其外推

```python
# RoPE 核心：对 Q、K 应用旋转矩阵，角度与位置相关
def apply_rotary_emb(x, freqs):
    x_r, x_i = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    cos, sin = freqs.cos(), freqs.sin()
    out_r = x_r * cos - x_i * sin
    out_i = x_r * sin + x_i * cos
    return torch.stack([out_r, out_i], dim=-1).flatten(-2).type_as(x)

def precompute_freqs(dim, max_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len)
    return torch.outer(t, freqs)
```

### 9.2 长度外推方法

| 方法 | 思路 | 代表 |
|---|---|---|
| **ALiBi** | 不用位置编码，给注意力分数加线性距离惩罚 | BLOOM |
| **位置插值 (PI)** | 把超长位置"压缩"到训练长度范围内 | Meta |
| **Dynamic NTK** | 动态调整 RoPE 的 base frequency | CodeLLaMA |
| **YaRN** | NTK-aware 插值 + 注意力缩放 | Together AI |
| **Ring Attention** | 跨多 GPU 分布式处理超长序列 | UC Berkeley |

```python
# Dynamic NTK-aware Scaling RoPE
def dynamic_ntk_rope(dim, max_len, base=10000.0, original_max_len=4096):
    if max_len <= original_max_len:
        return precompute_freqs(dim, max_len, base)

    # 根据目标长度动态调整 base
    scale = max_len / original_max_len
    new_base = base * (scale ** (dim / (dim - 2)))
    return precompute_freqs(dim, max_len, new_base)
```

---

## 10. Scaling Laws

### 10.1 Kaplan Scaling Laws (OpenAI, 2020)

模型性能（loss）与三个因素的幂律关系：

\[
L(N, D, C) \propto N^{-0.076} + D^{-0.095} + C^{-0.050}
\]

- \(N\)：模型参数量
- \(D\)：训练数据量（tokens）
- \(C\)：计算量（FLOPs）

### 10.2 Chinchilla 最优 (DeepMind, 2022)

**核心发现**：给定计算预算 \(C\)，最优策略是让参数量 \(N\) 和数据量 \(D\) 同比例增长。

```
Chinchilla 最优比例：D ≈ 20 × N

例如：
  7B 模型 → 需要 140B tokens
  70B 模型 → 需要 1.4T tokens

实际做法（2024 后）：
  LLaMA 3 8B → 在 15T tokens 上训练（远超 Chinchilla 最优）
  原因：推理计算 >> 训练计算，小模型训久一点更划算（Inference-optimal）
```

### 10.3 MoE 的 Scaling 特性

| 特性 | Dense 模型 | MoE 模型 |
|---|---|---|
| 参数利用率 | 100%（全部激活） | 10–30%（稀疏激活） |
| 同等 FLOPs 下的性能 | 基准 | 更好（相当于 2–4x 的 dense 模型） |
| 训练稳定性 | 较好 | 需要 load balancing |
| 推理显存 | 与参数量成正比 | 需要加载全部参数但只激活部分 |

---

## 11. 主流 LLM 架构对比

| 模型 | 参数 | 上下文 | 注意力 | KV 头 | 词表 | 激活 | 归一化 | 位置编码 | 训练数据 |
|---|---|---|---|---|---|---|---|---|---|
| LLaMA 2 7B | 6.7B | 4K | GQA | 8 | 32K | SwiGLU | RMSNorm | RoPE | 2T |
| LLaMA 3 8B | 8B | 128K | GQA | 8 | 128K | SwiGLU | RMSNorm | RoPE | 15T |
| Qwen 2.5 7B | 7.6B | 128K | GQA | 4 | 152K | SwiGLU | RMSNorm | RoPE | 18T |
| Mistral 7B | 7.3B | 32K | GQA+SWA | 8 | 32K | SwiGLU | RMSNorm | RoPE | 未公开 |
| Mixtral 8x7B | 46.7B (12.9B↑) | 32K | GQA | 8 | 32K | SwiGLU | RMSNorm | RoPE | 未公开 |
| DeepSeek-V2 | 236B (21B↑) | 128K | MLA | - | 100K | SwiGLU | RMSNorm | RoPE | 8.1T |
| DeepSeek-V3 | 671B (37B↑) | 128K | MLA | - | 129K | SwiGLU | RMSNorm | RoPE | 14.8T |

> ↑ = 激活参数量。MLA 没有传统意义上的 KV 头，用 latent 替代。

---

## 12. 如何读懂一个模型的 config.json

拿到一个新模型，第一件事就是看 `config.json`：

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
print(config)

# 关键字段解读：
# hidden_size          → d_model（模型维度）
# intermediate_size    → FFN 隐藏层维度（通常 ≈ 2.7 × hidden_size for SwiGLU）
# num_attention_heads  → Q 头数
# num_key_value_heads  → KV 头数（< Q 头数 = GQA）
# num_hidden_layers    → Transformer 层数
# max_position_embeddings → 最大上下文长度
# vocab_size           → 词表大小
# rope_theta           → RoPE base（越大越利于长上下文）
# rms_norm_eps         → RMSNorm epsilon
```

**快速估算参数量**：

```python
def estimate_params(config):
    d = config.hidden_size
    L = config.num_hidden_layers
    V = config.vocab_size
    ffn = config.intermediate_size
    n_h = config.num_attention_heads
    n_kv = config.num_key_value_heads
    h_dim = d // n_h

    embedding = V * d
    per_layer_attn = d * (n_h + 2 * n_kv) * h_dim + d * d
    per_layer_ffn = 3 * d * ffn   # SwiGLU: w1, w2, w3
    per_layer_norm = 2 * d
    lm_head = V * d

    total = embedding + L * (per_layer_attn + per_layer_ffn + per_layer_norm) + lm_head
    print(f"估算参数量: {total / 1e9:.2f}B")
    return total
```

---

## 13. 用 Transformers 加载并查看模型结构

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-1.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 查看模型结构
print(model)

# 统计参数量
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数: {total/1e9:.2f}B, 可训练: {trainable/1e9:.2f}B")

# 查看某一层的结构
print(model.model.layers[0])

# 查看注意力层权重形状
attn = model.model.layers[0].self_attn
print(f"Q proj: {attn.q_proj.weight.shape}")
print(f"K proj: {attn.k_proj.weight.shape}")
print(f"V proj: {attn.v_proj.weight.shape}")
```

---

## 14. 常见坑

### 14.1 混淆"总参数"和"激活参数"

Mixtral 8x7B 的总参数是 46.7B，但每个 token 只激活 12.9B。
对比模型时要看**激活参数量**和**推理 FLOPs**，而不是总参数。

### 14.2 以为 GQA 只是"减少参数"

GQA 的核心价值是**减少 KV Cache 显存**，而不是减少模型参数量。
模型参数只少了一点（KV projection 变小），但推理时 KV Cache 能小 4–8 倍。

### 14.3 把 Sliding Window 当成 Global Attention

Sliding Window Attention 在单层内只能看固定窗口大小的上下文。
虽然通过层堆叠可以扩大"理论感受野"，但远距离依赖的建模能力弱于 full attention。

### 14.4 忽视 Tokenizer 的影响

词表大小和 tokenizer 质量直接影响模型效率：
- 中文任务用 LLaMA 1 的 32K 词表，1 个汉字可能要 2–3 个 token
- 用 Qwen 的 152K 词表，1 个汉字通常 1 个 token
- 这意味着同样的中文文本，Qwen 的"有效上下文"比 LLaMA 1 长 2–3 倍

### 14.5 Scaling Law 不等于"越大越好"

Scaling Law 说的是**给定计算预算下的最优分配**。
盲目增大模型但数据不够（under-trained）或计算不够（under-compute），效果反而差。

---

## 小结

| 概念 | 一句话解释 |
|---|---|
| Pre-Norm | LayerNorm 放在注意力/FFN 之前，训练更稳定 |
| RMSNorm | 比 LayerNorm 快 10–15%，去掉了 mean centering |
| SwiGLU | 比 GELU 效果更好的 FFN 激活函数 |
| RoPE | 旋转位置编码，天然支持长度外推 |
| GQA | KV 头数 < Q 头数，大幅减少 KV Cache 显存 |
| MoE | 多专家混合，总参数大但激活参数少 |
| MLA | 用低秩 latent 压缩 KV Cache，DeepSeek 发明 |
| Scaling Law | 模型性能 ∝ 参数×数据×计算的幂律关系 |

下一节学习图像生成架构——从 GAN 到 Diffusion 到 Flow Matching。
