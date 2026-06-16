# 00 · 深度学习核心理论

> 本文是模块 02 的理论底座。后面的 PyTorch、Transformers、Diffusers 文档偏工程实践；
> 这里回答更底层的问题：模型为什么能训练、训练为什么会不稳定、显存为什么会爆、Transformer 和扩散模型到底在优化什么。
> Transformer 在本文中只作为深度学习机制的一部分出现；完整分层阅读见
> [06-transformer-principles-overview](./06-transformer-principles-overview.md)、
> [05-transformer-from-scratch](./05-transformer-from-scratch.md) 和
> [09/01-llm-architectures](../09-frontier-models/01-llm-architectures.md)。

---

## 1. 深度学习在优化什么？

监督学习最常见的目标是经验风险最小化：

$$
\min_\theta \frac{1}{N}\sum_{i=1}^{N} \mathcal{L}(f_\theta(x_i), y_i)
$$

- `x_i`：输入，例如图像、文本 token、带噪 latent。
- `y_i`：目标，例如类别、下一个 token、真实噪声。
- `f_\theta`：带参数的神经网络。
- `L`：损失函数，把预测误差变成一个标量。

工程上所有训练脚本最后都会落到三步：

```python
logits = model(x)
loss = loss_fn(logits, y)
loss.backward()
optimizer.step()
```

不同任务只是 `x/y/loss/model` 不同：

| 任务 | 输入 | 目标 | 常用损失 |
|---|---|---|---|
| 图像分类 | 图像 | 类别 id | Cross Entropy |
| 语言模型 | 前缀 token | 下一个 token | Cross Entropy |
| 扩散模型 | 加噪图像/latent + timestep | 噪声或 velocity | MSE |
| 对比学习 | 样本对 | 正负样本关系 | InfoNCE |
| 偏好对齐 | chosen/rejected responses | 人类偏好 | DPO / PPO / GRPO |

关键点：`loss` 必须是标量，反向传播才有明确的梯度入口。

### 1.1 自监督学习为什么改变了 AI 发展路径？

早期机器学习严重依赖人工标注：图像要有人标类别，文本要有人标意图，语音要有人转写。
标注数据昂贵、覆盖面窄，模型能力很容易被任务边界限制住。

自监督学习的关键思想是：**从原始数据自身构造监督信号**。

| 任务 | 自监督信号 | 模型学到的能力 |
|---|---|---|
| 语言模型 | 给定前文预测下一个 token | 语言结构、事实关联、代码模式、推理轨迹 |
| BERT 类模型 | 根据上下文预测被 mask 的 token | 双向语义理解 |
| 对比学习 | 让匹配样本更近、不匹配样本更远 | 跨视角或跨模态表示 |
| 扩散模型 | 从带噪样本预测噪声或速度 | 数据分布的生成路径 |

这件事的革命性在于：互联网规模的原始文本、图像、音频、视频都可以变成训练数据。
模型不再只学习一个小任务的标签映射，而是在大规模数据上先学习通用表示，再通过 SFT、对齐、RAG 或工具使用适配具体任务。

难点也在这里：

- 自监督目标不是最终任务本身，next-token loss 低不等于回答可靠。
- 数据规模越大，数据质量、去重、污染、版权、分布偏差越重要。
- 预训练学到的是统计规律和可迁移表示，不保证事实实时、价值观正确或工具使用可靠。

所以现代 AIGC 的基本范式是：

```text
大规模自监督预训练 → 指令/任务微调 → 偏好对齐 → 工具/RAG/评估闭环
```

#### 深度解读：为什么“预测下一个 token”能产生通用能力？

next-token prediction 表面上只是预测词，但训练数据里包含了人类写作、推理、代码、问答、教程、日志、对话和工具说明。
模型为了降低下一个 token 的预测误差，必须把很多隐含结构压进参数里：

- 语法结构：什么样的词序、括号、缩进、代码块是合理的。
- 事实关联：某个实体、事件、概念通常和哪些描述一起出现。
- 任务模式：问题后面通常跟答案，报错后面通常跟排查步骤。
- 推理轨迹：证明、解题、程序设计、实验报告都有常见展开方式。

这就是自监督的关键洞察：不用人工给每条数据标注任务标签，只要数据规模足够大、分布足够丰富，预测缺失信息本身就会迫使模型学习可迁移表示。

但这也解释了它的边界。模型学到的是“数据分布中什么最可能出现”，不是直接学到“什么一定正确”。
所以预训练模型可能会编造、延续错误文本、复现偏见或过时知识。
工程上不能只看预训练 loss，而要通过指令数据、偏好数据、检索证据、工具校验和任务评估，把“会生成合理文本”约束成“能完成可靠任务”。

---

## 2. Tensor、形状和批处理

Tensor 的三要素是：

```text
shape + dtype + device
```

算法工程师读代码时，第一反应应当是推 shape。

### 2.1 常见 shape 约定

| 场景 | shape |
|---|---|
| 图像 batch | `(B, C, H, W)` |
| 文本 token ids | `(B, T)` |
| Transformer hidden states | `(B, T, D)` |
| Multi-head attention Q/K/V | `(B, H, T, Dh)` |
| LM logits | `(B, T, vocab_size)` |
| Diffusion latent | `(B, C, H/8, W/8)` |

`B` 是 batch size，`T` 是序列长度，`D` 是 hidden size，`H` 是 attention head 数量，`Dh = D / H`。

### 2.2 为什么深度学习偏爱矩阵乘法？

GPU 最擅长的是大规模稠密矩阵乘法。一个 `nn.Linear(Din, Dout)` 本质是：

$$
Y = XW^\top + b
$$

输入 `X: (B, Din)`，权重 `W: (Dout, Din)`，输出 `Y: (B, Dout)`。

Transformer、MLP、卷积、注意力都能规约成大量 GEMM（General Matrix Multiply）。这就是为什么模型架构设计不仅是数学问题，也是硬件效率问题。

---

## 3. 反向传播和计算图

PyTorch 使用动态图：前向执行时记录参与梯度计算的操作，`loss.backward()` 从标量 loss 反向遍历图并累积梯度。

一个参数更新可以抽象成：

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}
$$

其中 `eta` 是学习率。`optimizer.step()` 做的就是根据梯度和优化器状态更新参数。

### 3.1 梯度为什么会累积？

PyTorch 的 `.grad` 默认累加，不自动清零。这是为了支持梯度累积：

```python
loss = loss / accum_steps
loss.backward()
if (step + 1) % accum_steps == 0:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

如果你忘了 `zero_grad`，实际优化方向会混入上一个 batch 的梯度，训练通常会异常。

### 3.2 推理为什么要 `no_grad`

推理不需要反传。如果不开 `torch.no_grad()`，PyTorch 会保存中间激活，浪费显存和时间。

```python
model.eval()
with torch.no_grad():
    y = model(x)
```

`model.eval()` 控制 Dropout/BatchNorm 行为，`torch.no_grad()` 控制 autograd 记录，两者不是一回事。

---

## 4. 损失函数的本质

### 4.1 Cross Entropy

分类和语言模型最常用的是交叉熵：

$$
\mathcal{L} = -\log p_\theta(y|x)
$$

模型输出 logits，`CrossEntropyLoss` 内部做 `log_softmax + nll_loss`。不要先手动 softmax 再喂给 `CrossEntropyLoss`，这会带来数值稳定性问题。

语言模型训练时，目标是预测下一个 token：

```text
input:  [BOS, I, love, deep]
target: [I,   love, deep, learning]
```

所以 LM logits 通常是 `(B, T, vocab)`，labels 是 `(B, T)`。

### 4.2 MSE 在扩散模型中的角色

DDPM 常见训练目标：

$$
\epsilon \sim \mathcal{N}(0, I), \quad x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
$$

模型学习：

$$
\epsilon_\theta(x_t, t) \approx \epsilon
$$

代码就是：

```python
noise = torch.randn_like(clean)
noisy = scheduler.add_noise(clean, noise, timesteps)
pred = unet(noisy, timesteps).sample
loss = F.mse_loss(pred, noise)
```

扩散模型并不是直接从 prompt 生成图像；训练阶段它学的是“如何从不同噪声强度下恢复噪声方向”。

---

## 5. 优化器：为什么 AdamW 是默认选择？

SGD 更新简单，但对不同参数尺度不自适应。Adam 维护一阶矩和二阶矩：

```text
m_t: 梯度的滑动平均
v_t: 梯度平方的滑动平均
```

AdamW 把 weight decay 从梯度更新里解耦出来，通常比 Adam 更适合深度网络。

工程默认配置：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=0.01,
)
```

### 5.1 学习率比优化器更敏感

训练不稳定时，第一优先级通常不是换优化器，而是调学习率：

- loss 直接 NaN：学习率大概率过高。
- loss 很慢：学习率可能过低，或 warmup 太长。
- eval 波动大：学习率、batch size、数据噪声、正则都要排查。

大模型常用 warmup + cosine：

```text
前 3% 到 10% step 线性升温，之后 cosine 衰减
```

warmup 的核心作用是避免随机初始化早期梯度过猛。

---

## 6. 初始化、激活和归一化

### 6.1 初始化控制信号方差

深层网络需要让每一层输出方差不要持续放大或衰减。否则会出现梯度爆炸/消失。

常见经验：

- ReLU/GELU 网络用 Kaiming/Xavier 类初始化。
- Transformer 残差分支要注意随深度缩放。
- 直接从成熟框架的默认初始化开始，不要轻易自定义。

### 6.2 激活函数

| 激活 | 特点 |
|---|---|
| ReLU | 简单快，但负半轴梯度为 0 |
| GELU | BERT/GPT-2 常用，更平滑 |
| SiLU/Swish | 现代网络常用 |
| SwiGLU | LLaMA/Qwen 类 LLM 的 FFN 主流选择 |

SwiGLU：

$$
\text{SwiGLU}(x)=W_3(\text{SiLU}(W_1x) \odot W_2x)
$$

它比普通 `Linear + GELU + Linear` 参数更多，但表达能力更强，现代 LLM 基本都采用 GLU 变体。

### 6.3 LayerNorm、RMSNorm 和 Pre-Norm

LayerNorm：

$$
\text{LN}(x)=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\gamma+\beta
$$

RMSNorm：

$$
\text{RMSNorm}(x)=\frac{x}{\sqrt{\text{mean}(x^2)+\epsilon}}\gamma
$$

现代 LLM 常用 `RMSNorm + Pre-Norm`：

```text
x = x + Attention(Norm(x))
x = x + FFN(Norm(x))
```

这样残差路径直接穿过整个网络，梯度更容易传到浅层。

---

## 7. Attention 的核心推导

Scaled Dot-Product Attention：

$$
\text{Attention}(Q,K,V)=\text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

为什么除以 `sqrt(d_k)`？

如果 `q` 和 `k` 的每个元素方差约为 1，点积 `q·k` 的方差约为 `d_k`。`d_k` 大时 logits 绝对值会变大，softmax 进入饱和区，梯度变小。除以 `sqrt(d_k)` 是为了稳定方差。

### 7.1 为什么 Attention 是一次关键转折？

Attention 的重要性不只是“能算相关性”，而是它改变了序列建模的基本约束。

RNN 时代的信息流是串行的：

```text
x1 → h1 → h2 → h3 → ... → ht
```

长距离信息必须经过很多步传递，训练难并行，长依赖容易衰减。
Self-Attention 让任意两个 token 在一层里直接交互：

```text
token_i 直接读取所有 token_j 的信息
```

这带来三个关键后果：

| 变化 | 意义 | 新代价 |
|---|---|---|
| 路径变短 | 长距离依赖更容易建模 | attention matrix 随序列长度二次增长 |
| 训练可并行 | 整个序列可以一次矩阵计算 | 推理仍需自回归逐 token 生成 |
| 架构统一 | 文本 token、图像 patch、音频 codec 都能接入 | 需要位置编码和高效 kernel 支撑 |

Transformer 之所以能成为通用骨架，靠的正是这个组合：表达能力强、并行友好、容易扩大规模。
但它不是没有代价。长上下文、KV cache、FlashAttention、PagedAttention、GQA/MQA 等后续技术，都是在处理 Attention 带来的计算和内存压力。

#### 深度解读：Transformer 革命性不只来自 Attention

Transformer 真正强的地方，是几种条件同时成立：

| 条件 | 作用 |
|---|---|
| Self-Attention | 任意 token 可以直接读取全局上下文 |
| 残差连接和归一化 | 深层网络更容易训练 |
| FFN / MLP 块 | 为每个 token 提供非线性变换能力 |
| 均质 block 堆叠 | 架构简单，容易 scale 到更多层和更大 hidden size |
| GEMM 友好 | 大部分计算能落到 GPU 擅长的矩阵乘法 |

如果只有 Attention，没有稳定的深层训练和硬件友好的大矩阵计算，它不会成为大模型主干。
如果只有并行计算，没有全局 token 交互，它也很难统一文本、代码、图像 patch、音频 codec 和视频 patch。

这也是为什么现代架构改动看起来很多，底层仍围绕 Transformer 变体展开：

- RoPE、ALiBi、位置插值：处理位置信息和长上下文。
- GQA/MQA/MLA：降低 KV cache 和 decode 带宽压力。
- FlashAttention：减少 attention 中间矩阵的 HBM 读写。
- MoE：扩大总参数，同时控制每个 token 的激活计算。

工程判断上，读一个 Transformer 变体不要只问“它多了什么模块”，要问这个模块在解决哪类约束：训练稳定性、上下文长度、推理显存、通信成本，还是跨模态接入。

### 7.2 Multi-Head 不一定增加参数

假设 `d_model=4096`，`n_heads=32`，`d_head=128`。

QKV 投影仍然是：

```text
Linear(4096, 3 * 4096)
```

head 只是把最后一维 reshape 成 `(32, 128)`，不是给每个 head 单独增加一套完整 `4096 x 4096` 权重。

### 7.3 因果掩码的含义

自回归模型必须保证第 `t` 个位置不能看到未来 token。训练时虽然整个序列并行输入，但 mask 会阻止未来信息泄漏。

```text
q0: k0
q1: k0 k1
q2: k0 k1 k2
```

如果因果 mask 写错，训练 loss 会异常低，但推理效果会崩，因为模型训练时偷看了答案。

---

## 8. 位置编码：从绝对位置到 RoPE

Self-attention 本身对 token 顺序不敏感。必须注入位置信息。

### 8.1 绝对位置编码

原始 Transformer 用固定正余弦编码，GPT-2 用可学习 position embedding。它们通常在输入端加一次：

```python
x = token_embedding + position_embedding
```

问题是长度外推差：训练时只见过 2048，推理到 8192 时未必可靠。

### 8.2 RoPE

RoPE 把位置编码作用在每层 attention 的 Q/K 上，本质是对二维子空间做旋转。旋转后的 `q_i · k_j` 会携带相对位置信息 `i-j`。

工程影响：

- LLaMA/Qwen/Mistral 等主流 LLM 都使用 RoPE。
- 长上下文扩展通常围绕 RoPE scaling 做文章。
- KV cache 中缓存的是已经带位置旋转后的 K/V。

---

## 9. KV Cache 与推理复杂度

训练时可以并行处理整个序列；自回归推理时必须一个 token 一个 token 生成。

没有 KV cache：

```text
每步重新计算整个前缀，浪费巨大
```

有 KV cache：

```text
历史 K/V 只算一次，新 token 只追加一行 K/V
```

KV cache 显存估算：

```text
2 * batch * seq_len * n_layers * n_kv_heads * d_head * dtype_bytes
```

这里的 `2` 是 K 和 V。GQA/MQA 通过减少 `n_kv_heads` 来显著降低 KV cache 显存。

---

## 10. 扩散模型的理论直觉

扩散模型包含两个过程：

| 过程 | 含义 |
|---|---|
| Forward process | 逐步给数据加高斯噪声 |
| Reverse process | 神经网络学习逐步去噪 |

DDPM 前向过程有闭式形式：

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
$$

所以训练时不需要一步步加噪，可以直接采样任意 `t`。

### 10.1 Scheduler 做什么？

UNet/DiT 预测噪声方向，Scheduler 决定每一步如何从 `x_t` 更新到 `x_{t-1}`。

换 scheduler 不换模型，也能改变采样速度和质量：

- DDPM：经典但慢。
- DDIM：确定性路径，可少步采样。
- DPM-Solver：少步高质量，工程常用。
- Euler/Heun：SDXL、ComfyUI 生态常见。
- LCM：极少步采样，但通常要配合蒸馏或 LoRA。

### 10.2 CFG 为什么有效？

Classifier-Free Guidance 同时计算有条件和无条件预测：

$$
\hat{\epsilon} = \epsilon_\text{uncond} + s(\epsilon_\text{cond} - \epsilon_\text{uncond})
$$

`s` 就是 `guidance_scale`。它放大 prompt 条件方向：

- 太低：图像和 prompt 关系弱。
- 适中：语义一致性更好。
- 太高：过饱和、伪影、构图僵硬。

---

## 11. 显存预算：必须会算

训练显存主要包括：

```text
参数 + 梯度 + 优化器状态 + 激活 + 临时 buffer
```

AdamW 全参训练的静态开销粗略是 `16N` 字节：

| 组件 | 字节/参数 |
|---|---|
| bf16/fp16 参数 | 2 |
| fp32 master weights | 4 |
| 梯度 | 2 或 4 |
| Adam m/v | 8 |

7B 模型全参训练静态开销就超过 100GB，还没算激活。这就是为什么真实训练需要 LoRA、ZeRO/FSDP、gradient checkpointing、混合精度和多卡。

### 11.1 激活为什么常常比参数更麻烦？

激活和 batch size、seq length、层数强相关。Transformer attention 的中间矩阵理论上有 `(B, H, T, T)`，长序列时会快速爆炸。

FlashAttention 的核心价值不是改变数学公式，而是避免显式保存完整 attention matrix，降低 HBM 读写和显存压力。

---

## 12. 混合精度和数值稳定性

### 12.1 fp16 vs bf16

| dtype | 优点 | 风险 |
|---|---|---|
| fp32 | 稳定 | 慢、占显存 |
| fp16 | 快、省显存 | 动态范围小，容易 overflow |
| bf16 | 快、省显存，动态范围接近 fp32 | 精度尾数少 |

Ampere 及以上 GPU 通常优先 bf16。老卡使用 fp16 时需要 `GradScaler`。

### 12.2 NaN 常见来源

- 学习率过高。
- 手写 softmax 没减 max，指数溢出。
- attention 忘记除以 `sqrt(d_k)`。
- 除以很小的数，没有加 epsilon。
- fp16 下 logits 或 loss scale 溢出。
- 标签越界，CUDA 上表现成 device-side assert。

训练稳定性排查要先用小 batch、小数据、CPU 或确定性输入复现。

---

## 13. 泛化、正则和过拟合

深度模型参数远多于训练样本时仍能泛化，原因很复杂；工程上重点是监控 train/eval gap。

常用正则：

- `weight_decay`：限制权重过大。
- `dropout`：随机丢激活，降低共适应。
- `data augmentation`：提升输入多样性。
- `label smoothing`：分类任务中避免过度自信。
- `early stopping`：验证集不再提升就停止。

但在大模型预训练里，最重要的“正则”往往是数据规模和数据质量。

---

## 14. 从理论映射到本模块代码

| 理论点 | 对应示例 |
|---|---|
| Tensor/autograd/module/device | `examples/pytorch_basics.py` |
| 训练闭环、优化器、scheduler、checkpoint | `examples/mlp_mnist.py --synthetic` |
| Tokenizer、generate、streaming、pipeline | `examples/transformers_quickstart.py` |
| DDPM 加噪、噪声预测、采样 | `examples/diffusers_quickstart.py` |
| RoPE、RMSNorm、SwiGLU、KV cache、采样 | `examples/transformer_from_scratch.py` |

推荐学习顺序：

1. 先读本文，把核心概念建立起来。
2. 跑通 `pytorch_basics.py` 和 `mlp_mnist.py --synthetic`。
3. 读 `05-transformer-from-scratch.md`，再跑 `transformer_from_scratch.py`。
4. 再进入 HuggingFace Transformers 和 Diffusers 的真实模型生态。

---

## 15. 工程判断清单

写或审训练代码时，至少检查：

- [ ] 每个关键 tensor 的 shape 是否符合预期。
- [ ] 模型和数据是否在同一 device。
- [ ] loss 是否是标量，labels dtype/range 是否正确。
- [ ] `zero_grad -> backward -> clip -> step -> scheduler.step` 顺序是否正确。
- [ ] eval 是否用了 `model.eval()` 和 `torch.no_grad()`。
- [ ] 是否能用一个 batch 过拟合。
- [ ] 是否有随机种子和可复现实验配置。
- [ ] checkpoint 是否只保存 `state_dict` 和必要 config。
- [ ] 长序列/大图训练前是否估算了显存。
- [ ] 真实模型脚本是否有离线 smoke test 路径。

理论不是为了背公式，而是为了在模型不收敛、显存爆、速度慢、效果差时知道该从哪里下手。
