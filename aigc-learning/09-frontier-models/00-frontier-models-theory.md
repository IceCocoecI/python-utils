# 00 · 前沿 AIGC 模型理论框架

> 本文是模块 09 的理论总览。
> 后续 LLM、图像生成、多模态、视频生成、语音音频专题都可以看成同一个问题的不同实例：
> 如何把一种模态表示成模型可处理的 token / latent / feature，并学习可控的生成或理解过程。

---

## 1. 本模块真正解决什么问题？

模块 02 解决“Transformer 和深度学习基础怎么工作”，模块 09 解决“现代 AIGC 模型为什么这样设计”。

你需要能回答：

- 为什么 LLM 主流是 decoder-only？
- 为什么图像生成从 GAN/VAE 走向 Diffusion/Flow Matching？
- 为什么 DiT、MMDiT、视频 patch、音频 codec token 都在向 Transformer 靠拢？
- 为什么 MoE、GQA、MLA、长上下文、KV cache 会成为架构重点？
- 一个模型架构的瓶颈在数据、参数、算力、上下文、模态对齐，还是采样过程？

模型架构学习的重点不是背模型名，而是建立结构化读法：

```text
输入表示 → 主干网络 → 条件注入 → 训练目标 → 采样/解码 → 评估与瓶颈
```

---

## 2. 统一视角：把世界变成可建模的序列或场

AIGC 模型处理不同模态，但底层通常先把数据变成某种可学习表示。

| 模态 | 常见表示 | 典型模型 |
|---|---|---|
| 文本 | token ids | GPT、LLaMA、Qwen |
| 图像 | pixels、latent、patch tokens | Stable Diffusion、DiT、Flux |
| 视频 | latent frames、spacetime patches | Sora 类模型、CogVideoX、Wan |
| 语音 | waveform、mel、codec tokens | Whisper、VALL-E、CosyVoice |
| 多模态 | text tokens + visual tokens | CLIP、LLaVA、Qwen-VL |

两个核心方向：

1. **离散化**：把模态变成 token 序列，用语言模型式目标建模。
2. **连续化**：把模态变成 latent / feature，用扩散或 flow 建模连续分布。

---

## 3. 生成模型的四大家族

| 家族 | 核心思想 | 优点 | 典型场景 |
|---|---|---|---|
| Autoregressive | 逐 token 预测下一个 token | 简单、可扩展、适合离散序列 | LLM、代码、codec audio |
| Diffusion | 学习从噪声逐步去噪 | 质量高、模式覆盖好 | 图像、视频、音频 |
| Flow Matching | 学习从噪声到数据的连续速度场 | 训练和采样路径更直接 | 新一代图像/视频生成 |
| VAE / Codec | 学习压缩表示和重建 | 降低维度、提供 latent/token | Stable Diffusion VAE、音频 codec |

GAN 仍然重要，但在通用 AIGC 路线里，扩展性和训练稳定性让 Diffusion / Flow / Transformer 成为主线。

---

## 4. Transformer 为什么成为统一骨架？

Transformer 的核心优势：

- 接受可变长度序列；
- 结构均匀，容易 scale；
- attention 可以建模远距离依赖；
- 大部分计算可以变成 GEMM，适合 GPU；
- 可以统一文本 token、图像 patch、视频 patch、音频 token。

它在不同模态中的角色不同：

| 场景 | Transformer 的角色 |
|---|---|
| LLM | 直接作为自回归主干 |
| DiT | 替代 U-Net 做扩散去噪网络 |
| VLM | 连接视觉 token 和文本 token |
| 视频生成 | 建模空间和时间 patch 之间的依赖 |
| 音频生成 | 对 codec token 做语言建模 |

所以模块 02 的 Transformer 基础是“数学和实现”，模块 09 的 Transformer 是“架构演进和跨模态迁移”。

---

## 5. LLM 架构主线

现代 LLM 通常是 decoder-only Transformer：

```text
token ids
  ↓
embedding + position
  ↓
N × decoder block
  ↓
lm head
  ↓
next-token distribution
```

关键设计点：

| 设计 | 解决的问题 |
|---|---|
| Pre-Norm / RMSNorm | 深层训练稳定性 |
| SwiGLU | FFN 表达能力和训练效果 |
| RoPE | 相对位置信息和长上下文外推 |
| GQA / MQA | 降低 KV cache 显存和带宽 |
| MoE | 增大总参数，同时控制每 token 激活计算 |
| MLA / KV 压缩 | 降低长上下文推理成本 |

看一个 LLM 架构时，优先看：

- hidden size；
- layers；
- attention heads 和 kv heads；
- intermediate size；
- context length；
- position encoding；
- dense 还是 MoE；
- tokenizer 和 vocab；
- 训练数据和对齐方法。

---

## 6. 图像生成主线

图像生成经历了几个关键阶段：

```text
GAN → VAE → Pixel Diffusion → Latent Diffusion → DiT / MMDiT → Flow Matching
```

### 6.1 为什么 Latent Diffusion 重要？

直接在像素空间扩散成本太高。
Latent Diffusion 先用 VAE 把图像压缩到低维 latent，再在 latent 空间去噪：

```text
image → VAE encoder → latent
latent diffusion denoising
latent → VAE decoder → image
```

这让高分辨率文生图变得可训练、可推理。

### 6.2 U-Net 到 DiT

U-Net 擅长多尺度空间建模，是早期扩散模型主干。
DiT 把图像 latent 切成 patch，用 Transformer 处理：

| U-Net | DiT |
|---|---|
| 卷积归纳偏置强 | 规模化更直接 |
| 多尺度结构天然 | 依赖 patch 和位置编码 |
| 工程成熟 | 更接近 LLM 的 scaling 路线 |

DiT 的意义不是“Transformer 更时髦”，而是让图像生成更接近大模型 scaling 规律。

### 6.3 Flow Matching

Diffusion 通常学习去噪过程，Flow Matching 更直接学习从噪声分布到数据分布的速度场。

直觉：

```text
Diffusion: 多步去噪，学如何反转加噪
Flow: 学一条从噪声到数据的连续路径
```

新一代图像/视频模型大量采用 flow 相关训练目标，是因为它在采样效率和训练路径上更有优势。

---

## 7. 多模态架构主线

多模态模型的关键问题是：不同模态如何进入同一个语义空间或同一个 LLM。

常见范式：

| 范式 | 做法 | 代表 |
|---|---|---|
| 对比学习 | 图像和文本分别编码，对齐 embedding 空间 | CLIP、SigLIP |
| Visual Tokens | 图像编码成 token，接入 LLM 上下文 | LLaVA、Qwen-VL |
| Cross-Attention | LLM 通过 cross-attention 读取视觉特征 | Flamingo 类 |
| Any-to-Any | 文本、图像、音频都 token 化并统一生成 | 新一代多模态系统 |

架构判断重点：

- 视觉编码器是否冻结；
- projector / adapter 怎么设计；
- 图像 token 数量如何控制；
- 分辨率如何处理；
- 训练数据是图文对齐、指令微调，还是多任务混合；
- 是否支持 grounding、OCR、视频、多图。

---

## 8. 视频生成主线

视频比图像多了时间维度，核心难点是：

- 时序一致性；
- 运动建模；
- 长视频记忆；
- 训练数据质量；
- 推理成本和显存；
- 评估困难。

常见路线：

```text
图像扩散模型 + 时间层
  ↓
3D U-Net / Temporal Attention
  ↓
Spacetime Patch Transformer
  ↓
Text-to-Video / Image-to-Video / Video Editing
```

视频模型的架构重点：

- latent 是否压缩时间维度；
- attention 是空间、时间分开做，还是统一做；
- 训练分辨率、帧率、时长如何 bucket；
- 文本条件如何注入；
- 是否支持首帧/尾帧/参考图控制。

---

## 9. 语音与音频主线

音频模型有两条主线：

| 方向 | 表示 | 典型任务 |
|---|---|---|
| 连续声学表示 | waveform、mel spectrogram | ASR、TTS、音频理解 |
| 离散 codec token | neural audio codec | 语音克隆、音乐生成、语音 LM |

神经音频 codec 的意义是把连续音频压缩成离散 token：

```text
waveform → codec encoder → discrete codes → language model → codec decoder → waveform
```

这让 TTS 和音频生成可以复用 LLM 的自回归建模能力。

---

## 10. Scaling Laws 与架构取舍

模型能力通常由三类资源共同决定：

```text
能力 ≈ f(参数量, 数据量, 计算量)
```

但不同架构的瓶颈不同：

| 架构 | 主要瓶颈 |
|---|---|
| Dense LLM | 训练算力、推理权重带宽、KV cache |
| MoE LLM | 路由、负载均衡、通信、专家并行 |
| Diffusion / Flow | 采样步数、去噪网络成本、条件控制 |
| VLM | 视觉 token 数、对齐数据、分辨率 |
| Video | 时空 token 数、显存、数据质量 |
| Audio LM | codec 质量、时序长度、韵律控制 |

架构设计本质是在这些约束之间做取舍：

- 质量 vs 速度；
- 参数量 vs 激活计算；
- 上下文长度 vs KV cache；
- 分辨率/帧数 vs 显存；
- 统一架构 vs 模态专用归纳偏置；
- 开源可部署性 vs 最高效果。

---

## 11. 如何读一篇模型论文或 config

建议按这个顺序读：

1. 输入表示：token、patch、latent、codec，shape 是什么？
2. 主干网络：Transformer、U-Net、hybrid、MoE？
3. 条件注入：prompt、图像、音频、控制信号如何进入模型？
4. 训练目标：next-token、denoising、flow matching、contrastive、preference？
5. 采样/解码：自回归、扩散步数、ODE/SDE solver、guidance？
6. 扩展策略：参数、数据、上下文、分辨率、专家数怎么 scale？
7. 推理成本：权重、KV cache、activation、采样步数哪个最贵？
8. 评估：用什么 benchmark，是否覆盖真实应用失败模式？

---

## 12. 从理论映射到本模块文档

| 理论问题 | 对应文档 |
|---|---|
| LLM 架构如何从原始 Transformer 演进到 LLaMA/Qwen/DeepSeek/MoE？ | [01-llm-architectures](./01-llm-architectures.md) |
| 图像生成如何从 GAN/VAE 演进到 Diffusion、DiT、Flux、Flow Matching？ | [02-image-generation](./02-image-generation.md) |
| CLIP、VLM、视觉 token、Any-to-Any 多模态系统如何组织？ | [03-multimodal-models](./03-multimodal-models.md) |
| 视频生成为什么比图像生成更难，时空架构如何设计？ | [04-video-generation](./04-video-generation.md) |
| 语音、音频 codec、TTS、ASR、音乐生成如何统一理解？ | [05-speech-and-audio](./05-speech-and-audio.md) |

---

## 13. 和模块 02 的关系

Transformer 相关内容分三层：

| 层次 | 位置 | 作用 |
|---|---|---|
| 概念速览 | [02/06 Transformer 原理白话速览](../02-deep-learning-libraries/06-transformer-principles-overview.md) | 第一遍建立直觉，不写代码 |
| 公式与实现 | [02/05 Transformer 架构深度剖析](../02-deep-learning-libraries/05-transformer-from-scratch.md) | 手写 Attention、RoPE、mask、KV cache |
| 架构演进 | [09/01 LLM 架构全解](./01-llm-architectures.md) | 解释 LLaMA/Qwen/DeepSeek/MoE 等现代变体 |

读法建议：

```text
02/06 → 02/05 → 09/00 → 09/01
```

如果已经理解基础 Transformer，可以直接从本文件和 `09/01` 开始看现代架构差异。

---

## 14. 工程判断清单

- [ ] 这个模型的输入表示是什么？token、latent、patch 还是 codec？
- [ ] 主干网络是什么？Transformer、U-Net、hybrid 还是 MoE？
- [ ] 条件信号如何注入？cross-attention、adapter、AdaLN、prompt token？
- [ ] 训练目标是什么？next-token、denoising、flow、contrastive？
- [ ] 推理瓶颈在哪里？权重带宽、KV cache、采样步数、token 数？
- [ ] 这个架构适合 scale 参数，还是 scale 数据/分辨率/上下文？
- [ ] 是否有明确的压缩模块，例如 VAE、codec、patch embedding？
- [ ] 多模态对齐发生在哪一层？embedding 空间、token 空间，还是 cross-attention？
- [ ] 评估指标是否真的覆盖目标应用？
- [ ] 开源实现能否在你的硬件预算内推理或微调？

