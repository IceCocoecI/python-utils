# 模块 09：前沿 AIGC 模型架构

> 了解模型架构是 AIGC 算法工程师的**核心素养**。
> 你不需要能从零训练 GPT-4、Claude、Gemini、Qwen 或 DeepSeek 级别的模型，
> 但你必须能看懂公开模型的架构图、`config.json`、技术报告和模型卡，
> 并能把这些判断落到微调、推理优化、应用开发和模型选型上。

---

## 为什么理解模型架构至关重要？

很多同学把模型当黑箱：下载权重 → 调 API → 出结果。这在应用层够用，但在算法层远远不够：

- **微调调不好**：不懂 GQA 就不知道为什么 LoRA 加在 q_proj/v_proj 效果差异大
- **推理优不动**：不懂 KV Cache 就不知道为什么长上下文 OOM
- **论文读不懂**：MLA、MoE、Flow Matching——不懂架构，论文全是天书
- **选型选不对**：什么时候用 Dense 模型，什么时候用 MoE？什么时候用 DiT 而不是 U-Net？
- **面试答不上**：架构题是 AIGC 岗位面试的重中之重

本模块带你系统梳理近几年 AIGC 领域主流和前沿的模型架构。

> 前沿模型更新很快。本文档按 `2026-06-30` 的公开资料组织学习框架；
> 对闭源模型只讨论公开信息和可验证的工程现象，不把社区传闻当成架构事实。

---

## AIGC 模型全景图

```
                          ┌─────────────────────────────────────────────┐
                          │            AIGC 模型架构全景                  │
                          └──────────────────┬──────────────────────────┘
                                             │
          ┌──────────────┬───────────────┬────┴────┬───────────────┐
          ▼              ▼               ▼         ▼               ▼
     ┌─────────┐   ┌──────────┐   ┌──────────┐ ┌───────┐   ┌──────────┐
     │  文本生成 │   │ 图像生成  │   │ 多模态    │ │ 视频   │   │ 语音/音频 │
     │  (LLM)   │   │          │   │          │ │ 生成   │   │          │
     └────┬─────┘   └────┬─────┘   └────┬─────┘ └───┬───┘   └────┬─────┘
          │              │              │            │            │
     GPT/LLaMA      SD/DiT/Flux     CLIP/LLaVA   Sora/Wan    VALL-E
     Qwen/DeepSeek  ControlNet      Qwen-VL      CogVideoX   CosyVoice
     Mistral/MoE    SD3/SDXL        InternVL     HunyuanVideo F5-TTS
     Qwen3/R1       Rectified Flow  GPT-4o类     Veo/Kling    Qwen2.5-Omni
```

---

## 学习内容

| # | 文档 | 核心话题 |
|---|---|---|
| 00 | [frontier-models-theory](./00-frontier-models-theory.md) | AIGC 模型统一理论：token/latent/patch、生成模型家族、跨模态架构、Scaling 与选型 |
| 01 | [llm-architectures](./01-llm-architectures.md) | Transformer 家族、GPT/LLaMA/Qwen/Mistral/DeepSeek 架构演进、MoE、长上下文、Scaling Laws |
| 02 | [image-generation](./02-image-generation.md) | GAN → VAE → Diffusion → Flow Matching、SD/SDXL/DiT/Flux/SD3、采样算法、ControlNet |
| 03 | [multimodal-models](./03-multimodal-models.md) | CLIP/SigLIP、BLIP-2、LLaVA、Qwen-VL、InternVL、视觉语言模型架构范式 |
| 04 | [video-generation](./04-video-generation.md) | 视频扩散架构、Sora 系、CogVideoX、Wan、HunyuanVideo、时序一致性 |
| 05 | [speech-and-audio](./05-speech-and-audio.md) | 神经音频编解码、VALL-E、CosyVoice、F5-TTS、Whisper、音乐生成 |

---

## 推荐阅读顺序

### 第 0 步：统一理论框架

先读 00，建立“输入表示 → 主干网络 → 条件注入 → 训练目标 → 采样/解码 → 评估瓶颈”的读模型方法。

### 第 1 步：LLM 架构（最核心）

先学 01，因为 Transformer 是所有 AIGC 模型的基础。不管你做图像、视频还是音频，
底层都绑着 Transformer。

### 第 2 步：图像生成

学 02，理解扩散模型的演进。从 U-Net 到 DiT，从 DDPM 到 Flow Matching——
这条线索贯穿了图像、视频、音频三个领域。

### 第 3 步：多模态 → 视频 → 语音

03 → 04 → 05 按兴趣选读。多模态和视频联系紧密，可以连着看。

---

## 理论与实践怎么组织

本模块建议按三层学习：

| 层次 | 要回答的问题 | 对应材料 |
|---|---|---|
| 统一架构层 | Transformer、扩散、Flow Matching、多模态对齐如何成为不同模态的共同底座？ | `00-frontier-models-theory.md` |
| 模态专题层 | 文本、图像、多模态、视频、语音各自的结构演进和核心瓶颈是什么？ | `01` ~ `05` 文档 |
| 选型判断层 | 什么时候用 Dense/MoE、U-Net/DiT、VLM/Any-to-Any、codec/audio LM？ | 各专题的对比表、时间线和常见坑 |
| 交付输出层 | 如何把论文和模型卡转成一页架构评审、资源估算和风险清单？ | `00` 的评审模板 + 各专题实践任务 |

学习顺序建议：

1. 先读 `00`，建立统一的模型架构读法。
2. 再读 `01`，把 Decoder-only、GQA、MoE、长上下文和 Scaling Laws 打牢。
3. 继续读 `02`，理解扩散和 Flow Matching，这条线会延伸到图像、视频、音频。
4. 最后按兴趣读多模态、视频和语音专题。

---

## 学完本模块应该产出什么？

不要只停留在“知道模型名”。完成本模块后，建议至少做出 4 份可复用材料：

| 产出 | 内容 | 价值 |
|---|---|---|
| 架构拆解卡 | 任选一个开源 LLM，拆 `config.json`、参数量、KV cache、上下文、tokenizer | 面试和选型都能直接复用 |
| 图像/视频 pipeline 图 | 标出 text encoder、VAE、DiT/U-Net、scheduler、guidance 的数据流和 shape | 防止把模型调用当黑箱 |
| 多模态 token 预算表 | 对比单图、多图、视频、音频输入会占多少 token 或 latent frame | 直接服务产品成本估算 |
| 模型选型建议书 | 在给定硬件、延迟、质量、安全要求下选 Dense/MoE/VLM/TTS/视频模型 | 训练“技术判断”而不是背诵 |

建议模板：

```text
模型名称：
任务目标：
输入表示：
主干架构：
训练目标：
推理瓶颈：
硬件预算：
主要风险：
是否适合当前业务：
```

---

## 推荐论文与资源

### 必读论文

| 论文 | 年份 | 关键贡献 |
|---|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Transformer 架构 |
| [LLaMA: Open and Efficient Foundation LMs](https://arxiv.org/abs/2302.13971) | 2023 | 开源 LLM 里程碑 |
| [LLaMA 2](https://arxiv.org/abs/2307.09288) | 2023 | GQA、安全对齐 |
| [LLaMA 3](https://arxiv.org/abs/2407.21783) | 2024 | 扩展至 405B |
| [Mixtral of Experts](https://arxiv.org/abs/2401.04088) | 2024 | 开源 MoE |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 2024 | MLA + DeepSeekMoE |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 2024 | 671B MoE |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 2025 | 大规模 RL 激发推理能力 |
| [Qwen2.5](https://arxiv.org/abs/2412.15115) | 2024 | 多尺寸、多语言、18T tokens |
| [Qwen3](https://arxiv.org/abs/2505.09388) | 2025 | Dense/MoE + thinking/non-thinking 统一 |
| [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | 2020 | DDPM |
| [High-Resolution Image Synthesis with LDM](https://arxiv.org/abs/2112.10752) | 2022 | Stable Diffusion |
| [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) | 2023 | DiT |
| [Scaling Rectified Flow Transformers](https://arxiv.org/abs/2403.03206) | 2024 | SD3 / MMDiT / Rectified Flow |
| [FLUX.1 Kontext](https://arxiv.org/abs/2506.15742) | 2025 | 图像生成与编辑统一 flow 模型 |
| [CLIP: Learning Transferable Visual Representations](https://arxiv.org/abs/2103.00020) | 2021 | 图文对齐 |
| [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) | 2023 | 视觉语言模型 |
| [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) | 2025 | 动态分辨率、文档/视频理解 |
| [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215) | 2025 | 文本/图像/音频/视频输入，文本/语音输出 |
| [HunyuanVideo](https://arxiv.org/abs/2412.03603) | 2024 | 13B 开源视频生成框架 |
| [Wan](https://arxiv.org/abs/2503.20314) | 2025 | 开源大规模视频生成模型套件 |
| [Neural Codec Language Models for Zero-Shot TTS](https://arxiv.org/abs/2301.02111) | 2023 | VALL-E |
| [CosyVoice 2](https://arxiv.org/abs/2412.10117) | 2024 | 流式语音合成与 LLM 语音建模 |
| [F5-TTS](https://arxiv.org/abs/2410.06885) | 2024 | Flow Matching 非自回归 TTS |
| [MaskGCT](https://arxiv.org/abs/2409.00750) | 2024 | Masked codec TTS |

### 推荐博客 / 教程

| 资源 | 说明 |
|---|---|
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 经典图解 Transformer |
| [Lilian Weng's Blog](https://lilianweng.github.io/) | 最好的 AI 技术博客之一 |
| [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) | 带代码的扩散模型讲解 |
| [Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | 手撸 GPT 视频 |
| [HuggingFace Model Cards](https://huggingface.co/models) | 各模型架构详情 |
| [Papers With Code](https://paperswithcode.com/) | 论文 + 代码 + 排行榜 |

---

## 自检清单

学完本模块，你应该能自信地回答以下问题：

- [ ] Decoder-only 和 Encoder-Decoder 架构各适合什么任务？为什么 LLM 主流是 Decoder-only？
- [ ] RoPE 比 Sinusoidal PE 好在哪？为什么它天然支持长度外推？
- [ ] GQA 和 MQA 分别是什么？它们如何减少 KV Cache 的显存占用？
- [ ] MoE 架构中 Top-K 路由是什么？Load Balancing Loss 解决什么问题？
- [ ] DeepSeek-V2 的 MLA 解决了什么问题？和标准 MHA 有什么区别？
- [ ] 扩散模型的前向过程和反向过程分别做什么？训练的是哪个过程？
- [ ] Latent Diffusion 和 Pixel-space Diffusion 的核心区别是什么？
- [ ] DiT 相比 U-Net 的优势是什么？为什么 DiT 成为新一代主流架构？
- [ ] Classifier-Free Guidance (CFG) 的原理是什么？guidance scale 调大会怎样？
- [ ] CLIP 的训练目标是什么？为什么它能做零样本分类？
- [ ] LLaVA 的架构是什么？图像 token 是怎么进入 LLM 的？
- [ ] 视频生成相比图像生成，核心挑战多了什么？
- [ ] 神经音频编解码器（如 EnCodec）的作用是什么？为什么它让 TTS 发生了范式转变？
- [ ] Flow Matching 和 DDPM 的训练目标有什么区别？
- [ ] 什么是 Scaling Law？Chinchilla 最优比例是什么？
- [ ] 为什么 Qwen3 这类模型要同时提供 thinking 和 non-thinking 模式？它影响哪些推理成本？
- [ ] 闭源多模态模型的架构信息不完整时，哪些结论可以说，哪些只能说是推断？
- [ ] 给定 24GB / 48GB / 80GB GPU，如何粗略判断一个 LLM、VLM 或视频模型能不能本地跑？
- [ ] 评估图像、视频、语音模型时，哪些 benchmark 容易和真实产品体验脱节？
