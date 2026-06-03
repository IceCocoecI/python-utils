# 模块 09：前沿 AIGC 模型架构

> 了解模型架构是 AIGC 算法工程师的**核心素养**。
> 你不需要能从零训练 GPT-4，但你必须能看懂它的架构图、理解每个组件的设计动机、
> 并能在此基础上做微调、推理优化和应用开发。

---

## 为什么理解模型架构至关重要？

很多同学把模型当黑箱：下载权重 → 调 API → 出结果。这在应用层够用，但在算法层远远不够：

- **微调调不好**：不懂 GQA 就不知道为什么 LoRA 加在 q_proj/v_proj 效果差异大
- **推理优不动**：不懂 KV Cache 就不知道为什么长上下文 OOM
- **论文读不懂**：MLA、MoE、Flow Matching——不懂架构，论文全是天书
- **选型选不对**：什么时候用 Dense 模型，什么时候用 MoE？什么时候用 DiT 而不是 U-Net？
- **面试答不上**：架构题是 AIGC 岗位面试的重中之重

本模块带你系统梳理 2024–2025 年 AIGC 领域最前沿的模型架构。

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
     Mistral/MoE    SD3/SDXL        InternVL     HunyuanV    F5-TTS
```

---

## 学习内容

| # | 文档 | 核心话题 |
|---|---|---|
| 01 | [llm-architectures](./01-llm-architectures.md) | Transformer 家族、GPT/LLaMA/Qwen/Mistral/DeepSeek 架构演进、MoE、长上下文、Scaling Laws |
| 02 | [image-generation](./02-image-generation.md) | GAN → VAE → Diffusion → Flow Matching、SD/SDXL/DiT/Flux/SD3、采样算法、ControlNet |
| 03 | [multimodal-models](./03-multimodal-models.md) | CLIP/SigLIP、BLIP-2、LLaVA、Qwen-VL、InternVL、视觉语言模型架构范式 |
| 04 | [video-generation](./04-video-generation.md) | 视频扩散架构、Sora 系、CogVideoX、Wan、HunyuanVideo、时序一致性 |
| 05 | [speech-and-audio](./05-speech-and-audio.md) | 神经音频编解码、VALL-E、CosyVoice、F5-TTS、Whisper、音乐生成 |

---

## 推荐阅读顺序

### 第 1 步：LLM 架构（最核心）

先学 01，因为 Transformer 是所有 AIGC 模型的基础。不管你做图像、视频还是音频，
底层都绑着 Transformer。

### 第 2 步：图像生成

学 02，理解扩散模型的演进。从 U-Net 到 DiT，从 DDPM 到 Flow Matching——
这条线索贯穿了图像、视频、音频三个领域。

### 第 3 步：多模态 → 视频 → 语音

03 → 04 → 05 按兴趣选读。多模态和视频联系紧密，可以连着看。

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
| [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | 2020 | DDPM |
| [High-Resolution Image Synthesis with LDM](https://arxiv.org/abs/2112.10752) | 2022 | Stable Diffusion |
| [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) | 2023 | DiT |
| [CLIP: Learning Transferable Visual Representations](https://arxiv.org/abs/2103.00020) | 2021 | 图文对齐 |
| [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) | 2023 | 视觉语言模型 |
| [Neural Codec Language Models for Zero-Shot TTS](https://arxiv.org/abs/2301.02111) | 2023 | VALL-E |

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
