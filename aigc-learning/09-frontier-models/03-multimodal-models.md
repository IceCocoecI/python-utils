# 03 · 多模态模型架构全解

> 目标：理解"看图说话"背后的技术——从 CLIP 对齐到 LLaVA 视觉指令跟随。
> 多模态是 2024–2025 年 AIGC 的核心战场，几乎所有主流模型都在走向"全模态"。
> 本文重点关注可验证的 VLM 架构范式；闭源 omni 模型只讨论公开能力和接口现象。

---

## 1. 什么是多模态？

**多模态 = 让模型同时理解和处理多种数据类型（文本、图像、音频、视频）**。

```
单模态时代：
  LLM:  文本 → 文本
  ViT:  图像 → 分类标签

多模态时代：
  VLM:  图像 + 文本 → 文本（看图回答问题）
  生成: 文本 → 图像/视频/音频
  统一: 任意模态 → 任意模态（GPT-4o, Gemini）
```

本节聚焦 **Vision-Language Models (VLMs)**——连接视觉和语言的模型。

### 1.1 多模态系统的三种目标

多模态不是一个任务，而是一组目标：

| 目标 | 输入输出 | 代表任务 | 架构重点 |
|---|---|---|---|
| 理解 | 图像/视频/音频 + 文本 → 文本 | VQA、OCR、文档问答、图表分析 | 编码器、token 压缩、LLM 对齐 |
| 生成 | 文本/图像 → 图像/视频/音频 | 文生图、图生视频、TTS | diffusion/flow/codec、条件注入 |
| 交互 | 多模态输入 → 文本/语音/动作 | 实时语音助手、屏幕操作 Agent | 低延迟、流式、工具调用、状态管理 |

这三类目标经常共用 LLM，但架构瓶颈不同。理解模型最怕 token 太多，生成模型最怕采样太慢，交互系统最怕端到端延迟和中断处理差。

---

## 2. CLIP：视觉-语言对齐的基石

### 2.1 核心思想

CLIP (Contrastive Language-Image Pre-training, Radford et al., 2021) 的核心：
**用对比学习让图像和文本在同一个嵌入空间中对齐**。

```
训练数据：4 亿 (图像, 文本) 对

训练目标：
  ┌──────────┐    ┌──────────┐
  │ Image    │    │  Text    │
  │ Encoder  │    │ Encoder  │
  │ (ViT)    │    │ (Trans.) │
  └────┬─────┘    └────┬─────┘
       │               │
       ▼               ▼
    img_embed       txt_embed
       │               │
       └───── 对比学习 ──┘

  匹配的 (image, text) 对 → 嵌入相似度高
  不匹配的 (image, text) 对 → 嵌入相似度低
```

### 2.2 对比损失 (InfoNCE)

```python
import torch
import torch.nn.functional as F

def clip_loss(image_embeds, text_embeds, temperature=0.07):
    # 归一化
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # 计算相似度矩阵 (B × B)
    logits = image_embeds @ text_embeds.T / temperature

    # 对角线是正样本，其余是负样本
    labels = torch.arange(len(logits), device=logits.device)

    # 双向对比损失
    loss_i2t = F.cross_entropy(logits, labels)      # 图→文
    loss_t2i = F.cross_entropy(logits.T, labels)     # 文→图
    return (loss_i2t + loss_t2i) / 2
```

### 2.3 CLIP 的用途

| 用途 | 说明 |
|---|---|
| **零样本分类** | 把类别名变成文本 → 计算图像与每个文本的相似度 → 选最高的 |
| **图文检索** | 给文本找最匹配的图像，或反过来 |
| **扩散模型条件** | SD 用 CLIP text encoder 编码提示词 |
| **图像评估** | CLIP Score 衡量生成图像与文本的匹配度 |
| **多模态基础** | 几乎所有 VLM 都用 CLIP 或其变体作为视觉编码器 |

```python
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 零样本分类
inputs = processor(
    text=["a photo of a cat", "a photo of a dog", "a photo of a bird"],
    images=image,
    return_tensors="pt",
    padding=True,
)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=-1)
print(probs)  # 例如 [0.85, 0.10, 0.05] → 猫
```

### 2.4 CLIP 架构细节

| 组件 | CLIP ViT-B/32 | CLIP ViT-L/14 |
|---|---|---|
| Image Encoder | ViT-B/32 (86M) | ViT-L/14 (304M) |
| Text Encoder | 12-layer Transformer (63M) | 12-layer Transformer (123M) |
| Embedding Dim | 512 | 768 |
| Image Resolution | 224×224 | 224×224 |
| Patch Size | 32×32 | 14×14 |

---

## 3. SigLIP：CLIP 的升级版

SigLIP (Zhai et al., 2023) 是 Google 提出的 CLIP 改进版：

| 改进 | CLIP | SigLIP |
|---|---|---|
| 损失函数 | softmax (InfoNCE) | **sigmoid** (二分类) |
| batch size 依赖 | 强依赖大 batch（负样本越多越好） | 对 batch size 不敏感 |
| 效果 | 基准 | 同等规模下更好 |

```
CLIP loss: 一行/一列做 softmax → 需要大 batch 提供足够的负样本
SigLIP loss: 每对独立做 sigmoid → 不需要同 batch 内的负样本对比
```

**SigLIP 已成为 2024 年多模态模型的默认视觉编码器**（取代 CLIP ViT-L）。

---

## 4. BLIP / BLIP-2

### 4.1 BLIP (2022)

BLIP (Bootstrapping Language-Image Pre-training) 创新之处：
- 同时做图文对比、图像到文本生成、图文匹配三个任务
- **CapFilt**：用模型自身清洗和生成训练数据（bootstrap）

### 4.2 BLIP-2 (2023)

BLIP-2 (Li et al., 2023) 的核心创新：**Q-Former bridge**。

```
问题：
  视觉编码器（ViT）和 LLM 的表示空间不同
  直接拼接效果差

BLIP-2 的解法：用 Q-Former 做"翻译"

  ┌──────────┐
  │ Image    │  → 图像特征 (257 tokens)
  │ Encoder  │
  │ (冻结 ViT)│
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │ Q-Former │  → 32 个 query token
  │ (可训练)  │     ← 学习从视觉特征中提取关键信息
  └────┬─────┘
       │  32 个 visual tokens
       ▼
  ┌──────────┐
  │   LLM    │  → 文本输出
  │ (冻结)    │
  └──────────┘
```

Q-Former 的作用：
- 把数百个视觉 token 压缩为固定数量（32 个）的 query token
- 弥合视觉和语言之间的 modality gap
- 只需训练 Q-Former（轻量级），冻结 ViT 和 LLM

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
)

inputs = processor(images=image, text="Question: What is shown in this image? Answer:", return_tensors="pt")
output = model.generate(**inputs)
print(processor.decode(output[0], skip_special_tokens=True))
```

---

## 5. Vision-Language Model 架构范式

2024 年 VLM 主要有两种架构范式：

### 5.1 范式 A：Visual Tokens（主流）

**将图像编码为 token，直接注入 LLM 的 token 序列中。**

```
图像 → ViT → 图像特征 → 映射层 (MLP) → visual tokens
                                            ↓
                              [BOS] visual_1 visual_2 ... visual_N 文本 tokens [EOS]
                                            ↓
                                          LLM
                                            ↓
                                        文本输出

代表模型：LLaVA, Qwen-VL, InternVL
```

优点：
- 架构简单，复用现有 LLM 基础设施
- 图像和文本在同一个序列中，统一处理

缺点：
- 高分辨率图像产生大量 visual tokens（例如 1024×1024 → 4096 tokens）
- 占用 LLM 的上下文窗口

### 5.2 范式 B：Cross-Attention

**LLM 通过交叉注意力层访问视觉特征，视觉 token 不占主序列位置。**

```
图像 → ViT → 图像特征 (存储在侧面)
                  ↑
                  │ cross-attention
                  │
文本 tokens → LLM (每隔几层插入 cross-attention) → 文本输出

代表模型：Flamingo, Qwen-VL (部分层)
```

优点：
- 视觉特征不占用文本上下文
- 多图场景更灵活

缺点：
- 需要修改 LLM 架构（插入 cross-attention 层）
- 训练和推理更复杂

### 5.3 范式 C：双路径 / 解耦视觉编码

新一代统一理解与生成模型开始把“理解用视觉编码”和“生成用视觉编码”分开。

```
理解路径: image → semantic encoder → LLM → text
生成路径: image/text → generative tokenizer/latent → decoder/diffusion → image
```

这样做的原因很现实：理解任务需要语义压缩和 OCR/grounding，生成任务需要保留像素细节和可逆表示。
如果强行用一套 visual token 同时服务理解和生成，要么理解 token 太多，要么生成细节不足。

代表方向包括 Chameleon、Emu3、Janus 等 unified / any-to-any 系统。

---

## 6. LLaVA 系列

### 6.1 LLaVA (2023)

LLaVA (Visual Instruction Tuning, Liu et al., 2023) 是最早也最有影响力的开源 VLM：

```
架构（极简）：
  ViT-L/14 (CLIP) → Linear Projection → [visual tokens] + [text tokens] → Vicuna (LLM)

训练两阶段：
  Stage 1: Feature Alignment（冻结 ViT 和 LLM，只训映射层）
    - 用 595K 图文对数据
    - 让映射层学会把视觉特征"翻译"到 LLM 的语言空间

  Stage 2: Visual Instruction Tuning（冻结 ViT，训映射层 + LLM）
    - 用 158K 视觉指令数据（GPT-4 生成的对话）
    - 让模型学会多轮视觉对话
```

### 6.2 LLaVA-1.5 / LLaVA-NeXT

| 改进 | LLaVA | LLaVA-1.5 | LLaVA-NeXT |
|---|---|---|---|
| 映射层 | Linear | 2-layer MLP | 2-layer MLP |
| 分辨率 | 224×224 | 336×336 | 动态分辨率（最高 672×672） |
| LLM | Vicuna-7B | Vicuna-7/13B | Qwen-1.5 / LLaMA 3 等 |
| 视觉编码器 | CLIP ViT-L | CLIP ViT-L | CLIP ViT-L / SigLIP |

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "这张图片里有什么？请详细描述。"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(output[0], skip_special_tokens=True))
```

---

## 7. Qwen-VL / Qwen2-VL

### 7.1 Qwen-VL (2023)

Qwen-VL 是阿里巴巴的视觉语言模型：
- 基于 Qwen LLM
- 支持图像、文本、边界框（bbox）输入
- 支持多图输入

### 7.2 Qwen2-VL (2024)

Qwen2-VL 带来了多个重要创新：

| 特性 | 说明 |
|---|---|
| **动态分辨率 (Dynamic Resolution)** | 支持任意分辨率输入，不强制缩放 |
| **Naive ViT** | 去掉 ViT 的 position embedding，改用 2D-RoPE |
| **视频理解** | 原生支持视频帧输入 |
| **多语言** | 中英文等多语言理解 |

```
动态分辨率原理：
  固定分辨率: 所有图像 resize 到 224×224 → 信息损失
  动态分辨率: 保留原始宽高比，拆成多个 tile

  例如 800×600 的图像：
  1. 计算最优 tile 数量（例如 2×2 = 4 个 tile）
  2. 每个 tile 是 448×448
  3. 每个 tile 过 ViT 得到 token
  4. 所有 tile 的 token 拼接 → visual tokens

  高分辨率图像: 更多 tile → 更多 visual tokens → 更多细节
  低分辨率图像: 更少 tile → 更少 visual tokens → 节省计算
```

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://example.com/image.jpg"},
            {"type": "text", "text": "描述这张图片"},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=256)
```

---

## 8. InternVL

InternVL (Chen et al., 2024) 的核心策略：**把视觉编码器做大**。

```
大部分 VLM 的视觉编码器：
  CLIP ViT-L/14 = 304M 参数
  SigLIP ViT-SO400M = 400M 参数

InternVL 的视觉编码器：
  InternViT-6B = 6B 参数（大了 15-20 倍！）
```

| 模型 | 视觉编码器 | 参数 | LLM |
|---|---|---|---|
| InternVL 1.0 | InternViT-6B | 6B | InternLM (8B) |
| InternVL 1.5 | InternViT-6B | 6B | InternLM2 (20B) |
| InternVL 2 | InternViT-300M–6B | 可选 | InternLM2 / LLaMA 3 |
| InternVL 2.5 | InternViT-300M | 300M | InternLM2.5 |

InternVL 的动态分辨率策略与 Qwen2-VL 类似：
- 将图像切成 448×448 的 tile
- 每个 tile 过 ViT
- 支持最多 12 个 tile（最高 ~2K 分辨率）

---

## 9. GPT-4V / GPT-4o 的多模态能力

这些闭源模型的具体架构未公开，因此只能基于公开接口和使用观察总结能力：

| 能力 | GPT-4V (2023.09) | GPT-4o (2024.05) |
|---|---|---|
| 图像理解 | ✅ 强 | ✅ 更强 |
| OCR / 文档理解 | ✅ 优秀 | ✅ 优秀 |
| 图表分析 | ✅ | ✅ |
| 视频 | ❌ | ✅（多帧） |
| 音频 | ❌ | ✅（端到端语音） |
| 实时性 | ❌ | ✅（低延迟） |
| 架构信息 | 未公开 | 未公开 |

GPT-4o 的"o"代表"omni"（全能）。从公开能力看，它把图像、文本、音频的低延迟交互体验做得更统一。

> 注意：可以说 GPT-4o 具备更统一的多模态交互体验；不能在没有公开证据时断言它内部一定是“单一 Transformer”或某个具体 token 化方案。

---

## 10. 视觉编码器详解

### 10.1 ViT (Vision Transformer)

```
图像 → 切成 patches → 线性映射为 token → 加位置编码 → Transformer

例如 224×224 图像，patch_size=14：
  → 16×16 = 256 个 patch
  → 每个 patch 映射为 d_model 维向量
  → [CLS] + 256 个 patch token → Transformer
```

```python
# ViT patch embedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=14, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x):
        # x: (B, 3, H, W) → (B, num_patches, embed_dim)
        x = self.proj(x)                # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)
        return x
```

### 10.2 分辨率处理策略对比

| 策略 | 做法 | 代表 | 问题 |
|---|---|---|---|
| 固定分辨率 | 强制 resize 到 224×224 | CLIP 原版 | 丢失细节 |
| 更高固定分辨率 | resize 到 336 或 448 | LLaVA-1.5 | token 数 4x |
| 动态分辨率 (tile) | 切成多个 tile 分别编码 | Qwen2-VL, InternVL | 灵活但实现复杂 |
| 插值位置编码 | 训练时 224，推理时 384（插值 PE） | 通用技巧 | 简单有效 |

### 10.3 Visual Token 预算

VLM 工程里最容易被低估的是 visual token 数量。它直接吃掉 LLM 上下文、KV cache 和推理时间。

粗略估算：

```text
num_visual_tokens ≈ tile_count × (tile_size / patch_size)^2 / merge_ratio
```

示例：

| 输入 | 可能的 token 量级 | 风险 |
|---|---|---|
| 单张 224×224，patch 14 | 256 tokens | 低成本，细节有限 |
| 单张 448×448，patch 14 | 1024 tokens | OCR 变强，KV cache 增加 |
| 4 个 448 tile | 4096 tokens | 多图/文档可用，但上下文被挤压 |
| 32 帧视频，每帧 256 tokens | 8192 tokens | 需要帧采样或 token pooling |
| 高清长文档多页 | 1万+ tokens | 必须做页面选择、OCR 或分层检索 |

所以文档理解、UI 截图理解、视频理解通常不是“把图直接丢给 VLM”这么简单。
工程上常用：

- 先 OCR / layout parsing，再把文本和局部截图送入 VLM；
- 先用小模型做页面/区域筛选；
- 对视频做关键帧采样；
- 对 visual tokens 做 pooling、merger 或 query token 压缩；
- 把多轮对话中的旧图像摘要化，避免 KV cache 不断膨胀。

### 10.4 主流视觉编码器

| 编码器 | 来源 | 参数 | 训练数据 | 常见用途 |
|---|---|---|---|---|
| CLIP ViT-L/14 | OpenAI | 304M | 4 亿图文对 | SD 1.5, LLaVA |
| OpenCLIP ViT-bigG | LAION | 1.8B | 20 亿+ | SDXL |
| SigLIP ViT-SO400M | Google | 400M | WebLI | PaliGemma, 多 VLM |
| InternViT-6B | 上海 AI Lab | 6B | 自建数据 | InternVL |
| EVA-02-CLIP | BAAI | 304M–1B | 合并数据 | 多种 VLM |

---

## 11. 训练数据

### 11.1 预训练数据

| 数据集 | 规模 | 说明 |
|---|---|---|
| LAION-5B | 58.5 亿图文对 | 最大的开源图文数据集 |
| DataComp-1B | 12.8 亿 | 精选高质量子集 |
| CC-3M / CC-12M | 3M / 12M | Conceptual Captions |
| SBU Captions | 1M | Flickr 图文对 |
| WebLI | 未公开 | Google 内部数据 |

### 11.2 指令微调数据

| 数据 | 说明 |
|---|---|
| LLaVA-Instruct-158K | GPT-4 生成的视觉指令对话 |
| ShareGPT4V | ShareGPT 风格的视觉对话 |
| ALLaVA | 高质量视觉问答 |
| Cauldron | 50+ 视觉任务的混合数据集 |

训练数据的质量比数量更重要——这是 VLM 领域的共识。

---

## 12. 评估基准

| 基准 | 全称 | 衡量能力 |
|---|---|---|
| **VQAv2** | Visual Question Answering v2 | 基础视觉问答 |
| **GQA** | Graph-based QA | 组合推理 |
| **TextVQA** | Text in VQA | OCR + 理解 |
| **DocVQA** | Document VQA | 文档理解 |
| **ChartQA** | Chart QA | 图表理解 |
| **MMMU** | Massive Multi-discipline Multimodal Understanding | 大学级多学科 |
| **MMBench** | Multimodal Benchmark | 综合视觉理解 |
| **SEED-Bench** | SEED Image/Video Benchmark | 图像+视频理解 |
| **RealWorldQA** | Real World QA | 真实场景理解 |

```python
# 用 lmms-eval 评估 VLM
# pip install lmms-eval
# lmms-eval --model llava --tasks vqav2,textvqa --batch_size 1
```

---

## 13. 新兴方向：Any-to-Any 模型

2024–2025 年的趋势：**统一理解和生成**——一个模型既能"看"又能"画"。

```
传统（分离式）：
  理解: 图像 → LLM → 文本
  生成: 文本 → Diffusion → 图像
  两个独立模型

统一式（Any-to-Any）：
  一个模型同时支持：
  - 图像理解（图 → 文）
  - 图像生成（文 → 图）
  - 图像编辑（图 + 文 → 图）
  - 视频理解（视频 → 文）
  - ...

代表模型：
  - Gemini (Google)：原生多模态
  - GPT-4o (OpenAI)：原生多模态
  - Qwen2.5-Omni (Alibaba)：文本、图像、音频、视频输入，文本和语音输出
  - Chameleon (Meta)：early fusion, 统一 tokenizer
  - Emu3 (BAAI)：next-token prediction 统一生成和理解
  - Janus (DeepSeek)：解耦的视觉编码路径
```

### 统一 tokenizer 方案

```
思路：把图像也变成 discrete token（像文字一样）

图像 → VQ-VAE / VQGAN → 离散 token 序列
文本 → Tokenizer → 离散 token 序列

两种 token 混合在一起 → 统一的 next-token prediction

例如：
  输入: [文本 token] "一只猫" [图像 token] <img_01> <img_02> ... <img_256>
  输出: [文本 token] "图中是一只橘猫坐在窗台上"
  或者:
  输入: [文本 token] "画一只猫"
  输出: [图像 token] <img_01> <img_02> ... → 解码为图像
```

---

## 14. VLM 架构选型指南

| 需求 | 推荐模型 | 理由 |
|---|---|---|
| 快速实验 / 开源 | LLaVA-NeXT | 架构简单，社区活跃 |
| 中文理解 | Qwen2.5-VL / Qwen2-VL | 中文能力强，动态分辨率 |
| 高精度文档理解 | Qwen2.5-VL / InternVL 2.5+ | 动态分辨率、OCR、文档和图表能力强 |
| 商用 API | GPT-4o / Gemini / Claude 等 | 综合能力强，但闭源且成本和合规要单独评估 |
| 资源受限 | PaliGemma / Phi-3-Vision | 小模型也能用 |
| 科研 | LLaVA 系列 | 代码清晰，易于修改 |
| 实时语音视觉交互 | Qwen2.5-Omni / GPT-4o 类 API | 关注端到端延迟和流式输出 |

### 14.1 选型时必须补充的约束

| 约束 | 为什么重要 |
|---|---|
| 图片分辨率和页数 | 决定 visual token 数和 OCR 能力 |
| 是否需要坐标输出 | Grounding、UI 自动化、文档抽取都需要 bbox 能力 |
| 是否需要视频 | 视频不是“多张图”，还涉及帧采样和时序推理 |
| 是否能联网/API | 闭源 API 强但有隐私、成本、稳定性限制 |
| 是否能微调 | 私有领域任务通常需要 LoRA/SFT 或数据蒸馏 |
| 许可证 | 开源模型不等于可商用 |

---

## 15. 常见坑

### 15.1 以为 CLIP 能理解复杂语义

CLIP 的文本编码器只有 12 层 Transformer，上下文窗口 77 tokens。
它擅长简短描述的匹配，但对复杂逻辑、否定、计数等理解较弱。

```python
# CLIP 的局限
# "a photo of a cat" vs "a photo without a cat" → 相似度可能差不多
# "three dogs" vs "two dogs" → 几乎分不开
```

### 15.2 忽视分辨率对 VLM 性能的影响

把 1000×1000 的图 resize 到 224×224 再送进 VLM，小字、细节全丢了。
**OCR 和文档理解任务必须用高分辨率**（或动态分辨率方案）。

### 15.3 Visual token 数量爆炸

高分辨率图像 + 多图场景，visual token 数量可能上千，
严重挤压 LLM 的文本上下文空间。需要 token 压缩或选择策略。

### 15.4 训练数据质量 > 数量

用低质量图文对预训练，模型会学到噪声。
BLIP-2 的 CapFilt 启示：**用模型自身过滤和重新标注数据**效果更好。

### 15.5 冻结 vs 解冻视觉编码器

| 策略 | 优点 | 缺点 |
|---|---|---|
| 冻结 ViT | 训练快，不会"遗忘"预训练知识 | 无法适应新域 |
| 解冻 ViT | 视觉特征更适配任务 | 训练慢，可能过拟合 |
| 先冻后解 | 折中方案 | 需要调参（何时解冻） |

大部分 VLM 采用 **Stage 1 冻结 ViT → Stage 2 解冻部分或全部** 的策略。

### 15.6 多模态幻觉 (Hallucination)

VLM 可能"看到"图中不存在的东西——这是多模态模型最严重的问题之一。
原因：LLM 的语言先验太强，会"编造"看似合理但图中没有的内容。

缓解方法：

- 要求模型引用图中证据，例如区域、文字、坐标或局部描述；
- 对 OCR/数字/表格任务使用专门解析器交叉验证；
- 用检测模型或 grounding 模型验证关键实体；
- 对高风险任务保留人工审核；
- prompt 中明确允许回答“不确定/看不清”。

---

## 16. 实践任务：估算一次 VLM 调用成本

任选一个 VLM，做一张 token 预算表：

```text
模型：
输入图片尺寸：
tile 策略：
patch size：
visual tokens：
文本 tokens：
max_new_tokens：
是否多轮保留图像 KV cache：
预估延迟：
最可能的失败模式：
```

然后用三类输入测试：

| 输入 | 观察点 |
|---|---|
| 普通照片 | 物体识别、空间关系、幻觉 |
| 带小字截图/文档 | OCR、布局、数字准确率 |
| 多图或视频帧 | 跨图一致性、时序理解、token 成本 |

---

## 小结

| 概念 | 一句话解释 |
|---|---|
| CLIP | 用对比学习对齐图像和文本的嵌入空间 |
| SigLIP | sigmoid 损失版 CLIP，对 batch size 不敏感 |
| Q-Former | BLIP-2 的桥梁模块，压缩视觉 token 送入 LLM |
| Visual Tokens | 把图像编码为 token，直接拼入 LLM 序列 |
| LLaVA | ViT + MLP 映射 + LLM，最简洁的 VLM 架构 |
| 动态分辨率 | 将图像切成多个 tile 分别编码，保留细节 |
| Any-to-Any | 一个模型统一理解和生成所有模态 |

下一节学习视频生成——高成本、高门槛的前沿生成任务。
