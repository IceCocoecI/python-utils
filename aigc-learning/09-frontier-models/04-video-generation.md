# 04 · 视频生成架构全解

> 目标：理解视频生成的核心挑战和主流架构——从 AnimateDiff 到 Sora 系到 Wan。
> 视频生成是 2024–2025 年 AIGC 最火热的前沿方向，但也是技术门槛最高的领域之一。

---

## 1. 视频生成：AIGC 的终极战场

### 1.1 为什么视频生成比图像生成难得多？

```
图像生成：生成一张 1024×1024 的图
  → 处理 1M 像素
  → latent 空间: 128×128×4 = 65K tokens

视频生成：生成 5 秒 24fps 的 720p 视频
  → 120 帧 × 1280×720 = 110M 像素
  → latent 空间: 120 × 160×90×4 ≈ 8.6M tokens
  → 计算量是图像的 100x+
```

| 挑战 | 说明 |
|---|---|
| **时序一致性** | 相邻帧之间必须连贯，人物/场景不能突变 |
| **运动建模** | 物体的运动要符合物理规律 |
| **长时生成** | 5 秒就难，更别说分钟级视频 |
| **计算成本** | 一个视频 = 几十到几百张图的计算量 |
| **训练数据** | 高质量视频-文本配对数据稀缺 |
| **评估困难** | 没有像 FID 那样公认的单一指标 |

### 1.2 视频生成的应用场景

| 场景 | 说明 |
|---|---|
| 短视频/广告 | 自动生成产品宣传视频 |
| 影视特效 | 概念预览、背景生成 |
| 虚拟人 | 数字人驱动、直播 |
| 教育 | 可视化解释抽象概念 |
| 游戏 | 过场动画、场景生成 |
| 世界模型 | 模拟物理世界（Sora 的愿景） |

---

## 2. 从图像到视频：添加时间维度

### 2.1 核心思路

视频 = 图像序列。最直觉的做法：**在图像生成模型的基础上添加时间维度的建模**。

```
图像 latent:  (B, C, H, W)          → 2D 空间
视频 latent:  (B, C, T, H, W)       → 3D 时空

T = 帧数 (temporal dimension)

图像生成器只处理空间维度 (H, W)
视频生成器需要同时处理空间 (H, W) 和时间 (T) 维度
```

### 2.2 两种基本策略

```
策略 A：在 2D 模型基础上加时间模块
  ┌───────────────────────────┐
  │  Spatial Block (2D, 预训练) │  ← 处理每帧的空间信息
  │  ↓                         │
  │  Temporal Block (1D, 新加)  │  ← 建模帧间关系
  │  ↓                         │
  │  Spatial Block              │
  │  ↓                         │
  │  Temporal Block             │
  └───────────────────────────┘
  代表：AnimateDiff, SVD

策略 B：原生 3D 架构
  ┌───────────────────────────┐
  │  3D Attention (时空联合)    │  ← 同时处理空间和时间
  │  ↓                         │
  │  3D Attention               │
  │  ↓                         │
  │  ...                       │
  └───────────────────────────┘
  代表：Sora 系, CogVideoX
```

---

## 3. 视频扩散模型的关键组件

### 3.1 3D VAE

图像生成用 2D VAE；视频生成需要 **3D VAE**——在时间维度也进行压缩。

```
2D VAE (图像):
  512×512×3 → 64×64×4
  空间压缩 8x

3D VAE (视频):
  T×512×512×3 → (T/4)×64×64×4
  空间压缩 8x + 时间压缩 4x

  例如 120 帧 512×512 视频:
  120×512×512×3 → 30×64×64×4
```

```python
# CogVideoX 的 3D VAE 使用示意
from diffusers import AutoencoderKLCogVideoX

vae = AutoencoderKLCogVideoX.from_pretrained(
    "THUDM/CogVideoX-5b",
    subfolder="vae",
    torch_dtype=torch.bfloat16,
)

# 编码：(B, C, T, H, W) → latent
video_tensor = torch.randn(1, 3, 49, 480, 720)   # 49 帧
latent = vae.encode(video_tensor).latent_dist.sample()
print(latent.shape)  # 时间和空间都被压缩

# 解码：latent → 视频
reconstructed = vae.decode(latent).sample
```

### 3.2 时间注意力 (Temporal Attention)

```python
class TemporalAttention(nn.Module):
    """在时间维度上做自注意力，建模帧间关系"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, H, W, D)
        B, T, H, W, D = x.shape

        # 把空间维度折叠，在时间维度上做注意力
        x_flat = x.reshape(B * H * W, T, D)     # 每个空间位置独立
        x_flat = x_flat + self.attn(
            self.norm(x_flat), self.norm(x_flat), self.norm(x_flat)
        )[0]
        return x_flat.reshape(B, T, H, W, D)
```

### 3.3 3D 卷积

```python
# 3D 卷积同时处理时空
temporal_conv = nn.Conv3d(
    in_channels=320,
    out_channels=320,
    kernel_size=(3, 1, 1),     # 只在时间维度卷积
    padding=(1, 0, 0),
)

spatial_conv = nn.Conv3d(
    in_channels=320,
    out_channels=320,
    kernel_size=(1, 3, 3),     # 只在空间维度卷积
    padding=(0, 1, 1),
)
```

---

## 4. Sora 类架构

### 4.1 Sora 的技术要点（OpenAI, 2024.02）

Sora 是 OpenAI 的视频生成模型，虽然没有公开论文，但技术报告揭示了关键设计：

```
Sora 的核心架构推测：

  文本 ─→ Text Encoder ──────────────────┐
                                          │
  噪声 latent (T×H×W×C) ──→ Patchify ──→ DiT (Spacetime Transformer) ──→ Unpatchify ──→ 3D VAE Decode ──→ 视频
                              │
                         Spacetime Patches
                         (时空块作为 token)
```

| 关键特性 | 说明 |
|---|---|
| **Spacetime Patches** | 把视频 latent 切成 3D patches（如 1×2×2），每个 patch 是一个 token |
| **可变分辨率/时长** | 不同分辨率和时长的视频都能处理 |
| **DiT 架构** | 用 Transformer 替代 U-Net |
| **长视频** | 支持 ~60 秒视频生成 |
| **世界模型** | 号称学到了物理规律（虽然还不完美） |

### 4.2 Spacetime Patches

```
传统图像 DiT:
  图像 latent (64×64×4) → 2D patches (32×32 个 2×2 patch) → 1024 tokens

Sora 式视频 DiT:
  视频 latent (T'×H'×W'×C) → 3D patches (t×h×w) → 大量 tokens

  例如 30×64×64×4 的视频 latent，patch_size = (2, 2, 2):
  → 15×32×32 = 15,360 tokens
  → Transformer 的序列长度 = 15,360
  → 这就是为什么视频生成的计算量如此巨大
```

```python
class SpacetimePatchEmbed(nn.Module):
    """将视频 latent 切成 spacetime patches"""
    def __init__(self, in_channels, embed_dim, patch_size=(2, 2, 2)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)                # (B, D, T', H', W')
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, D)
        return x
```

---

## 5. CogVideoX

CogVideoX (Yang et al., 2024) 是智谱 AI 开源的高质量视频生成模型。

### 5.1 架构

```
CogVideoX 架构：
  ┌─────────────────────────────────────────┐
  │  Text Encoder (T5-XXL)                   │
  │  → 文本 token embeddings                 │
  └───────────────┬─────────────────────────┘
                  │
  ┌───────────────┴─────────────────────────┐
  │  Expert Transformer (3D Full Attention)  │
  │                                          │
  │  文本 tokens + 视频 patch tokens         │
  │  → Joint Attention (类似 MMDiT)          │
  │  → 3D RoPE (时空位置编码)                │
  └───────────────┬─────────────────────────┘
                  │
  ┌───────────────┴─────────────────────────┐
  │  3D VAE Decoder                          │
  │  → latent → 视频帧                      │
  └─────────────────────────────────────────┘
```

| 特性 | CogVideoX-2B | CogVideoX-5B |
|---|---|---|
| 参数量 | 2B | 5B |
| 视频长度 | 6 秒 / 49 帧 | 6 秒 / 49 帧 |
| 分辨率 | 720×480 | 720×480 |
| 3D VAE 压缩 | 空间 8x, 时间 4x | 空间 8x, 时间 4x |
| 文本编码器 | T5-XXL | T5-XXL |

```python
from diffusers import CogVideoXPipeline
import torch

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

prompt = "A golden retriever playing in the snow, cinematic lighting"
video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6.0,
).frames[0]

# 保存视频
from diffusers.utils import export_to_video
export_to_video(video, "output.mp4", fps=8)
```

---

## 6. Wan（万象）

Wan (万象, 2025) 是阿里巴巴开源的视频生成模型。

### 6.1 架构概览

| 特性 | 说明 |
|---|---|
| 去噪网络 | DiT (Full 3D Attention) |
| 训练方式 | Flow Matching |
| 文本编码器 | 多语言 (中英文支持) |
| 3D VAE | WanVAE (Causal 3D VAE) |
| 模型规格 | 1.3B / 14B |
| 视频长度 | 最长 ~20 秒 |
| 分辨率 | 最高 1280×720 |

### 6.2 WanVAE

Wan 使用 **Causal 3D VAE**——时间维度上采用因果卷积，
使得模型可以同时用于图像（1 帧）和视频（多帧）编码。

```
Causal 3D Conv:
  普通 3D Conv: 卷积核 [t-1, t, t+1] → 看过去和未来
  Causal 3D Conv: 卷积核 [t-2, t-1, t] → 只看过去和当前

  好处：
  - 图像 = 视频的特例（T=1），统一处理
  - 支持自回归式帧扩展
```

```python
from diffusers import WanPipeline
import torch

pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

video = pipe(
    prompt="一只小猫在花园里追蝴蝶，阳光明媚",
    num_frames=81,
    guidance_scale=5.0,
).frames[0]
```

---

## 7. HunyuanVideo

HunyuanVideo (Tencent, 2024) 是腾讯开源的大规模视频生成模型。

### 7.1 关键特性

| 特性 | 说明 |
|---|---|
| 参数量 | 13B（去噪 DiT） |
| 架构 | Dual-stream DiT → Single-stream DiT |
| 训练目标 | Flow Matching |
| 3D VAE | CausalConv3D，空间 8x + 时间 4x |
| 文本编码器 | MLLM (多模态大模型) + CLIP |
| 视频长度 | 最长 ~5 秒 / 129 帧 |

### 7.2 双流到单流架构

```
HunyuanVideo 的 Dual-stream → Single-stream 设计：

前半部分（Dual-stream）：
  ┌─────────┐    ┌─────────┐
  │ 文本流   │    │ 视频流   │
  │ (独立)   │    │ (独立)   │
  └────┬────┘    └────┬────┘
       │              │
       └──── 交互 ────┘   ← Cross-attention 或 Joint-attention

后半部分（Single-stream）：
  ┌──────────────────┐
  │ 合并流            │
  │ 文本+视频 tokens  │  ← 拼接在一起做 Self-attention
  └──────────────────┘

原因：前期分开处理让各模态先学好自身特征
      后期合并让它们深度交互
```

---

## 8. 开源视频模型全景

| 模型 | 机构 | 参数 | 时长 | 分辨率 | 架构 | 开源 |
|---|---|---|---|---|---|---|
| AnimateDiff | 上海 AI Lab | ~1B | 2 秒 | 512×512 | U-Net + Temporal | ✅ |
| SVD (Stable Video Diffusion) | Stability AI | 1.5B | 2–4 秒 | 576×1024 | U-Net + Temporal | ✅ |
| CogVideoX | 智谱 | 2B / 5B | 6 秒 | 720×480 | 3D DiT | ✅ |
| Wan | 阿里 | 1.3B / 14B | ~20 秒 | 1280×720 | 3D DiT | ✅ |
| HunyuanVideo | 腾讯 | 13B | 5 秒 | 960×544 | DiT | ✅ |
| Mochi 1 | Genmo | ~10B | 5 秒 | 848×480 | DiT | ✅ |
| Sora | OpenAI | 未公开 | 60 秒 | 1080p | DiT | ❌ |
| Kling | 快手 | 未公开 | 5–10 秒 | 1080p | 未公开 | ❌ |
| Runway Gen-3 | Runway | 未公开 | 10 秒 | 1080p | 未公开 | ❌ |

---

## 9. Image-to-Video (I2V)

### 9.1 核心思路

给定第一帧图像，生成后续视频——这比纯文本生视频更可控。

```
文本 → 视频 (T2V): 从纯噪声开始，文本引导去噪
图像 → 视频 (I2V): 第一帧是给定的，生成后续帧的运动

I2V 的条件注入方式：
  方式 1: 第一帧编码后替换 latent 的第一帧（replace first frame）
  方式 2: 第一帧编码后通过 cross-attention 注入
  方式 3: 第一帧 + 文本同时作为条件
```

```python
from diffusers import CogVideoXImageToVideoPipeline
import torch
from PIL import Image

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

image = Image.open("first_frame.jpg")
video = pipe(
    prompt="the cat starts walking toward the camera",
    image=image,
    num_frames=49,
    guidance_scale=6.0,
).frames[0]
```

### 9.2 SVD (Stable Video Diffusion)

SVD (Blattmann et al., 2023) 是 Stability AI 的 Image-to-Video 模型：

| 特性 | SVD | SVD-XT |
|---|---|---|
| 输入 | 单张图像 | 单张图像 |
| 输出帧数 | 14 帧 | 25 帧 |
| 分辨率 | 576×1024 | 576×1024 |
| 基础架构 | SD 2.1 + Temporal Layers | 同左 |
| 条件 | 图像 + FPS + 运动强度 | 同左 |

---

## 10. 视频编辑

### 10.1 核心挑战：时序一致性

```
图像编辑: 只需要改一帧 → 相对简单
视频编辑: 改每一帧，且帧间必须一致 → 非常困难

时序一致性的常见问题：
  - 闪烁 (flickering): 相邻帧的颜色/亮度跳变
  - 形变 (warping): 物体形状帧间不一致
  - 断裂 (discontinuity): 编辑区域和未编辑区域不连贯
```

### 10.2 主流技术

| 技术 | 思路 | 代表 |
|---|---|---|
| **逐帧编辑 + 后处理** | 独立编辑每帧，再用光流对齐 | 早期方法 |
| **共享注意力 (Cross-frame Attention)** | 帧之间共享注意力 KV cache | TokenFlow |
| **视频 ControlNet** | 光流/深度图作为控制信号 | 多种 |
| **训练式编辑** | 专门训练视频编辑模型 | InsV2V |

---

## 11. Text-to-Video Pipeline 详解

一个完整的 T2V pipeline：

```
输入: "A astronaut riding a horse on Mars, cinematic, 4K"
                     │
                     ▼
          ┌─────────────────┐
Step 1:   │  Text Encoding   │  文本 → token embeddings
          │  (T5-XXL / CLIP) │  
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
Step 2:   │  Noise Sampling  │  采样随机噪声 latent
          │  (T'×H'×W'×C)   │  (3D VAE 的 latent 空间)
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
Step 3:   │  Iterative       │  多步去噪
          │  Denoising       │  (DiT / U-Net)
          │  (30-50 steps)   │  条件: 文本 embedding + timestep
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
Step 4:   │  3D VAE Decode   │  latent → 视频帧
          │                  │  
          └────────┬────────┘
                   │
                   ▼
输出: video.mp4 (T 帧 × H × W × 3)
```

---

## 12. 训练数据

### 12.1 视频-文本数据集

| 数据集 | 规模 | 说明 |
|---|---|---|
| WebVid-10M | 10M 视频-文本对 | 早期视频生成标配 |
| InternVid | 234M 视频片段 | 大规模视频-文本 |
| Panda-70M | 70M 视频-文本对 | 高质量自动标注 |
| HD-VG-130M | 130M 视频 | 大规模视频数据 |
| 自建数据 | 数亿级 | 商业模型的核心壁垒 |

### 12.2 视频 Caption 生成

高质量视频文本描述是训练的关键瓶颈：

```
早期: alt-text / 弱标注 → 噪声大，质量差
现在: 用 VLM (如 GPT-4V, Qwen-VL) 自动标注

自动标注 pipeline:
  1. 从视频中等间隔抽帧 (如每 2 秒 1 帧)
  2. 用 VLM 描述每帧内容
  3. 用 LLM 合并多帧描述为视频描述
  4. 过滤低质量描述
```

---

## 13. 评估指标

| 指标 | 全称 | 衡量什么 | 方向 |
|---|---|---|---|
| **FVD** | Fréchet Video Distance | 生成视频分布 vs 真实视频分布 | 越低越好 |
| **CLIPSIM** | CLIP Similarity | 视频帧与文本的匹配度 | 越高越好 |
| **时序一致性** | Temporal Consistency | 相邻帧的一致性（CLIP/LPIPS 距离） | 越高/低越好 |
| **运动质量** | Motion Quality | 运动是否流畅自然 | 主观评估 |
| **美学评分** | Aesthetic Score | 画面的审美质量 | 越高越好 |
| **VBench** | Video Benchmark | 16 维度综合评估 | 综合指标 |

```
VBench 的 16 个评估维度（部分）：
  - 主题一致性 (Subject Consistency)
  - 背景一致性 (Background Consistency)
  - 时序闪烁 (Temporal Flickering)
  - 运动平滑度 (Motion Smoothness)
  - 动态程度 (Dynamic Degree)
  - 美学质量 (Aesthetic Quality)
  - 图像质量 (Imaging Quality)
  - 文本-视频对齐 (Overall Consistency)
```

---

## 14. 计算需求与实际考虑

### 14.1 计算成本

```
推理成本对比（单个生成）：

图像 (SD 1.5, 512×512):
  - 显存: ~4 GB
  - 时间: ~2 秒
  - GPU: 单张 RTX 3090

视频 (CogVideoX-5B, 49帧 720×480):
  - 显存: ~30 GB
  - 时间: ~3 分钟
  - GPU: 单张 A100 80GB

视频 (Wan-14B, 81帧 1280×720):
  - 显存: ~60 GB
  - 时间: ~10 分钟
  - GPU: 多张 A100 80GB

训练成本：
  - CogVideoX-5B: 数百 GPU-days
  - Sora 级别: 推测数万 GPU-days
```

### 14.2 推理优化技巧

| 技巧 | 说明 | 节省 |
|---|---|---|
| **VAE Tiling** | 将 latent 切块解码，避免 OOM | 显存 2–4x |
| **CPU Offload** | 模型部分放 CPU | 显存 ~40% |
| **Sequential Offload** | 逐层移入 GPU | 显存最省，但慢 |
| **torch.compile** | 编译模型加速 | 速度 30–50% |
| **FP8/INT8 量化** | 降低精度 | 显存 + 速度 |
| **步数优化** | 减少去噪步数 (50 → 20) | 速度 2.5x |

```python
# 推理优化示例
pipe.enable_model_cpu_offload()       # CPU offload
pipe.vae.enable_tiling()              # VAE tiling
pipe.vae.enable_slicing()             # VAE slicing

# 减少步数
video = pipe(prompt=prompt, num_inference_steps=20).frames[0]
```

---

## 15. 视频生成技术演进时间线

```
2022.10  Make-A-Video (Meta)       首个高质量 T2V
2023.03  Gen-1 (Runway)            商业视频编辑
2023.07  AnimateDiff               U-Net + 时间模块（开源）
2023.11  SVD (Stability AI)        Image-to-Video（开源）
2023.11  Pika 1.0                  商业 T2V
2024.02  Sora (OpenAI)             60 秒长视频，震惊世界
2024.06  Kling (快手)              商业 T2V，运动质量高
2024.08  CogVideoX (智谱)          开源 3D DiT 视频模型
2024.10  Mochi 1 (Genmo)           开源 DiT 视频模型
2024.10  Movie Gen (Meta)          30 秒高质量视频
2024.12  HunyuanVideo (腾讯)       13B 开源视频模型
2024.12  Veo 2 (Google)            商业 T2V
2025.02  Wan 万象 (阿里)           14B 开源，最长 ~20 秒
2025     Sora 正式发布             商业化
```

---

## 16. 常见坑

### 16.1 以为视频生成只是"逐帧图像生成"

独立生成每帧再拼接 = 闪烁 + 不连贯。**时序一致性需要显式建模**。

### 16.2 显存不够就 OOM

视频模型的显存需求远超图像模型。必须使用 tiling、offload 等技巧。
推荐开发时用低分辨率 + 少帧数（如 256×256、16 帧）快速验证。

### 16.3 忽视 3D VAE 的重要性

3D VAE 的重建质量直接决定视频质量的上限。
如果 VAE 重建就有闪烁或模糊，后面的扩散模型再好也救不回来。

### 16.4 过度依赖 FVD

FVD 是最常用的视频生成指标，但它有严重局限：
- 对帧采样策略敏感
- 不能反映文本一致性
- 参考数据集偏差大
推荐使用 **VBench** 做多维度评估。

### 16.5 训练数据的时序标注不够

视频文本描述如果只描述静态内容（"一只猫在草地上"）而不描述动作（"一只猫正在追逐蝴蝶"），
模型就学不好运动。**动态描述比静态描述重要得多**。

### 16.6 生成时长和质量的 trade-off

目前的模型在长视频上质量会下降：
- 5 秒内：大部分模型能做到高质量
- 10–20 秒：质量开始下降，一致性变差
- 60 秒+：只有少数顶尖模型（如 Sora）宣称支持

---

## 小结

| 概念 | 一句话解释 |
|---|---|
| 3D VAE | 在时间和空间维度同时压缩视频到潜在空间 |
| Temporal Attention | 在时间维度做注意力，建模帧间关系 |
| Spacetime Patches | 把视频 latent 切成 3D 块作为 Transformer 的 token |
| I2V | 给定第一帧，生成后续视频 |
| CogVideoX | 智谱开源的 3D DiT 视频模型 |
| Wan | 阿里开源的大规模视频生成模型 |
| VBench | 16 维度视频生成综合评估基准 |
| Causal 3D VAE | 因果 3D VAE，统一图像和视频编码 |

下一节学习语音与音频生成——AIGC 的另一个重要战场。
