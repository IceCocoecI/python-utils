# 02 · 图像生成架构全解

> 目标：理清图像生成从 GAN 到 Flow Matching 的完整演进脉络。
> 重点掌握扩散模型家族——这是 2023–2025 年图像/视频/音频生成的绝对主流范式。
> 本文按 `2026-06-30` 的公开资料组织，重点关注能迁移到工程选型的架构规律。

---

## 1. 图像生成方法演进

```
2014        2017         2020         2022           2023           2024
 │           │            │            │              │              │
 GAN ──→ VAE 成熟 ──→  DDPM ──→  Stable Diffusion ──→ DiT ──→  Flux / SD3
 │                       │            │              │              │
StyleGAN              DDIM/DPM++    SDXL          ControlNet    Flow Matching
ProGAN                              LDM 范式       IP-Adapter     MMDiT
```

| 方法 | 核心思想 | 优点 | 缺点 |
|---|---|---|---|
| GAN | 生成器 vs 判别器对抗 | 生成速度快 | 训练不稳定，模式坍塌 |
| VAE | 编码到潜在空间，解码重建 | 训练稳定，有显式概率模型 | 生成模糊 |
| Diffusion | 逐步加噪 → 逐步去噪 | 生成质量高，训练稳定 | 采样慢（需多步） |
| Flow Matching | 学习从噪声到数据的直线路径 | 训练简单，采样更快 | 较新，生态还在成熟 |

### 1.1 架构选型先看什么？

图像生成模型不要只按“哪个更新”排序，而要按任务约束排序。

| 场景 | 更优先的架构能力 | 常见选择 |
|---|---|---|
| 快速出图、生态插件 | 成熟 scheduler、LoRA、ControlNet、社区模型 | SD 1.5 / SDXL |
| 高质量通用文生图 | 文本理解、构图、审美、少步采样 | Flux / SD3.5 / 商业 API |
| 文本渲染和复杂 prompt | 强文本编码器、MMDiT/joint attention、数据质量 | SD3.5 / Flux / DALL-E 类 |
| 可控生成 | ControlNet、IP-Adapter、参考图、mask/inpaint | SDXL 生态仍很强 |
| 产品级一致性 | 固定风格、可复现、低失败率、审核和水印 | 自建 SDXL/Flux 工作流 + 评估集 |
| 研究新架构 | DiT/MMDiT、Flow Matching、蒸馏、consistency | SD3/Flux 系技术路线 |

工程判断的核心是：你缺的是质量、控制、速度、成本、版权安全，还是工作流生态。
这几个目标经常冲突。

---

## 2. GAN 概述

### 2.1 核心思想

```
Generator (G): 噪声 z → 假图像
Discriminator (D): 图像 → 真/假概率

min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]

训练过程：
  1. D 学会分辨真假 → D 变强
  2. G 学会骗过 D → G 变强
  3. 二者交替训练，达到纳什均衡
```

### 2.2 GAN 的问题

| 问题 | 说明 |
|---|---|
| **模式坍塌 (Mode Collapse)** | G 只学会生成几种图像，多样性丧失 |
| **训练不稳定** | G 和 D 的平衡很脆弱，容易崩溃 |
| **没有显式似然** | 无法计算 p(x)，评估困难 |
| **难以控制生成** | 条件生成需要额外设计 |

### 2.3 StyleGAN

StyleGAN (Karras et al., 2019–2021) 是 GAN 的巅峰：
- 风格注入机制：通过 Mapping Network 将 z 映射为 style vector w
- 逐层风格控制：不同层控制不同粒度（姿态/表情/纹理）
- 高分辨率人脸生成：1024×1024 照片级真实

**但 2022 年后，扩散模型在质量和可控性上全面超越 GAN，GAN 逐渐退出主流。**

---

## 3. VAE 概述

### 3.1 核心思想

```
Encoder: x → μ, σ (潜在空间的均值和方差)
Reparameterization: z = μ + σ * ε,  ε ~ N(0, 1)
Decoder: z → x̂ (重建图像)

训练目标 (ELBO):
  L = 重建损失 (MSE 或 BCE) + KL 散度 (拉近 q(z|x) 和 p(z))
```

### 3.2 VAE 在扩散模型中的角色

虽然纯 VAE 生成的图像模糊，但 VAE 在扩散模型中扮演关键角色：

**Latent Diffusion 的 VAE**：将图像压缩到低维潜在空间，扩散过程在潜在空间进行。

```python
# SD 中的 VAE 使用
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

# 编码：图像 → 潜在表示
latent = vae.encode(image).latent_dist.sample()
latent = latent * vae.config.scaling_factor   # 缩放因子

# 解码：潜在表示 → 图像
image = vae.decode(latent / vae.config.scaling_factor).sample
```

SD 1.5 的 VAE 将 512×512 图像压缩到 64×64×4 的潜在表示——**分辨率降低 8x，通道数为 4**。

---

## 4. 扩散模型基础

### 4.1 前向过程（加噪）

给原始图像 \(x_0\) 逐步添加高斯噪声，经过 \(T\) 步变成纯噪声 \(x_T \sim \mathcal{N}(0, I)\)：

\[
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
\]

关键性质：可以直接从 \(x_0\) 一步跳到 \(x_t\)（不需要逐步加噪）：

\[
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\]

```python
def forward_diffusion(x_0, t, noise_schedule):
    alpha_bar = noise_schedule.alpha_bar[t]
    noise = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
    return x_t, noise
```

### 4.2 反向过程（去噪）

训练一个神经网络 \(\epsilon_\theta\) 来预测 \(x_t\) 中的噪声：

\[
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
\]

```python
def training_step(model, x_0, noise_scheduler):
    noise = torch.randn_like(x_0)
    t = torch.randint(0, T, (x_0.shape[0],))

    # 一步加噪
    x_t = noise_scheduler.add_noise(x_0, noise, t)

    # 预测噪声
    noise_pred = model(x_t, t)

    loss = F.mse_loss(noise_pred, noise)
    return loss
```

### 4.3 DDPM (Denoising Diffusion Probabilistic Models)

DDPM (Ho et al., 2020) 是现代扩散模型的开山之作：

| 组件 | 设计 |
|---|---|
| Noise Schedule | 线性 β schedule，从 β₁=0.0001 到 β_T=0.02 |
| 网络 | U-Net（含 ResNet blocks + Self-Attention） |
| 预测目标 | 预测噪声 ε |
| 采样步数 | T=1000（非常慢） |

### 4.4 DDIM (Denoising Diffusion Implicit Models)

DDIM (Song et al., 2020) 的突破：**确定性采样 + 大幅减少步数**。

```
DDPM: 需要 1000 步采样（~20 秒/张图）
DDIM: 可以用 50 步甚至 20 步（~1 秒/张图）
原因: DDIM 定义了一个非马尔可夫过程，可以跳步
```

---

## 5. Latent Diffusion Model (LDM)

### 5.1 核心思想

**不在像素空间做扩散，而在 VAE 编码的潜在空间做扩散——计算量降低 4–16 倍。**

```
LDM = VAE Encoder + 去噪网络 (U-Net/DiT) + VAE Decoder

训练流程：
  1. VAE Encoder: 图像 (512×512×3) → latent (64×64×4)
  2. 对 latent 加噪
  3. 去噪网络学习从噪声 latent 恢复干净 latent
  4. VAE Decoder: 干净 latent → 图像

推理流程：
  1. 从纯噪声 latent 开始 (64×64×4)
  2. 去噪网络逐步去噪
  3. VAE Decoder 解码为图像
```

### 5.2 条件注入（Conditioning）

文本条件通过 **Cross-Attention** 注入 U-Net：

```python
# Cross-Attention: 图像特征作为 Q，文本特征作为 K 和 V
class CrossAttention(nn.Module):
    def __init__(self, d_model, d_context, n_heads):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_context, d_model)
        self.v_proj = nn.Linear(d_context, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, x, context):
        # x: 图像 latent 特征 (B, HW, D)
        # context: 文本编码 (B, seq_len, D_text)
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        # Multi-head attention...
        return self.out_proj(attn_output)
```

---

## 6. Stable Diffusion 1.5

SD 1.5 (Rombach et al., 2022) 是 LDM 的工程化产物，架构三件套：

```
┌──────────────────────────────────────────────────┐
│              Stable Diffusion 1.5                 │
│                                                   │
│  "a photo of an astronaut riding a horse"         │
│           │                                       │
│           ▼                                       │
│  ┌─────────────────┐                              │
│  │ CLIP Text Encoder│  text → 77×768 embeddings   │
│  │ (ViT-L/14, 冻结) │                              │
│  └────────┬────────┘                              │
│           │ cross-attention                       │
│           ▼                                       │
│  ┌─────────────────┐                              │
│  │   U-Net (860M)   │  64×64×4 noisy latent       │
│  │  ResNet + Attn   │  → 64×64×4 denoised latent  │
│  │  + CrossAttn     │                              │
│  └────────┬────────┘                              │
│           │                                       │
│           ▼                                       │
│  ┌─────────────────┐                              │
│  │  VAE Decoder     │  64×64×4 → 512×512×3        │
│  └─────────────────┘                              │
└──────────────────────────────────────────────────┘
```

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

image = pipe(
    "a photo of an astronaut riding a horse",
    num_inference_steps=50,
    guidance_scale=7.5,
).images[0]
```

---

## 7. SDXL (Stable Diffusion XL)

SDXL (Podell et al., 2023) 是 SD 1.5 的全面升级：

| 特性 | SD 1.5 | SDXL |
|---|---|---|
| U-Net 参数 | 860M | 2.6B |
| 默认分辨率 | 512×512 | 1024×1024 |
| 文本编码器 | CLIP ViT-L/14 (1个) | OpenCLIP ViT-bigG + CLIP ViT-L (2个) |
| 文本 embedding | 77×768 | 77×2048 (拼接两个编码器) |
| Refiner | 无 | 可选的 refiner 模型 |
| Micro-conditioning | 无 | 分辨率 + 裁剪参数作为条件 |

```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

image = pipe(
    "a majestic lion in a cosmic nebula, digital art",
    num_inference_steps=30,
    guidance_scale=7.0,
    height=1024,
    width=1024,
).images[0]
```

---

## 8. Classifier-Free Guidance (CFG)

### 8.1 核心原理

同时训练有条件和无条件的去噪：

```
训练时：
  - 以概率 p（通常 10%）丢弃文本条件，用空文本 "" 代替
  - 模型同时学会有条件和无条件去噪

推理时：
  noise_pred = noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)

  scale = guidance_scale（CFG 强度）：
    - scale = 1.0: 等于无引导
    - scale = 7.0–7.5: SD 常用值
    - scale > 15: 过饱和、过锐化
```

### 8.2 代码实现

```python
def classifier_free_guidance(model, x_t, t, text_embed, empty_embed, guidance_scale):
    # 有条件预测
    noise_cond = model(x_t, t, encoder_hidden_states=text_embed)
    # 无条件预测（用空文本）
    noise_uncond = model(x_t, t, encoder_hidden_states=empty_embed)
    # 引导
    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    return noise_pred
```

**直觉**：CFG 就是在说"更像文本描述的方向走，远离无条件方向"。
scale 越大，对文本越忠实，但多样性降低、可能出现伪影。

---

## 9. ControlNet

ControlNet (Zhang et al., 2023) 为扩散模型添加空间控制能力：

```
输入控制信号：
  - Canny 边缘图
  - 深度图 (depth map)
  - 人体姿态 (OpenPose)
  - 语义分割图
  - ...

架构：
  - 复制 U-Net 的 encoder 作为 ControlNet
  - ControlNet 处理控制信号
  - 输出通过 zero convolution 加回原始 U-Net

  ┌──────────┐    ┌──────────┐
  │ U-Net    │←───│ControlNet│
  │ (冻结)   │    │ (可训练)  │
  └──────────┘    └──────────┘
       ↑               ↑
   noisy latent    control image
   + text embed    (edge/depth/pose)
```

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16,
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)

image = pipe("a beautiful house", image=canny_image).images[0]
```

---

## 10. DiT (Diffusion Transformer)

### 10.1 核心创新

DiT (Peebles & Xie, 2023) 用 **Transformer 替换 U-Net** 作为扩散模型的去噪网络：

```
U-Net 时代 (SD 1.5, SDXL):
  去噪网络 = U-Net (ResNet + Self-Attn + Cross-Attn)

DiT 时代 (Flux, SD3, Sora):
  去噪网络 = Vision Transformer (ViT)

图像 latent → 拆成 patches → 当作 token 序列 → Transformer 处理
```

| 特性 | U-Net | DiT |
|---|---|---|
| 结构 | 编码器-解码器 + skip connections | 纯 Transformer |
| 缩放性 | 缩放规律不清晰 | 遵循 ViT 的 scaling law |
| 灵活性 | 固定分辨率结构 | 可变分辨率（token 序列长度可变） |
| 条件注入 | Cross-Attention | AdaLN-Zero（更高效） |

### 10.2 AdaLN-Zero 条件注入

DiT 不用 Cross-Attention 注入条件，而是用 **Adaptive Layer Normalization**：

```python
class DiTBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = MLP(d_model)

        # AdaLN: 从条件 (timestep + class label) 预测 scale 和 shift
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        # Modulated LayerNorm
        h = self.norm1(x) * (1 + scale_msa) + shift_msa
        h = self.attn(h, h, h)[0]
        x = x + gate_msa * h

        h = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        x = x + gate_mlp * h
        return x
```

---

## 11. Flux

Flux (Black Forest Labs, 2024) 是 SD 原班人马打造的新一代模型：

### 11.1 核心架构

| 特性 | 说明 |
|---|---|
| 去噪网络 | **MMDiT** (Multimodal DiT) |
| 训练目标 | **Flow Matching**（非 DDPM） |
| 文本编码器 | CLIP ViT-L + **T5-XXL**（更强的文本理解） |
| 图像编码 | Patch-based（2×2 patches on latent） |

### 11.2 MMDiT (Multimodal DiT)

MMDiT 的关键创新：**文本和图像 token 在同一个 Transformer 中联合处理**。

```
传统 (SD/SDXL):
  文本 token → CLIP encoder（单独处理）
  图像 latent → U-Net（通过 cross-attention 引用文本）

MMDiT (Flux/SD3):
  文本 token ─┐
              ├─→ 拼接 → Joint Transformer → 分离
  图像 token ─┘

  文本和图像在注意力层中相互交互（joint attention）
```

```python
class MMDiTBlock(nn.Module):
    """简化的 MMDiT block：文本和图像联合注意力"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm_img = nn.LayerNorm(d_model)
        self.norm_txt = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp_img = MLP(d_model)
        self.mlp_txt = MLP(d_model)

    def forward(self, img_tokens, txt_tokens, c):
        # 拼接图像和文本 token
        n_img = img_tokens.shape[1]
        x = torch.cat([img_tokens, txt_tokens], dim=1)

        # Joint attention
        x_norm = torch.cat([self.norm_img(img_tokens), self.norm_txt(txt_tokens)], dim=1)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]

        # 分离后各自过 MLP
        img_tokens, txt_tokens = x[:, :n_img], x[:, n_img:]
        img_tokens = img_tokens + self.mlp_img(img_tokens)
        txt_tokens = txt_tokens + self.mlp_txt(txt_tokens)
        return img_tokens, txt_tokens
```

### 11.3 Flow Matching

与 DDPM 的对比：

| 特性 | DDPM | Flow Matching |
|---|---|---|
| 前向过程 | 马尔可夫链逐步加噪 | 噪声到数据的**直线插值** |
| 训练目标 | 预测噪声 ε | 预测**速度场** v（从噪声指向数据的方向） |
| 路径 | 曲线路径 | 最优传输直线路径 |
| 采样 | 需要特殊 scheduler | ODE 求解器（Euler 等） |

> 这里的“直线”是最简化的教学直觉。真实模型可能使用 rectified flow、不同噪声参数化、时间采样策略、蒸馏和专门的 solver。
> 工程上不要只看模型是否叫 Flow Matching，还要看官方推荐的 scheduler、步数、CFG 范围和分辨率 bucket。

```python
# Flow Matching 训练（简化）
def flow_matching_loss(model, x_0):
    noise = torch.randn_like(x_0)
    t = torch.rand(x_0.shape[0], 1, 1, 1, device=x_0.device)

    # 直线插值: x_t = (1-t) * noise + t * x_0
    x_t = (1 - t) * noise + t * x_0

    # 目标速度: v = x_0 - noise（从噪声指向数据）
    target_v = x_0 - noise

    # 模型预测速度
    pred_v = model(x_t, t)

    return F.mse_loss(pred_v, target_v)
```

---

## 12. SD3 / SD3.5

Stable Diffusion 3 (Esser et al., 2024) 集合了最新技术：

| 特性 | SDXL | SD3 |
|---|---|---|
| 去噪网络 | U-Net (2.6B) | MMDiT (2B / 8B) |
| 训练目标 | ε-prediction | Flow Matching |
| 文本编码器 | CLIP×2 | CLIP ViT-L + OpenCLIP ViT-bigG + **T5-XXL** |
| 文本理解 | 中等 | 显著提升（得益于 T5） |
| 分辨率 | 1024×1024 | 可变分辨率 |

```python
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to("cuda")

image = pipe(
    "a cat holding a sign that says 'hello world'",
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
```

---

## 13. 训练目标对比

| 预测目标 | 公式 | 代表模型 | 说明 |
|---|---|---|---|
| ε-prediction | 预测添加的噪声 ε | DDPM, SD 1.5, SDXL | 最经典，高噪声时效果好 |
| v-prediction | 预测 v = √ᾱ ε - √(1-ᾱ) x₀ | SD 2.x | 数值更稳定 |
| x₀-prediction | 直接预测干净图像 | 部分模型 | 低噪声时效果好 |
| Flow Matching | 预测速度场 v = x₀ - ε | Flux, SD3 | 训练简单，路径更直 |

### 13.1 预测目标和 scheduler 必须匹配

扩散/flow 模型最容易踩的坑是“模型、scheduler、prediction type 不匹配”。

| 错误组合 | 现象 | 原因 |
|---|---|---|
| ε-prediction 模型配 flow scheduler | 图像发灰、结构崩坏 | 模型输出语义不是速度场 |
| Flow 模型配 DDPM/DDIM scheduler | 采样方向错误 | 时间参数化和更新公式不一致 |
| v-prediction 当 ε 用 | 细节差、颜色异常 | 预测目标坐标系不同 |
| CFG scale 沿用旧模型经验 | 过饱和或 prompt 不跟随 | 不同训练目标的 guidance 响应不同 |

调试生成质量时，先确认 pipeline 配置，再调 prompt。很多“模型不行”其实是 scheduler 或 VAE 配错。

---

## 14. 采样算法

采样算法决定了生成质量和速度的平衡：

| 算法 | 步数 | 特点 | 适用场景 |
|---|---|---|---|
| DDPM | 1000 | 原始方法，质量好但极慢 | 学术研究 |
| DDIM | 20–50 | 确定性采样，可减少步数 | 通用 |
| DPM++ 2M | 20–30 | 多步 ODE 求解器，质量稳定 | SD 系列推荐 |
| Euler | 20–30 | 最简单的 ODE 求解器 | Flow Matching 模型 |
| Euler Ancestral | 20–30 | 加入随机性，更多样 | 需要随机性时 |
| DPM++ 2M Karras | 20–30 | Karras noise schedule | SDXL 推荐 |
| UniPC | 10–20 | 统一预测-校正器 | 快速生成 |

```python
from diffusers import DPMSolverMultistepScheduler

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="dpmsolver++",
    use_karras_sigmas=True,
)
```

---

## 15. 分辨率与宽高比处理

### 15.1 问题

扩散模型通常在固定分辨率上训练。如果推理时用不同分辨率/宽高比：
- 可能出现变形、重复结构
- 质量下降

### 15.2 解决方案

| 方法 | 说明 | 代表 |
|---|---|---|
| 多分辨率训练 | 训练时使用多种分辨率 bucket | SDXL |
| Micro-conditioning | 将原始分辨率和裁剪参数作为条件输入 | SDXL |
| Patch-based | 图像切成 patch 后长度可变 | DiT, Flux |
| NaViT | 不同分辨率图像在同一 batch 中混合 | Google |

---

## 16. 图像生成评估指标

| 指标 | 全称 | 衡量什么 | 分数方向 |
|---|---|---|---|
| **FID** | Fréchet Inception Distance | 生成图像分布 vs 真实图像分布 | 越低越好 |
| **CLIP Score** | CLIP 相似度 | 图像与文本的匹配程度 | 越高越好 |
| **Aesthetic Score** | 美学评分 | 图像的审美质量 | 越高越好 |
| **IS** | Inception Score | 生成图像的质量和多样性 | 越高越好 |
| **LPIPS** | Learned Perceptual Similarity | 感知相似度（用于图像重建） | 越低越好 |

### 16.1 指标和产品体验的落差

公开指标只能回答一部分问题：

| 你关心的问题 | 单一指标是否足够 | 更可靠的做法 |
|---|---|---|
| 画面是否美观 | 不足 | 人工偏好评测 + aesthetic model + 业务 prompt 集 |
| 是否严格跟随 prompt | 不足 | 分解式 VQA/CLIPScore + 人工审核 |
| 文字是否正确 | 不足 | OCR 检测 + 字符级准确率 |
| 人脸/手部是否稳定 | 不足 | 专门的失败类型标注 |
| 品牌风格是否一致 | 不足 | 固定 seed/LoRA/参考图 + 内部 golden set |
| 是否可上线 | 不足 | 安全审核、版权过滤、敏感内容评估 |

产品落地时建议维护一套 50–200 条内部 prompt，覆盖真实失败模式，而不是只看论文里的 FID 或榜单。

```python
# 计算 CLIP Score
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

inputs = processor(text=["a cat"], images=[generated_image], return_tensors="pt")
outputs = model(**inputs)
clip_score = outputs.logits_per_image.item()
```

---

## 17. 领域演进时间线

```
2014  GAN                 生成对抗网络
2015  DCGAN               卷积 GAN
2017  VAE 成熟             变分自编码器
2019  StyleGAN             高质量人脸生成
2020  DDPM                 扩散模型复兴
2020  DDIM                 快速采样
2021  Guided Diffusion     classifier guidance
2021  GLIDE                文本引导扩散
2022  DALL-E 2             CLIP + 扩散
2022  LDM / SD 1.x         潜在扩散 → 开源爆发
2022  DPM++ / Karras       高效采样器
2023  SDXL                 更大 U-Net + 双编码器
2023  ControlNet           空间可控生成
2023  DiT                  Transformer 替代 U-Net
2023  IP-Adapter           图像条件注入
2024  SD3 / SD3.5          MMDiT + Flow Matching
2024  Flux                 MMDiT + T5 + Flow Matching
2024  DALL-E 3             更好的文本理解
2025  Flux Kontext         图像生成与编辑统一
2025+ 更快蒸馏、更强编辑、更高分辨率、图像视频统一
```

---

## 18. 常见坑

### 18.1 CFG scale 设太高

guidance_scale 不是越高越好。过高会导致过饱和、色彩失真、出现伪影。
推荐范围：SD 1.5/SDXL 用 5–8，Flux/SD3 用 2–5。

### 18.2 混淆像素空间和潜在空间

SD 的 U-Net/DiT 在**潜在空间**工作，分辨率是图像的 1/8。
512×512 图像 → 64×64 latent。不要把像素尺寸和 latent 尺寸搞混。

### 18.3 忽略 VAE 的缩放因子

SD 的 VAE 有一个 `scaling_factor`（通常 0.18215），编码后要乘、解码前要除。
忘记这一步会导致生成质量严重下降。

### 18.4 采样步数不是越多越好

20–30 步通常就够了。步数太多可能反而引入累积误差，浪费计算。

### 18.5 以为所有扩散模型都用 ε-prediction

SD3 和 Flux 用的是 Flow Matching（预测速度场），不是 ε-prediction。
混用 scheduler 会导致完全错误的结果。

### 18.6 分辨率不对

用 SD 1.5 生成 1024×1024 会出问题（训练分辨率是 512×512）。
同理，用 SDXL 生成 512×512 也不理想。**用模型训练时的分辨率**。

### 18.7 忽略 VAE 和文本编码器版本

同一个 U-Net/DiT，换 VAE 或 text encoder 都可能改变画质、文字能力和风格稳定性。
复现实验时要记录：

- base model / refiner；
- VAE 权重；
- text encoder；
- scheduler；
- prediction type；
- 分辨率、步数、CFG、seed。

只记录 prompt 不足以复现图像生成结果。

---

## 19. 实践任务：拆一个图像模型 pipeline

任选一个 SDXL、SD3.5 或 Flux pipeline，写一页拆解：

```text
模型：
输入 prompt：
文本编码器：
VAE 压缩率：
去噪主干：
训练目标：
scheduler：
默认步数：
默认分辨率：
CFG 推荐范围：
最可能的失败模式：
```

再做 5 组对比实验：

| 实验 | 观察点 |
|---|---|
| 步数 10 / 20 / 30 / 50 | 质量是否继续提升，速度差多少 |
| CFG 2 / 4 / 7 / 12 | prompt 跟随、饱和、伪影 |
| 512 / 768 / 1024 / 非标准宽高比 | 构图和重复结构 |
| 同 prompt 不同 seed | 多样性和稳定性 |
| 同 seed 不同 scheduler | 细节、颜色、结构差异 |

---

## 小结

| 概念 | 一句话解释 |
|---|---|
| LDM | 在 VAE 编码的潜在空间做扩散，计算量大幅减少 |
| U-Net | SD 1.5/SDXL 的去噪骨干，含 ResNet + Attention |
| DiT | 用 Transformer 替代 U-Net，缩放性更好 |
| MMDiT | 文本和图像 token 联合注意力，Flux/SD3 的核心 |
| CFG | 有条件预测和无条件预测的线性组合，控制文本忠实度 |
| Flow Matching | 学习噪声到数据的直线路径，训练更简单 |
| ControlNet | 冻结原模型 + 可训练控制分支，添加空间控制 |
| DDIM / DPM++ | 快速采样算法，20–30 步即可生成高质量图像 |

下一节学习多模态模型——看图像和语言如何连接起来。
