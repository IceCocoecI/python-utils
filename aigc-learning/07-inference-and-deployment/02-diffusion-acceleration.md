# 02 · 扩散模型推理加速

> 扩散模型天生就慢——一张图要跑几十步去噪。
> 但好消息是，加速手段非常丰富：从算法到工程，可以把生成时间从分钟级压到秒级甚至亚秒级。

---

## 1. 为什么扩散模型慢？

### 迭代去噪的本质

```
┌──────────────────────────────────────────────────┐
│  扩散模型生成过程                                  │
│                                                  │
│  纯噪声 → UNet → 稍微清楚一点                     │
│           → UNet → 更清楚一点                     │
│           → UNet → ……                            │
│           → UNet → 清晰图像     （20-50 步）      │
│                                                  │
│  每一步都要跑一次完整的 UNet / DiT 前向传播        │
│  UNet (SD 1.5): ~1.6B 参数                       │
│  DiT (FLUX): ~12B 参数                           │
│  每步 ≈ 几十~几百 ms                              │
└──────────────────────────────────────────────────┘
```

**核心矛盾**：生成质量要求足够多的去噪步骤，但每一步都有显著的计算成本。

### 加速手段全景图

```
┌────────────────────────────────────────────────────────┐
│  扩散模型加速全景                                       │
│                                                        │
│  算法层面              工程层面             模型层面     │
│  ┌──────────┐     ┌──────────────┐    ┌────────────┐  │
│  │ 更快的    │     │ torch.compile│    │ 模型蒸馏   │  │
│  │ Scheduler │     │ TensorRT     │    │ LCM        │  │
│  │ (减少步数) │     │ ONNX Runtime │    │ Turbo      │  │
│  │ DDIM      │     │ FlashAttn    │    │ Lightning  │  │
│  │ DPM++     │     │ xformers     │    │ Schnell    │  │
│  │ LCM       │     │ SDPA         │    │            │  │
│  └──────────┘     └──────────────┘    └────────────┘  │
│                                                        │
│  显存优化                                              │
│  ┌──────────────────────────────────────┐              │
│  │ FP16/BF16 · VAE tiling · CPU offload│              │
│  │ Attention slicing · Sequential offload│              │
│  └──────────────────────────────────────┘              │
└────────────────────────────────────────────────────────┘
```

---

## 2. Scheduler 优化：减少去噪步数

Scheduler（也叫 Sampler）决定了去噪的步进策略。更高效的 Scheduler 能用更少的步数达到相似的质量。

### 2.1 常用 Scheduler 对比

| Scheduler | 典型步数 | 质量 | 速度 | 说明 |
|---|---|---|---|---|
| DDPM | 1000 | 最好 | 极慢 | 原始论文的方法，不实用 |
| DDIM | 50 | 好 | 较快 | 确定性采样，可跳步 |
| Euler | 20-30 | 好 | 快 | 简单高效 |
| DPM++ 2M | 20-30 | 很好 | 快 | 目前最流行的通用选择 |
| DPM++ 2M Karras | 20-30 | 很好 | 快 | 带 Karras 噪声调度，细节更好 |
| DPM++ SDE Karras | 20-30 | 很好 | 稍慢 | 随机性采样，多样性好 |
| LCM | 4-8 | 好 | 极快 | 蒸馏专用，极少步数 |
| Turbo / Lightning | 1-4 | 好 | 极快 | 蒸馏模型专用 |

### 2.2 在 Diffusers 中切换 Scheduler

```python
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

# 切换为 DPM++ 2M Karras
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="dpmsolver++",
    use_karras_sigmas=True,
)

image = pipe(
    "a photo of a cat sitting on a windowsill, golden hour lighting",
    num_inference_steps=25,
    guidance_scale=7.5,
).images[0]
```

### 2.3 LCM Scheduler：4 步出图

```python
from diffusers import LCMScheduler

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# 加载 LCM LoRA
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")

image = pipe(
    "a beautiful sunset over mountains",
    num_inference_steps=4,      # 只需 4 步！
    guidance_scale=1.5,         # LCM 用低 CFG
).images[0]

pipe.unload_lora_weights()
```

---

## 3. `torch.compile`：零成本加速

### 3.1 原理

`torch.compile` 通过以下方式加速：
- **算子融合**：把多个小算子合并成一个大 kernel，减少 GPU 启动开销。
- **内存优化**：减少中间张量的分配和复制。
- **代码生成**：用 Triton 或 CUDA 生成优化的 kernel。

### 3.2 应用到 UNet / Transformer

```python
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

# 编译 UNet（主要耗时组件）
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

# 编译 VAE decoder
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="reduce-overhead", fullgraph=True)

# 第一次推理会慢（编译中），后续会快很多
# 建议做 warmup
for _ in range(3):
    _ = pipe("warmup", num_inference_steps=20)

# 正式推理
image = pipe("a starry night, Van Gogh style", num_inference_steps=25).images[0]
```

### 3.3 `torch.compile` mode 选择

| Mode | 编译时间 | 运行速度 | 显存 | 推荐场景 |
|---|---|---|---|---|
| `default` | 中等 | 中等加速 | 不变 | 通用 |
| `reduce-overhead` | 较长 | 最快 | 略增 | 推理服务（编译一次，推理多次） |
| `max-autotune` | 很长 | 最快 | 略增 | 极致性能（愿意等编译） |

### 3.4 典型加速效果

| 模型 | 未编译 | 编译后 | 加速比 |
|---|---|---|---|
| SDXL UNet (A100) | ~42 ms/step | ~28 ms/step | ~1.5x |
| FLUX DiT (A100) | ~120 ms/step | ~80 ms/step | ~1.5x |
| SD 1.5 UNet (RTX 4090) | ~18 ms/step | ~12 ms/step | ~1.5x |

> **注意**：加速比因模型、硬件、PyTorch 版本而异。实际数字需要自己 benchmark。

---

## 4. TensorRT 加速

### 4.1 原理

TensorRT 将 PyTorch 模型编译为优化的 GPU 引擎：
- 层融合、kernel 自动调优
- FP16/INT8 精度优化
- 针对特定 GPU 架构优化

### 4.2 使用 `torch_tensorrt`

```python
import torch
import torch_tensorrt
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

# 使用 torch.compile 的 TensorRT 后端
pipe.unet = torch.compile(
    pipe.unet,
    backend="torch_tensorrt",
    options={
        "truncate_long_and_double": True,
        "precision": torch.float16,
    },
)

image = pipe("a photo of an astronaut riding a horse").images[0]
```

### 4.3 使用 Diffusers 的 ONNX + TensorRT 路径

```bash
pip install optimum[onnxruntime-gpu]
```

```python
from optimum.onnxruntime import ORTStableDiffusionXLPipeline

pipe = ORTStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    export=True,
    provider="TensorrtExecutionProvider",
)

image = pipe("a beautiful landscape painting").images[0]
```

> **权衡**：TensorRT 编译耗时长（几分钟到几十分钟），且固定了输入尺寸。
> 适合生产部署（固定分辨率、大量推理），不适合开发调试。

---

## 5. ONNX Runtime

### 5.1 导出与优化

```python
from optimum.onnxruntime import ORTStableDiffusionPipeline

# 导出为 ONNX 并加载
pipe = ORTStableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    export=True,
)

# 或从已导出的模型加载
pipe = ORTStableDiffusionPipeline.from_pretrained("./sd-onnx/")

image = pipe("a cat in space, digital art").images[0]
```

### 5.2 选择 Execution Provider

| Provider | 硬件 | 性能 | 安装复杂度 |
|---|---|---|---|
| `CPUExecutionProvider` | CPU | 基线 | 简单 |
| `CUDAExecutionProvider` | NVIDIA GPU | 快 | 中等 |
| `TensorrtExecutionProvider` | NVIDIA GPU | 最快 | 复杂 |
| `CoreMLExecutionProvider` | Apple Silicon | 快 | 中等 |
| `DirectMLExecutionProvider` | Windows GPU | 快 | 简单 |

---

## 6. 注意力优化

Attention 是扩散模型中最耗时、最耗显存的操作。优化注意力可以同时提速和省显存。

### 6.1 三种优化方案

| 方案 | 来源 | 安装 | 说明 |
|---|---|---|---|
| **SDPA** | PyTorch 2.0+ 内置 | 无需安装 | `F.scaled_dot_product_attention` |
| **xformers** | Meta 开源 | `pip install xformers` | Memory-efficient attention |
| **FlashAttention** | Tri Dao | `pip install flash-attn` | 最快，SDPA 后端之一 |

### 6.2 SDPA（推荐，零配置）

PyTorch 2.0+ 的 `scaled_dot_product_attention` 会自动选择最优后端：

```python
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

# Diffusers 在 PyTorch 2.0+ 上默认使用 SDPA，无需任何配置
# 如果想确认，可以查看：
# pipe.unet.config._attn_implementation  # "sdpa"
```

### 6.3 xformers（PyTorch < 2.0 或需要更多控制）

```python
pipe.enable_xformers_memory_efficient_attention()

# 关闭
pipe.disable_xformers_memory_efficient_attention()
```

### 6.4 性能对比（SD 1.5, 512×512, 50 steps, A100）

| 方案 | 推理时间 | 峰值显存 |
|---|---|---|
| 原始 Attention | ~5.2 s | ~7.4 GB |
| xformers | ~3.8 s | ~4.8 GB |
| SDPA (FlashAttn 后端) | ~3.6 s | ~4.6 GB |

---

## 7. VAE 优化

VAE 负责将 latent 解码为像素图像，在高分辨率场景下是显存瓶颈。

### 7.1 VAE Tiling（分块解码）

把大 latent 切成小块分别解码，然后拼接。几乎不影响质量，显存大幅降低。

```python
pipe.enable_vae_tiling()

image = pipe(
    "a panoramic view of a mountain range",
    height=1024,
    width=2048,
    num_inference_steps=30,
).images[0]
```

### 7.2 VAE Slicing（batch 切片）

多张图同时生成时，逐张解码 VAE 而不是一次性 batch 解码。

```python
pipe.enable_vae_slicing()

images = pipe(
    ["prompt 1", "prompt 2", "prompt 3", "prompt 4"],
    num_inference_steps=30,
).images
```

### 7.3 FP16 VAE

某些 VAE 在 FP16 下会产生 NaN（黑图）。Diffusers 提供了专门的 FP16 修复版本：

```python
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
).to("cuda")
```

---

## 8. 模型蒸馏：从根本上减少步数

### 8.1 思路

与其在推理时找更好的 Scheduler，不如训一个**天生只需要少步数的模型**。

| 方法 | 步数 | 质量 | 说明 |
|---|---|---|---|
| LCM (Latent Consistency Model) | 2-8 | 好 | 通过 consistency distillation |
| SDXL-Turbo | 1-4 | 好 | Adversarial diffusion distillation |
| SDXL-Lightning | 1-4 | 很好 | Progressive distillation |
| FLUX.1-schnell | 1-4 | 很好 | FLUX 官方蒸馏版 |

### 8.2 SDXL-Turbo：1 步出图

```python
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

image = pipe(
    "a cinematic shot of a baby raccoon wearing a hat",
    num_inference_steps=1,       # 1 步！
    guidance_scale=0.0,          # Turbo 不需要 CFG
).images[0]
```

### 8.3 SDXL-Lightning：灵活步数

```python
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# 加载 4-step Lightning LoRA
pipe.load_lora_weights(
    hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors")
)

pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    timestep_spacing="trailing",
)

image = pipe(
    "a photo of a woman in a red dress",
    num_inference_steps=4,
    guidance_scale=0.0,
).images[0]
```

### 8.4 FLUX.1-schnell

```python
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe(
    "a tiny astronaut hatching from an egg on the moon",
    num_inference_steps=4,
    guidance_scale=0.0,
).images[0]
```

---

## 9. 显存优化

当 GPU 显存不够用时，可以用以下策略换取更多的可用空间。

### 9.1 CPU Offloading

```python
# 模型级 offload：整个模型在 CPU/GPU 间移动
pipe.enable_model_cpu_offload()

# 更激进：Sequential offload（逐层搬运）
# 更省显存，但更慢
pipe.enable_sequential_cpu_offload()
```

**对比**：

| 方式 | 峰值显存 | 速度损失 | 原理 |
|---|---|---|---|
| 全部在 GPU | 最高 | 无 | 所有组件常驻显存 |
| `enable_model_cpu_offload()` | 中等 | ~10-20% | 组件级搬运（UNet → VAE） |
| `enable_sequential_cpu_offload()` | 最低 | ~50-100% | 逐层搬运 |

### 9.2 Attention Slicing

```python
pipe.enable_attention_slicing()
# 或指定切片大小
pipe.enable_attention_slicing(slice_size=1)
```

> Attention Slicing 在 PyTorch 2.0+ SDPA 下通常不需要，SDPA 自身就是 memory-efficient 的。

### 9.3 FP16 / BF16 推理

```python
import torch
from diffusers import StableDiffusionXLPipeline

# FP16（最常用）
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,  # 显存减半，速度翻倍
).to("cuda")

# BF16（Ampere+ GPU）
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16,
).to("cuda")
```

| 精度 | 显存 | 速度 | 数值稳定性 | 推荐 |
|---|---|---|---|---|
| FP32 | 2x | 1x | 最好 | 不推荐用于推理 |
| FP16 | 1x | 2x | 好（偶尔溢出） | **默认选择** |
| BF16 | 1x | 2x | 很好（范围更大） | Ampere+ GPU 推荐 |

---

## 10. 分辨率与 Batch Size 的影响

### 10.1 分辨率影响

Attention 的计算量与分辨率的**平方**成正比：

```
512×512  → 262,144 像素 → 基准
1024×1024 → 1,048,576 像素 → 4 倍计算量
2048×2048 → 4,194,304 像素 → 16 倍计算量
```

> 生成大图的策略：先在低分辨率生成，再用超分辨率模型放大（如 SDXL Refiner、Real-ESRGAN）。

### 10.2 Batch Size 的影响

```python
# 多图生成
images = pipe(
    prompt="a beautiful sunset",
    num_images_per_prompt=4,  # 一次生成 4 张
    num_inference_steps=25,
).images

# batch size 越大：
# - 吞吐越高（总 images/sec 更多）
# - 单图延迟不变或略增
# - 显存线性增长
# - 注意不要 OOM
```

---

## 11. 实用加速配方

### 配方 1：通用高质量（推荐起点）

```python
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="dpmsolver++",
    use_karras_sigmas=True,
)

pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
pipe.enable_vae_tiling()
```

### 配方 2：极速实时（牺牲一些质量）

```python
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
).to("cuda")

pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

# 1 step, 512×512 → 可达 ~50ms/image (A100)
image = pipe("prompt", num_inference_steps=1, guidance_scale=0.0).images[0]
```

### 配方 3：显存受限（8GB GPU）

```python
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
)

pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()
pipe.enable_vae_slicing()

image = pipe("a landscape photo", num_inference_steps=25).images[0]
```

---

## 12. 性能基准参考

> 以下数据仅供参考，实际性能受硬件、驱动、PyTorch 版本影响。

### SDXL (1024×1024, 25 steps)

| 优化组合 | A100 (80GB) | RTX 4090 | RTX 3090 |
|---|---|---|---|
| FP32, 原始 | ~12 s | ~18 s | ~25 s |
| FP16 | ~5.5 s | ~7 s | ~10 s |
| FP16 + SDPA | ~4.5 s | ~5.5 s | ~8 s |
| FP16 + SDPA + compile | ~3.2 s | ~4 s | ~6 s |
| SDXL-Turbo, 1 step | ~0.2 s | ~0.3 s | ~0.5 s |
| SDXL-Lightning, 4 steps | ~0.7 s | ~1.0 s | ~1.5 s |

### FLUX.1-schnell (1024×1024, 4 steps)

| 优化组合 | A100 (80GB) | RTX 4090 |
|---|---|---|
| BF16 | ~2.5 s | ~4.0 s |
| BF16 + compile | ~1.8 s | ~2.8 s |

---

## 13. 常见坑

### 13.1 `torch.compile` 首次编译慢

```python
# 第一次推理会触发编译，耗时几十秒到几分钟
# 解决方案：
# 1. 服务启动时做 warmup（跑 2-3 次推理）
# 2. 使用 torch._dynamo.config.cache_size_limit 调大编译缓存
# 3. 输入形状尽量固定（动态形状会导致重新编译）
```

### 13.2 `torch.compile` + 动态形状

```python
# 扩散模型经常需要不同分辨率，每个新形状都会触发重新编译
# 解决方案：
# 1. 固定输出分辨率（如只支持 512x512, 1024x1024）
# 2. 使用 dynamic=True（但加速效果会打折扣）

pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", dynamic=True)
```

### 13.3 FP16 VAE 出黑图

```python
# 某些 VAE（特别是 SDXL 的）在 FP16 下数值不稳定，解码出 NaN
# 症状：输出全黑或有大块色块
# 解决：
# 1. 使用 madebyollin/sdxl-vae-fp16-fix
# 2. 或者 VAE 用 FP32，其他用 FP16（但会多占显存）
pipe.vae = pipe.vae.to(torch.float32)
```

### 13.4 xformers 与 SDPA 冲突

```python
# PyTorch 2.0+ 已内置 SDPA，不需要再装 xformers
# 同时启用可能产生意外行为
# 建议：PyTorch 2.0+ 直接用 SDPA（默认启用，无需配置）
```

### 13.5 CPU Offload 导致速度骤降

```python
# enable_sequential_cpu_offload 虽然省显存，但速度下降 50-100%
# 因为每一层都要 CPU → GPU 搬运

# 更好的选择（按优先级）：
# 1. 换更小的模型
# 2. 降低分辨率
# 3. 用 enable_model_cpu_offload()（组件级，损失小）
# 4. 最后才考虑 enable_sequential_cpu_offload()
```

### 13.6 Scheduler 参数没有随模型调整

```python
# 不同蒸馏模型需要不同的 Scheduler 配置
# SDXL-Turbo: guidance_scale=0.0, 不需要负面 prompt
# LCM: guidance_scale=1.0-2.0, 需要 LCMScheduler
# Lightning: guidance_scale=0.0, 需要 trailing timestep_spacing

# 错误示范：用 SDXL-Turbo 但设 guidance_scale=7.5 → 效果极差
```

---

## 14. 小结

| 优化手段 | 加速倍数 | 显存影响 | 质量影响 | 实施难度 |
|---|---|---|---|---|
| FP16 推理 | ~2x | 减半 | 无 | ★ |
| SDPA / FlashAttn | ~1.3x | 减少 | 无 | ★ |
| `torch.compile` | ~1.5x | 略增 | 无 | ★★ |
| DPM++ Scheduler | ~2x（vs DDIM 50 步） | 不变 | 微小 | ★ |
| TensorRT | ~1.5-2x | 不变 | 无 | ★★★★ |
| LCM/Turbo 蒸馏 | ~5-10x | 不变 | 轻微下降 | ★★ |
| VAE Tiling | 不加速 | 大幅减少 | 无 | ★ |
| CPU Offload | 减速 | 大幅减少 | 无 | ★ |

**一句话**：**先 FP16 + SDPA + DPM++ 25 步打底，需要更快就上 `torch.compile`，极速场景用蒸馏模型。**
