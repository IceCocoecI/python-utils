# 04 · HuggingFace Diffusers

> Diffusers 是扩散模型的"事实标准"：SD 1.5 / SDXL / Flux / Kolors / CogVideoX / Qwen-Image 等 SOTA 模型都基于它构建。
> 本节讲清楚 **Pipeline / Model / Scheduler** 三大抽象，并动手做文生图、图生图、ControlNet、LoRA。

---

## 1. 30 秒上手：文生图

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe(
    "a photo of a red panda eating noodles, cinematic lighting",
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]

image.save("panda.png")
```

三行就能跑 SD。Diffusers 的哲学是"简单易用优先，可配置优先于抽象"。

---

## 2. 三大核心抽象

```
Pipeline  ——  高层 API，把 Model+Scheduler+Tokenizer 串起来
  ├── Model     —— 神经网络（UNet / DiT / VAE / Text Encoder）
  └── Scheduler —— 噪声调度/采样（DDIM / DPM++ / Euler ...）
```

### 2.1 Pipeline 全家福

| Pipeline | 用途 |
|---|---|
| `StableDiffusionPipeline` | SD 1.5 / 2.1 文生图 |
| `StableDiffusionXLPipeline` | SDXL 文生图 |
| `StableDiffusionImg2ImgPipeline` | 图生图 |
| `StableDiffusionInpaintPipeline` | 图像补全 |
| `StableDiffusionControlNetPipeline` | ControlNet 条件控制 |
| `FluxPipeline` | Flux 系列（2024 年 SOTA） |
| `CogVideoXPipeline` | 视频生成 |
| `AutoPipelineForText2Image` | 根据权重自动选 Pipeline |

### 2.2 AutoPipeline：最灵活的加载方式

```python
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")
```

它会自动识别是 SD / SDXL / Flux 并加载正确的 Pipeline。

---

## 3. Scheduler：扩散采样器

**记住一点**：UNet 每一步预测噪声，Scheduler 决定"怎么用预测的噪声更新图像"。
换 Scheduler 不换模型，就能改变生成质量和速度。

```python
from diffusers import DPMSolverMultistepScheduler

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

image = pipe(prompt, num_inference_steps=20).images[0]
```

### 3.1 常用 Scheduler 推荐

| Scheduler | 特点 |
|---|---|
| `DDIMScheduler` | 最稳定，确定性采样 |
| `DPMSolverMultistepScheduler` (DPM++ 2M) | 20 步出好图，主流默认 |
| `EulerDiscreteScheduler` | SDXL 默认，简单快速 |
| `EulerAncestralDiscreteScheduler` | 带噪声，多样性好 |
| `UniPCMultistepScheduler` | 8–10 步即可 |
| `LCMScheduler` | 搭配 LCM LoRA，4 步出图 |

---

## 4. 图生图 / 局部重绘 / ControlNet

### 4.1 Image-to-Image

```python
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

init = Image.open("cat.jpg").resize((512, 512))
out = pipe(
    prompt="a watercolor painting of a cat",
    image=init,
    strength=0.7,
    guidance_scale=7.5,
).images[0]
```

`strength` ∈ [0, 1]：越大越"重画"，越小越保留原图。

### 4.2 ControlNet：给生成加结构条件

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

cn = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=cn,
    torch_dtype=torch.float16,
).to("cuda")

image = pipe(prompt="a knight in armor", image=canny_image).images[0]
```

ControlNet 的条件类型：canny / depth / pose / segmentation / scribble / normal map 等。

---

## 5. LoRA：轻量级风格微调

### 5.1 加载已有 LoRA

```python
pipe.load_lora_weights("path/or/hub_id_to/lora.safetensors")

pipe.set_adapters(["style_lora"], adapter_weights=[0.8])

pipe.load_lora_weights("anime.safetensors", adapter_name="anime")
pipe.load_lora_weights("sketch.safetensors", adapter_name="sketch")
pipe.set_adapters(["anime", "sketch"], adapter_weights=[0.6, 0.4])
```

### 5.2 训练自己的 LoRA

Diffusers 官方 `examples/` 提供了完整训练脚本：

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers/examples/dreambooth

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="./my_dog_photos" \
  --output_dir="./lora-dog" \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --max_train_steps=500 \
  --mixed_precision="fp16"
```

5–20 张图就能做出一个个性化 LoRA。

---

## 6. 性能优化

### 6.1 显存优化

```python
pipe.enable_attention_slicing()
pipe.enable_vae_tiling()
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
```

**组合用法**：24GB 显存跑 SDXL → `enable_model_cpu_offload()`；8GB 跑 SD 1.5 → `attention_slicing + vae_tiling`。

### 6.2 推理加速

```python
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
image = pipe(prompt, num_inference_steps=20).images[0]

pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
image = pipe(prompt, num_inference_steps=4, guidance_scale=1).images[0]
```

---

## 7. 从零训练一个小扩散模型（UNet2D）

Diffusers 官方 tutorial（butterflies 数据集）展示了完整训练流程。核心代码：

```python
from diffusers import UNet2DModel, DDPMScheduler
from accelerate import Accelerator

model = UNet2DModel(
    sample_size=128,
    in_channels=3, out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=("DownBlock2D",) * 3 + ("AttnDownBlock2D",) + ("DownBlock2D",) * 2,
    up_block_types=("UpBlock2D",) * 2 + ("AttnUpBlock2D",) + ("UpBlock2D",) * 3,
)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

accelerator = Accelerator(mixed_precision="fp16")
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

for epoch in range(num_epochs):
    for batch in loader:
        clean = batch["images"]
        noise = torch.randn_like(clean)
        t = torch.randint(0, 1000, (clean.size(0),), device=clean.device)
        noisy = noise_scheduler.add_noise(clean, noise, t)

        pred = model(noisy, t).sample
        loss = F.mse_loss(pred, noise)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

**核心思想**：
1. 给干净图像 `clean` 加随机噪声 `noise` → `noisy`。
2. UNet 学习根据 `(noisy, t)` 预测噪声 `noise`。
3. 推理时从纯噪声反向去噪，得到图像。

完整脚本见 [diffusers/examples/unconditional_image_generation](https://github.com/huggingface/diffusers/tree/main/examples/unconditional_image_generation)。

---

## 8. 推荐学习资源

### 官方
- [Diffusers 官方文档](https://huggingface.co/docs/diffusers)
- [Diffusion Models Course](https://github.com/huggingface/diffusion-models-class)
- [Diffusers examples/](https://github.com/huggingface/diffusers/tree/main/examples) — 训练脚本金矿

### 论文（必读）
- DDPM：Denoising Diffusion Probabilistic Models
- DDIM：Denoising Diffusion Implicit Models
- Classifier-Free Guidance（CFG）
- Latent Diffusion（SD 的基础）
- DiT：Scalable Diffusion Models with Transformers

### 实战仓库
- [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) — 最流行的 Web UI
- [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) — 节点式工作流
- [black-forest-labs/flux](https://github.com/black-forest-labs/flux) — Flux 官方实现

---

## 小结

- Pipeline = Model + Scheduler + Tokenizer，高层 API 一行推理。
- Scheduler 可热插拔：DPM++ 2M 是万能默认。
- ControlNet / IP-Adapter / LoRA 是扩散模型最实用的三大"外挂"。
- 显存紧张用 `enable_model_cpu_offload()`；加速用 `torch.compile` + LCM。

至此，模块 02 完成。下一步进入 **模块 03：数据处理与科学计算**。
