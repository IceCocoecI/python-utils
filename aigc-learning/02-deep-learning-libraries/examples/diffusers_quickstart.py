"""HuggingFace Diffusers 快速入门示例。
演示：加载 SD、文生图、切换 Scheduler、显存优化、图生图

默认离线运行 toy DDPM：python diffusers_quickstart.py
真实 Stable Diffusion：python diffusers_quickstart.py --stable-diffusion
建议 GPU 环境（>= 6GB）。纯 CPU 跑 SD 会非常慢。
需要联网下载权重（SD1.5 约 4GB）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "outputs"


def make_toy_images(batch_size: int, image_size: int = 32, device: torch.device | str = "cpu") -> torch.Tensor:
    """生成简单几何图像，用于离线验证扩散训练流程。"""
    images = torch.zeros(batch_size, 3, image_size, image_size, device=device)
    rows = torch.arange(image_size, device=device)[None, :, None]
    cols = torch.arange(image_size, device=device)[None, None, :]
    for idx in range(batch_size):
        center = image_size // 4 + (idx % 4) * image_size // 8
        radius = image_size // 8 + (idx % 3)
        circle = (rows - center).pow(2) + (cols - center).pow(2) < radius ** 2
        stripe = ((cols + idx * 3) % 11) < 4
        images[idx, 0] = circle.float()
        images[idx, 1] = stripe.float() * 0.7
        images[idx, 2] = torch.flip(circle.float(), dims=[1]) * 0.8
    return images * 2 - 1


def tensor_to_pil_grid(images: torch.Tensor, nrow: int = 4):
    from PIL import Image
    from torchvision.utils import make_grid

    images = (images.clamp(-1, 1) + 1) / 2
    grid = make_grid(images.cpu(), nrow=nrow)
    array = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(array)


def demo_toy_ddpm(num_train_steps: int = 3, batch_size: int = 4, out_dir: Path = DEFAULT_OUT_DIR, seed: int = 0):
    print("== 0. 离线 Toy DDPM：训练噪声预测 + 采样 ==")
    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
    ).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=50)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for step in range(num_train_steps):
        clean = make_toy_images(batch_size, device=device)
        noise = torch.randn_like(clean)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
        ).long()
        noisy = noise_scheduler.add_noise(clean, noise, timesteps)
        pred = model(noisy, timesteps).sample
        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"step={step} loss={loss.item():.4f}")

    clean_grid = tensor_to_pil_grid(make_toy_images(8, device=device), nrow=4)
    clean_path = out_dir / "toy_clean_grid.png"
    clean_grid.save(clean_path)

    model.eval()
    pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    generator = torch.Generator(device=device).manual_seed(seed)
    sample = pipe(
        batch_size=4,
        generator=generator,
        num_inference_steps=5,
        output_type="pil",
    ).images
    sample_path = out_dir / "toy_ddpm_sample.png"
    sample[0].save(sample_path)
    print(f"saved: {clean_path.resolve()}")
    print(f"saved: {sample_path.resolve()}")


def demo_text2image(out_dir: Path):
    print("== 1. 文生图 ==")
    from diffusers import StableDiffusionPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()

    prompt = "a cute red panda eating noodles, cinematic lighting, ultra detailed"
    negative = "blurry, low quality, watermark, text"
    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=512, height=512,
    ).images[0]

    out = out_dir / "panda.png"
    image.save(out)
    print(f"saved: {out.resolve()}")
    return pipe


def demo_scheduler(pipe, out_dir: Path):
    print("\n== 2. 换 Scheduler：DPM++ 2M（20 步） ==")
    from diffusers import DPMSolverMultistepScheduler

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    image = pipe(
        "a photo of an astronaut surfing on a wave, vibrant colors, 8k",
        num_inference_steps=20,
        guidance_scale=7.0,
    ).images[0]
    out = out_dir / "astronaut.png"
    image.save(out)
    print(f"saved: {out.resolve()}")


def demo_img2img(out_dir: Path):
    print("\n== 3. 图生图 ==")
    from diffusers import StableDiffusionImg2ImgPipeline
    from PIL import Image

    img_path = out_dir / "panda.png"
    if not img_path.exists():
        print(f"skip img2img: {img_path} not found (run demo_text2image first)")
        return

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    init = Image.open(img_path).resize((512, 512))
    out_img = pipe(
        prompt="a watercolor painting of a red panda, soft colors",
        image=init,
        strength=0.7,
        guidance_scale=7.5,
    ).images[0]
    out = out_dir / "panda_watercolor.png"
    out_img.save(out)
    print(f"saved: {out.resolve()}")


def demo_memory_profile():
    print("\n== 4. 显存报告 ==")
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"current: {used:.2f} GB  peak: {peak:.2f} GB")
    else:
        print("no CUDA device")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable-diffusion", action="store_true", help="Run SD1.5 demos; requires weights.")
    parser.add_argument("--toy-steps", type=int, default=3)
    parser.add_argument("--toy-batch-size", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.stable_diffusion:
        pipe = demo_text2image(args.out_dir)
        demo_scheduler(pipe, args.out_dir)
        demo_img2img(args.out_dir)
        demo_memory_profile()
    else:
        demo_toy_ddpm(num_train_steps=args.toy_steps, batch_size=args.toy_batch_size, out_dir=args.out_dir, seed=args.seed)
