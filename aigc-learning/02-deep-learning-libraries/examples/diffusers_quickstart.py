"""HuggingFace Diffusers 快速入门示例。
演示：加载 SD、文生图、切换 Scheduler、显存优化、图生图

运行：python diffusers_quickstart.py
建议 GPU 环境（>= 6GB）。纯 CPU 跑 SD 会非常慢。
需要联网下载权重（SD1.5 约 4GB）。
"""
from __future__ import annotations

from pathlib import Path

import torch


OUT_DIR = Path("./outputs")
OUT_DIR.mkdir(exist_ok=True)


def demo_text2image():
    print("== 1. 文生图 ==")
    from diffusers import StableDiffusionPipeline

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

    out = OUT_DIR / "panda.png"
    image.save(out)
    print(f"saved: {out.resolve()}")
    return pipe


def demo_scheduler(pipe):
    print("\n== 2. 换 Scheduler：DPM++ 2M（20 步） ==")
    from diffusers import DPMSolverMultistepScheduler

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    image = pipe(
        "a photo of an astronaut surfing on a wave, vibrant colors, 8k",
        num_inference_steps=20,
        guidance_scale=7.0,
    ).images[0]
    out = OUT_DIR / "astronaut.png"
    image.save(out)
    print(f"saved: {out.resolve()}")


def demo_img2img():
    print("\n== 3. 图生图 ==")
    from diffusers import StableDiffusionImg2ImgPipeline
    from PIL import Image

    img_path = OUT_DIR / "panda.png"
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
    out = OUT_DIR / "panda_watercolor.png"
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
    pipe = demo_text2image()
    demo_scheduler(pipe)
    demo_img2img()
    demo_memory_profile()
