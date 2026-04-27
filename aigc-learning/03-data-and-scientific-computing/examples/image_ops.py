"""图像处理综合示例：PIL / NumPy / PyTorch 之间的转换 + torchvision v2 管线。
运行：python image_ops.py
会在当前目录生成一张测试图 test.jpg。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

OUT = Path(".")


def create_test_image():
    """生成一张彩色测试图（避免依赖本地图片文件）。"""
    h, w = 256, 256
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    rgb[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    rgb[:, :, 2] = 128
    img = Image.fromarray(rgb)
    img.save(OUT / "test.jpg")
    print(f"created: {OUT / 'test.jpg'}")
    return img


def demo_pil_numpy():
    print("\n== 1. PIL <-> NumPy ==")
    img = Image.open(OUT / "test.jpg").convert("RGB")
    print(f"PIL size: {img.size}, mode: {img.mode}")

    arr = np.array(img)
    print(f"NumPy shape: {arr.shape}, dtype: {arr.dtype}, range: [{arr.min()}, {arr.max()}]")

    img2 = Image.fromarray(arr)
    print(f"back to PIL: {img2.size}")


def demo_basic_ops():
    print("\n== 2. 基础操作（resize / crop / flip） ==")
    img = Image.open(OUT / "test.jpg").convert("RGB")
    resized = img.resize((128, 128), resample=Image.BICUBIC)
    cropped = img.crop((32, 32, 160, 160))
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    rotated = img.rotate(30, expand=True)

    for name, im in [("resized", resized), ("cropped", cropped), ("flipped", flipped), ("rotated", rotated)]:
        p = OUT / f"test_{name}.jpg"
        im.save(p)
        print(f"  {name}: {im.size} -> {p.name}")


def demo_torchvision_v2():
    print("\n== 3. torchvision.transforms.v2 训练管线 ==")
    from torchvision.transforms import v2

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_tf = v2.Compose([
        v2.RandomResizedCrop(224, antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    img = Image.open(OUT / "test.jpg").convert("RGB")
    t = train_tf(img)
    print(f"tensor shape: {t.shape}, dtype: {t.dtype}")
    print(f"tensor range: [{t.min():.3f}, {t.max():.3f}]")
    print(f"per-channel mean: {t.mean(dim=(1, 2))}")


def demo_tensor_to_image():
    print("\n== 4. Tensor -> Image 可视化（反归一化） ==")
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    from torchvision.transforms import v2
    tf = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(OUT / "test.jpg").convert("RGB")
    t = tf(img)

    denorm = (t * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    arr = (denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(OUT / "test_roundtrip.jpg")
    print(f"round-trip saved: {OUT / 'test_roundtrip.jpg'}")


def demo_vae_style_normalize():
    print("\n== 5. SD VAE 风格：归一化到 [-1, 1] ==")
    from torchvision.transforms import v2
    tf = v2.Compose([
        v2.Resize((128, 128), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(OUT / "test.jpg").convert("RGB")
    t = tf(img)
    print(f"shape: {t.shape}, range: [{t.min():.3f}, {t.max():.3f}]")


def demo_save_grid():
    print("\n== 6. 批量图像网格保存 ==")
    from torchvision.utils import make_grid, save_image
    from torchvision.transforms import v2

    tf = v2.Compose([
        v2.Resize((64, 64), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    img = Image.open(OUT / "test.jpg").convert("RGB")
    batch = torch.stack([tf(img) for _ in range(8)])
    grid = make_grid(batch, nrow=4)
    save_image(grid, OUT / "grid.png")
    print(f"saved grid: {OUT / 'grid.png'}")


if __name__ == "__main__":
    create_test_image()
    demo_pil_numpy()
    demo_basic_ops()
    demo_torchvision_v2()
    demo_tensor_to_image()
    demo_vae_style_normalize()
    demo_save_grid()
