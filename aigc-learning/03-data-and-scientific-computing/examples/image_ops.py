"""图像处理综合示例：PIL / OpenCV / NumPy / PyTorch 转换 + 训练增强管线。
运行：python image_ops.py
会在 examples/outputs/image_ops 下生成测试图和中间结果。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

OUT = Path(__file__).resolve().parent / "outputs" / "image_ops"


def create_test_image():
    """生成一张彩色测试图（避免依赖本地图片文件）。"""
    OUT.mkdir(parents=True, exist_ok=True)
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


def demo_opencv_bgr_rgb():
    print("\n== 2. OpenCV：BGR <-> RGB ==")
    import cv2

    img_bgr = cv2.imread(str(OUT / "test.jpg"))
    if img_bgr is None:
        raise RuntimeError("cv2.imread failed")
    print(f"OpenCV shape: {img_bgr.shape}, dtype: {img_bgr.dtype}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    Image.fromarray(img_rgb).save(OUT / "test_opencv_rgb.png")
    Image.fromarray(edges).save(OUT / "test_edges.png")
    print("saved OpenCV RGB and edge maps")


def demo_basic_ops():
    print("\n== 3. 基础操作（resize / crop / flip） ==")
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
    print("\n== 4. torchvision.transforms.v2 训练管线 ==")
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
    print("\n== 5. Tensor -> Image 可视化（反归一化） ==")
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
    print("\n== 6. SD VAE 风格：归一化到 [-1, 1] ==")
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
    print("\n== 7. 批量图像网格保存 ==")
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


def demo_albumentations():
    print("\n== 8. Albumentations：image/mask 同步增强 ==")
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    img = np.array(Image.open(OUT / "test.jpg").convert("RGB"))
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[64:192, 80:176] = 1

    transform = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0), p=1.0),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    out = transform(image=img, mask=mask)
    tensor, aug_mask = out["image"], out["mask"]
    print(f"aug tensor: {tensor.shape}, {tensor.dtype}")
    print(f"aug mask: {aug_mask.shape}, foreground pixels: {int(aug_mask.sum())}")

    vis = tensor.float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    vis = (vis * std + mean).clamp(0, 1)
    vis_arr = (vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    mask_arr = (aug_mask.numpy().astype(np.uint8) * 255)
    cv2.imwrite(str(OUT / "albumentations_image.png"), cv2.cvtColor(vis_arr, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT / "albumentations_mask.png"), mask_arr)
    print("saved Albumentations image/mask outputs")


if __name__ == "__main__":
    create_test_image()
    demo_pil_numpy()
    demo_opencv_bgr_rgb()
    demo_basic_ops()
    demo_torchvision_v2()
    demo_tensor_to_image()
    demo_vae_style_normalize()
    demo_save_grid()
    demo_albumentations()
