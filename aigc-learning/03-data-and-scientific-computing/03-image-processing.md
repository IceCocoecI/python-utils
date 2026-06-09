# 03 · 图像处理：Pillow / OpenCV / torchvision

> AIGC 图像方向的工程师，对图像格式、通道顺序、数据类型必须了如指掌。
> 一半以上的"训出来图像不对"问题都来自数据前后处理的 bug。

---

## 1. 三大库的定位

| 库 | 定位 | 典型场景 |
|---|---|---|
| **Pillow (PIL)** | Python 原生图像库 | 单张图片读取/保存、简单变换 |
| **OpenCV** | C++ 高性能库 | 实时处理、复杂算法、视频 |
| **torchvision.transforms** | PyTorch 数据管线 | 训练/推理时的张量化 |

**90% 的 AIGC 代码用 Pillow + torchvision**；只在需要高性能或高级算法时引入 OpenCV。

---

## 2. 关键常识：通道顺序 & 形状

这是最容易搞混的地方，**请背诵**：

| 库 | 通道顺序 | 形状 | 数据类型 | 值域 |
|---|---|---|---|---|
| PIL `Image` | RGB | 抽象（无 shape） | uint8 | [0, 255] |
| NumPy (from PIL) | RGB | `(H, W, C)` | uint8 | [0, 255] |
| **OpenCV `imread`** | **BGR** ⚠️ | `(H, W, C)` | uint8 | [0, 255] |
| PyTorch Tensor | RGB | `(C, H, W)` | float32 | [0, 1] |
| matplotlib `imshow` | RGB | `(H, W, C)` | uint8 或 float | - |

**陷阱警告**：OpenCV 的 BGR 是最容易出错的地方。只要你看到红色变蓝、蓝色变红——99% 是忘了 BGR→RGB 转换。

---

## 3. Pillow 基础

```python
from PIL import Image

img = Image.open("cat.jpg")
print(img.size, img.mode)

rgb = img.convert("RGB")
gray = img.convert("L")

resized = img.resize((256, 256), resample=Image.BICUBIC)
cropped = img.crop((10, 10, 210, 210))
rotated = img.rotate(30)
flipped = img.transpose(Image.FLIP_LEFT_RIGHT)

img.save("out.png")

import numpy as np
arr = np.array(img)
img2 = Image.fromarray(arr)
```

---

## 4. OpenCV 基础（注意 BGR！）

```python
import cv2
import numpy as np

img = cv2.imread("cat.jpg")
print(img.shape, img.dtype)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("out.png", img)

blur = cv2.GaussianBlur(img, (5, 5), sigmaX=1.0)
edges = cv2.Canny(gray, 100, 200)
```

---

## 5. torchvision.transforms：训练数据管线的标配

### 5.1 v2 API（推荐，PyTorch 2.x）

```python
import torch
from torchvision.transforms import v2
from PIL import Image

train_transform = v2.Compose([
    v2.RandomResizedCrop(224, antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = v2.Compose([
    v2.Resize(256, antialias=True),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("cat.jpg").convert("RGB")
tensor = train_transform(img)
print(tensor.shape, tensor.dtype, tensor.min().item(), tensor.max().item())
```

### 5.2 ImageNet 归一化（必背数字）

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
```

### 5.3 反归一化（可视化时用）

```python
def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean
```

---

## 6. 完整流程：自定义 Dataset

```python
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image


class ImageFolderDataset(Dataset):
    """从目录加载图像 + 标签。"""

    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.items: list[tuple[Path, int]] = []
        for cls in classes:
            for p in (self.root / cls).glob("*.jpg"):
                self.items.append((p, self.class_to_idx[cls]))

        self.transform = transform or v2.Compose([
            v2.Resize((224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
```

---

## 7. AIGC 场景小抄

### 7.1 扩散模型的图像预处理

SD 系列的 VAE 要求输入 [-1, 1] 而不是 ImageNet 归一化：

```python
vae_transform = v2.Compose([
    v2.Resize(512, antialias=True),
    v2.CenterCrop(512),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
```

### 7.2 把 Tensor 批量存成图片

```python
from torchvision.utils import make_grid, save_image

save_image(tensor, "out.png", normalize=True, value_range=(-1, 1))

grid = make_grid(tensors, nrow=4, normalize=True, value_range=(-1, 1))
save_image(grid, "grid.png")
```

### 7.3 PIL ↔ Tensor ↔ NumPy 的相互转换

```python
from torchvision.transforms import v2

to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
t = to_tensor(pil_img)

arr = (t.permute(1, 2, 0).numpy() * 255).astype("uint8")

from torchvision.transforms.functional import to_pil_image
pil_img = to_pil_image(t)
```

### 7.4 快速可视化 batch

```python
import matplotlib.pyplot as plt
from einops import rearrange

def show_batch(tensor: torch.Tensor, n: int = 8):
    """tensor: (B, C, H, W) in [-1, 1] or [0, 1]."""
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    grid = rearrange(tensor[:n], "b c h w -> h (b w) c")
    plt.imshow(grid.clamp(0, 1).numpy())
    plt.axis("off")
    plt.show()
```

---

## 8. 常见坑 & 调试技巧

### 8.1 "我的图看起来像火星人"

排查顺序：
1. 通道顺序（RGB vs BGR）？
2. 数据类型（uint8 vs float32）？
3. 值域（[0,1] vs [0,255] vs [-1,1]）？
4. 归一化用对了没？

### 8.2 `Image.open` 是惰性的

```python
img = Image.open("cat.jpg")
img.load()
```

不然你关了文件，后续读像素时可能报错。

### 8.3 大图抗锯齿

`torchvision` v2 里 `Resize` 建议加 `antialias=True`，否则下采样会有锯齿。

### 8.4 OpenCV 中文路径

`cv2.imread` 不支持中文路径，用：

```python
import numpy as np
img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
```

---

## 9. Albumentations：更强大的增强库

`torchvision.transforms.v2` 覆盖了 80% 场景，剩下 20% 需要更强的组合时，用 **Albumentations**。

### 9.1 为什么用 Albumentations？

- **更多增强**：70+ 种，包括 MotionBlur、ElasticTransform、GridDistortion、CoarseDropout。
- **联合变换**：一次调用**同步**变换 image + mask + keypoint + bbox（分割/检测任务必备）。
- **速度**：底层用 OpenCV（比 PIL 快 2–5×）。
- **概率控制**：每个操作都能设 `p=0.5`，组合出丰富的增强空间。

### 9.2 典型用法

```python
import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=1.0),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(std_range=(0.05, 0.2)),
    ], p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

img = cv2.cvtColor(cv2.imread("cat.jpg"), cv2.COLOR_BGR2RGB)
out = transform(image=img)
tensor = out["image"]
```

### 9.3 分割任务的"同步变换"

```python
seg_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)),
])

out = seg_transform(image=img, mask=mask)
aug_img, aug_mask = out["image"], out["mask"]
```

**注意**：图像做了翻转，mask 会**自动同步翻转**——这是 torchvision 直到 v2 才补齐的功能，Albumentations 一直原生支持。

### 9.4 什么时候该用哪个？

| 任务 | 首选 |
|---|---|
| 分类 | torchvision v2 |
| 扩散模型训练 | torchvision v2 |
| 分割 / 检测 | **Albumentations** |
| 需要 MotionBlur / ElasticTransform 等奇技 | **Albumentations** |
| 纯 PIL 生态 / 轻依赖 | torchvision |

---

## 10. 生成图像质量评估指标

AIGC 训练最大的难题不是 loss 好不好看，而是**"生成的图真的好看吗"**。
以下是业界标准的评估指标。

### 10.1 像素级（需要参考图）

**PSNR（Peak Signal-to-Noise Ratio）**：值越大越像。

```python
from torchmetrics.image import PeakSignalNoiseRatio
psnr = PeakSignalNoiseRatio()
val = psnr(pred, target)
```

**SSIM（Structural Similarity）**：结构相似度，比 PSNR 更符合人眼。

```python
from torchmetrics.image import StructuralSimilarityIndexMeasure
ssim = StructuralSimilarityIndexMeasure()
val = ssim(pred, target)
```

适用场景：**超分、去噪、修复**等有 ground truth 的任务。

### 10.2 感知级（深度特征）

**LPIPS（Learned Perceptual Image Patch Similarity）**：用预训练 VGG/AlexNet 特征算距离，值越小越像。

```python
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
val = lpips(pred, target)
```

比 PSNR/SSIM 更贴合"人眼看起来像"的直觉。

### 10.3 分布级（无需配对参考）

**FID（Fréchet Inception Distance）**：生成图与真图的特征分布距离，**GAN / 扩散模型的金标准**。值越小越好（0 = 完全一致）。

```python
from torchmetrics.image.fid import FrechetInceptionDistance

fid = FrechetInceptionDistance(feature=2048)
fid.update(real_imgs, real=True)
fid.update(fake_imgs, real=False)
print(fid.compute())
```

**注意事项**：
- 需要至少 **10,000 张**图才稳定（< 2,000 张的 FID 噪声很大）。
- 输入必须是 `uint8`、`(N, 3, H, W)`、`H=W=299`。
- SD 论文报的 FID 在 **MS-COCO-30K** 上计算。

### 10.4 多模态（评估文生图"符合文本"）

**CLIPScore**：用 CLIP 算图文相似度，反映"图是否符合 prompt"。

```python
from torchmetrics.multimodal.clip_score import CLIPScore

clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
score = clip_score(images, prompts)
```

SD / DALL-E / Flux 评测必用。

### 10.5 指标怎么选？

| 任务 | 首选指标 |
|---|---|
| 超分 / 去噪 | PSNR + SSIM + LPIPS |
| GAN / 扩散模型（无条件） | FID |
| 文生图 | FID + CLIPScore（兼顾质量和对齐） |
| 图像修复 | LPIPS + FID |
| **最终版本** | **加上人工评测**——所有指标都会骗人 |

---

## 11. 其他模态简介（有概念即可）

### 11.1 视频：`decord`

```python
from decord import VideoReader, cpu

vr = VideoReader("video.mp4", ctx=cpu(0))
print(len(vr), vr.get_avg_fps())

frames = vr.get_batch([0, 30, 60, 90]).asnumpy()
```

Sora / CogVideoX 等视频生成训练的标配数据加载器。

### 11.2 音频：`torchaudio` / `librosa`

```python
import torchaudio

waveform, sr = torchaudio.load("audio.wav")
print(waveform.shape, sr)

mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=80)(waveform)
```

TTS（VITS、XTTS）、ASR（Whisper）、音乐生成（MusicGen）都基于这些工具。

---

## 小结

- **PIL + torchvision v2** 是 AIGC 数据管线的默认组合；增强进阶用 **Albumentations**。
- 背熟通道顺序 / 形状 / dtype / 值域四件事，少走 90% 弯路。
- ImageNet 归一化 = 分类任务默认；SD 系列 VAE = [-1, 1]。
- 评估图像质量：像素级（PSNR/SSIM）→ 感知级（LPIPS）→ 分布级（FID/CLIPScore），**最后还要人工评测**。
- 出问题先打印 `(shape, dtype, min, max)`，比什么调试器都有效。

至此 `03-data-and-scientific-computing` 模块完成。恭喜你走完了 AIGC 算法工程师的**编程基础三大支柱**！
