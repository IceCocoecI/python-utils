# 04 · 可复现性

> 目标：让你的训练结果**任何人、任何时候、任何机器**上都能复现。
> 这不是强迫症——论文审稿、生产部署、bug 定位都离不开可复现性。

---

## 1. 为什么可复现性如此重要？

| 场景 | 不可复现的后果 |
|---|---|
| 论文审稿 | "我跑不出你的结果" → reject |
| 生产部署 | "线上模型行为和离线不一致" → 事故 |
| Bug 定位 | "这个 bug 偶尔出现，但我复现不了" → 永远修不好 |
| 团队协作 | "我跑出来 loss=0.5，你跑出来 loss=0.8" → 谁的是对的？ |
| 模型选型 | "A 比 B 好 0.3%"—— 这是真的好还是随机波动？ |

**可复现性不是可选的——它是科学方法的基础。**

本章对应示例：

```bash
cd aigc-learning/04-training-engineering/examples
conda run -n aigc python reproducible_train.py --epochs 1
```

示例会用同一份配置连续训练两次，并输出 `reproducibility_report.json`，其中包含配置、环境、数据 checksum、两次指标和一致性检查结果。

---

## 1.1 三个容易混淆的概念

| 概念 | 含义 | 训练工程里的例子 |
|---|---|---|
| Repeatability | 同一机器、同一代码、同一配置，多次运行一致 | 本地连续跑两次 loss 完全一致 |
| Reproducibility | 另一位成员用同一代码、配置、数据和环境跑出等价结果 | 同事按 run 记录恢复出同等指标 |
| Replicability | 独立实现或不同系统验证同一结论 | 另一个框架也证明方法 A 优于方法 B |

日常工程优先保证 reproducibility。bit-level deterministic 更适合 debug 和严格对照实验，不一定适合所有日常训练。

---

## 2. 随机种子：最基础的一步

深度学习训练中有大量随机性来源：

```
随机性来源：
├── Python（random 模块、hash 函数）
├── NumPy（数据增强、采样）
├── PyTorch（权重初始化、dropout、数据洗牌）
│   ├── CPU 随机数
│   └── CUDA 随机数
├── 数据加载（DataLoader 的 worker、shuffle）
└── 第三方库（transformers、diffusers……）
```

### 2.1 完整的种子设置函数

```python
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """设置所有随机种子，尽最大努力保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
```

**在训练脚本的最开头调用**：

```python
if __name__ == "__main__":
    set_seed(42)
    main()
```

### 2.2 HuggingFace 的 `set_seed`

```python
from transformers import set_seed

set_seed(42)
```

内部做了和上面一样的事，直接用即可。

### 2.3 DataLoader 的可复现性

DataLoader 用多进程加载数据时，每个 worker 有自己的随机状态。

```python
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(42)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=g,
)
```

不加 `worker_init_fn` 和 `generator`，即使设了全局 seed，**多 worker 的 DataLoader 仍然不可复现**。

---

## 3. 确定性模式

设了种子还不够——PyTorch 的一些底层算子（尤其是 CUDA 算子）默认使用非确定性实现（更快但不可复现）。

### 3.1 `torch.use_deterministic_algorithms`

```python
torch.use_deterministic_algorithms(True)
```

开启后，所有非确定性算子会抛错，强制你替换成确定性版本。

### 3.2 `CUBLAS_WORKSPACE_CONFIG`

cuBLAS 的矩阵乘法有非确定性问题。需要设置环境变量：

```python
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

或在命令行：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py
```

### 3.3 cuDNN

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

- `deterministic=True`：强制 cuDNN 使用确定性卷积算法
- `benchmark=False`：禁止 cuDNN 自动选最快算法（自动选择本身是非确定性的）

### 3.4 完整的确定性设置

```python
import os
import random

import numpy as np
import torch


def set_deterministic(seed: int = 42) -> None:
    """最大程度保证可复现——代价是速度降低 10-30%。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 3.5 哪些算子不支持确定性？

开启 `use_deterministic_algorithms(True)` 后，如果你用了不支持的算子，会报类似：

```
RuntimeError: ... does not have a deterministic implementation ...
```

常见的不支持确定性的操作：
- `torch.nn.functional.interpolate`（某些模式）
- `scatter_add` / `index_add`
- `ctc_loss`
- 一些 Flash Attention 实现

解决方案：
1. 换用替代实现
2. 对该操作 `torch.use_deterministic_algorithms(True, warn_only=True)` 只警告不报错

```python
torch.use_deterministic_algorithms(True, warn_only=True)
```

---

## 4. 环境管理

**"在我的机器上能跑"** ≠ 可复现。

### 4.1 问题层次

```
可复现环境需要锁定：
├── Python 版本（3.10 vs 3.11 可能行为不同）
├── 包版本（torch 2.3 vs 2.4 行为不同）
├── CUDA / cuDNN 版本
├── 系统库（glibc 等）
└── 硬件（不同 GPU 架构的浮点行为不同）
```

### 4.2 方案对比

| 方案 | 锁定粒度 | 复杂度 | 推荐场景 |
|---|---|---|---|
| `requirements.txt` (pinned) | 包版本 | ★☆☆ | 个人项目 |
| `pyproject.toml` + `uv.lock` | 包版本 + hash | ★★☆ | 推荐默认方案 |
| `conda-lock` | 包 + 系统库 | ★★☆ | 科学计算 |
| **Docker** | 全部（除硬件） | ★★★ | 生产部署 / 论文 |

### 4.3 `requirements.txt`（最基础）

```bash
# 生成（带精确版本）
pip freeze > requirements.txt

# 安装
pip install -r requirements.txt
```

```
# requirements.txt（好的写法：精确到版本）
torch==2.4.0
transformers==4.44.0
datasets==3.0.0
wandb==0.17.5
hydra-core==1.3.2
optuna==3.6.0
```

### 4.4 `uv`（推荐）

```bash
# 初始化项目
uv init my-project
cd my-project

# 添加依赖
uv add torch transformers wandb

# 自动生成 uv.lock（精确到 hash）
# 直接提交到 git

# 别人安装——100% 相同的版本
uv sync
```

`uv.lock` 会精确记录每个包的版本和 hash，跨平台可复现。

### 4.5 Docker（最终方案）

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train.py"]
```

```bash
docker build -t my-training .
docker run --gpus all my-training
```

**Docker 的好处**：不管在谁的机器上跑，环境都是一样的。

### 4.6 记录 CUDA / cuDNN 版本

```python
import torch

env_info = {
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    "gpu_count": torch.cuda.device_count(),
}
print(env_info)
```

把这些信息记录到实验追踪工具（W&B / MLflow）中。

---

## 5. 代码版本

### 5.1 Git Commit Hash

```python
import subprocess


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_git_diff() -> str:
    """检查是否有未提交的修改。"""
    try:
        return subprocess.check_output(
            ["git", "diff", "--stat"], text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return ""
```

```python
import wandb

wandb.init(
    config={
        "git_hash": get_git_hash(),
        "git_dirty": bool(get_git_diff()),
        ...
    },
)
```

**好习惯**：训练前先 `git commit`，确保代码干净。

### 5.2 配置快照

```python
import shutil
from pathlib import Path


def save_config_snapshot(output_dir: str, config_path: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, out / "config.yaml")

    with open(out / "git_info.txt", "w") as f:
        f.write(f"commit: {get_git_hash()}\n")
        f.write(f"diff:\n{get_git_diff()}\n")
```

Hydra 会自动做配置快照（`.hydra/config.yaml`），但 git 信息需要自己加。

---

## 6. 数据版本

### 6.1 数据校验和

```python
import hashlib
from pathlib import Path


def file_checksum(path: str | Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


print(file_checksum("data/train.jsonl"))
# "a3f2b8c..."  ← 记录下来
```

### 6.2 DVC（Data Version Control）

DVC 是 Git 的数据版本管理扩展：

```bash
pip install dvc

# 初始化
dvc init

# 追踪数据文件
dvc add data/train.jsonl
git add data/train.jsonl.dvc data/.gitignore
git commit -m "Track training data with DVC"

# 推送到远程存储
dvc remote add myremote s3://my-bucket/dvc
dvc push

# 别人拉取
git clone <repo>
dvc pull    # 自动下载数据
```

**DVC 的核心思想**：Git 管代码，DVC 管数据。`.dvc` 文件（数据的指针）提交到 Git，实际数据存在云存储。

### 6.3 HuggingFace Datasets 的版本

```python
from datasets import load_dataset

ds = load_dataset("allenai/c4", "en", revision="main")

ds = load_dataset("allenai/c4", "en", revision="v2.0")
```

用 `revision` 锁定数据集版本。

---

## 7. Checkpoint 可复现

保存 checkpoint 时，要保存**完整训练状态**——不只是模型权重。

### 7.1 完整 checkpoint

```python
def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    rng_states: dict | None = None,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "rng_states": rng_states or {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None,
        },
    }
    torch.save(checkpoint, path)
```

### 7.2 恢复训练

```python
def load_checkpoint(path: str, model, optimizer, scheduler):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    random.setstate(ckpt["rng_states"]["python"])
    np.random.set_state(ckpt["rng_states"]["numpy"])
    torch.random.set_rng_state(ckpt["rng_states"]["torch"])
    if ckpt["rng_states"]["cuda"] is not None:
        torch.cuda.set_rng_state_all(ckpt["rng_states"]["cuda"])

    return ckpt["epoch"], ckpt["global_step"], ckpt["best_val_loss"]
```

**保存随机状态的意义**：从 checkpoint 恢复训练后，后续的 batch 顺序、dropout mask 等都和原来完全一样——**仿佛训练从未中断**。

### 7.3 HuggingFace Trainer 的做法

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,       # 只保留最新 3 个 checkpoint
)
```

Trainer 自动保存完整状态（optimizer、scheduler、随机状态）。恢复训练：

```bash
python train.py --resume_from_checkpoint ./output/checkpoint-1500
```

---

## 8. 硬件非确定性

即使你做了上面的一切，**不同 GPU 的浮点行为仍然不同**。

### 8.1 为什么？

```
原因：
├── 不同 GPU 架构（A100 vs V100）使用不同的 CUDA kernel
├── FP16/BF16 的精度差异
├── 不同的浮点运算顺序导致舍入误差积累
├── Tensor Core 的融合运算
└── 多 GPU 通信的 reduce 顺序
```

### 8.2 实际影响

| 场景 | 典型偏差 |
|---|---|
| 同 GPU 同种子 | 完全一致 |
| 同 GPU 不同种子 | ±1-5%（正常） |
| 不同 GPU 同种子 | ±0.1-2%（通常可接受） |
| 不同框架 (PyTorch vs JAX) | ±1-10% |
| FP32 vs BF16 | ±0.5-3% |

### 8.3 建议

1. **论文结果**：报告多个 seed 的 mean ± std
2. **模型对比**：用 paired test（相同数据、相同 seed）
3. **精确复现**：锁定 GPU 型号 + Docker + 确定性模式
4. **接受现实**：0.1% 级别的差异是正常的

---

## 9. 完整的可复现性清单

```
✅ 可复现性 Checklist
├── 随机性
│   ├── [ ] 设置 Python/NumPy/PyTorch 全套随机种子
│   ├── [ ] DataLoader 设置 worker_init_fn + generator
│   ├── [ ] 开启确定性模式（至少在 debug 时）
│   └── [ ] 记录所用的 seed
│
├── 环境
│   ├── [ ] 锁定 Python 版本
│   ├── [ ] 锁定所有包版本（uv.lock / requirements.txt pinned）
│   ├── [ ] 记录 CUDA / cuDNN 版本
│   ├── [ ] 记录 GPU 型号
│   └── [ ] 提供 Docker 镜像（论文 / 生产）
│
├── 代码
│   ├── [ ] 训练前 git commit，代码无 uncommitted 修改
│   ├── [ ] 记录 git commit hash 到实验日志
│   └── [ ] 配置快照保存到输出目录
│
├── 数据
│   ├── [ ] 记录训练数据的 checksum / 版本号
│   ├── [ ] 数据预处理脚本可重跑
│   └── [ ] 数据 split 使用固定 seed
│
├── 训练状态
│   ├── [ ] Checkpoint 包含完整状态（模型 + 优化器 + scheduler + RNG）
│   ├── [ ] 从 checkpoint 恢复后能续跑出一样的结果
│   └── [ ] 配置（超参）完整记录
│
└── 报告
    ├── [ ] 重要结果跑多个 seed，报告 mean ± std
    └── [ ] 记录硬件信息（GPU 型号 / 数量）
```

---

## 10. 常见坑

### 10.1 只设 `torch.manual_seed`，漏了 NumPy 和 Python

```python
# ❌ 不完整
torch.manual_seed(42)

# ✅ 完整
set_seed(42)  # 用前面定义的完整版本
```

### 10.2 多 GPU 训练的种子问题

```python
# ❌ 所有 GPU 用同一个 seed → 数据重复
torch.manual_seed(42)

# ✅ 用 rank 区分（DDP 场景）
torch.manual_seed(42 + rank)
```

但注意：模型初始化的 seed 应该相同（保证所有 rank 初始化一样的权重），只有数据采样的 seed 要区分。HuggingFace Trainer 和 PyTorch DDP 会自动处理这个。

### 10.3 `torch.compile` 的非确定性

`torch.compile` 生成的 kernel 可能有非确定性行为。如果需要精确复现：

```python
model = torch.compile(model, mode="reduce-overhead")
torch.use_deterministic_algorithms(True)
```

某些情况下 compile 和 deterministic 不兼容——需要测试。

### 10.4 混合精度的非确定性

BF16/FP16 训练的舍入误差比 FP32 大。完全复现需要：

```python
# 同精度 + 同硬件
scaler = torch.amp.GradScaler()  # 如果用 FP16
```

BF16 在 Ampere 和 Hopper GPU 上行为一致，但和 FP16 不同。

### 10.5 数据增强的随机性

```python
# ❌ 用了随机数据增强但没固定种子
transform = transforms.RandomHorizontalFlip(p=0.5)

# ✅ 确保全局种子已设置（影响 PyTorch 的随机增强）
set_seed(42)
transform = transforms.RandomHorizontalFlip(p=0.5)
```

### 10.6 "可复现"不等于"确定性"

```
可复现（Reproducibility）：
  → 同代码 + 同配置 + 同环境 → 同结果（合理范围内）

确定性（Determinism）：
  → bit-level 完全一致

大部分场景你需要的是"可复现"而不是"确定性"。
确定性模式会降低 10-30% 速度，日常训练不建议开。
仅在 debug 或需要精确对比时开启。
```

### 10.7 不记录环境信息

```python
# ❌ 只记录超参
wandb.init(config={"lr": 1e-4, "bs": 32})

# ✅ 同时记录环境
wandb.init(config={
    "lr": 1e-4, "bs": 32,
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "git_hash": get_git_hash(),
    "seed": 42,
})
```

---

## 11. 实用模板

### 11.1 可复现训练脚本骨架

```python
import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_git_hash() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip()
    except Exception:
        return "unknown"


def main():
    seed = 42
    set_seed(seed)

    wandb.init(
        project="my-project",
        config={
            "seed": seed,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu",
            "git": get_git_hash(),
        },
    )

    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        worker_init_fn=lambda wid: np.random.seed(seed + wid),
        generator=g,
    )

    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):
        for step, batch in enumerate(loader):
            loss = train_step(model, batch, optimizer)
            if step % 100 == 0:
                wandb.log({"train/loss": loss.item()})

        val_loss = evaluate(model, val_loader)
        wandb.log({"val/loss": val_loss, "epoch": epoch})

        save_checkpoint(
            f"ckpt/epoch_{epoch}.pt", model, optimizer,
            scheduler=None, epoch=epoch,
            global_step=epoch * len(loader),
            best_val_loss=val_loss,
        )

    wandb.finish()


if __name__ == "__main__":
    main()
```

---

## 小结

| 层次 | 做什么 | 工具 |
|---|---|---|
| 随机性 | 设全套 seed | `set_seed()` |
| 确定性 | 开 deterministic mode（debug 时） | `torch.use_deterministic_algorithms` |
| 环境 | 锁版本 | `uv.lock` / Docker |
| 代码 | 记 git hash | `subprocess` + wandb |
| 数据 | 记 checksum / 用 DVC | `hashlib` / `dvc` |
| 训练状态 | 保存完整 checkpoint | `torch.save` (含 RNG states) |
| 报告 | 多 seed 取平均 | 实验追踪工具 |

**一条黄金规则**：**如果一个实验结果你无法复现，那它就不存在。** 可复现性不是锦上添花——它是底线。
