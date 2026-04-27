# 05 · 工程化最佳实践：测试 / 调试 / Profiling / 项目结构

> 科研代码写完就扔；工程代码要**别人能跑、能改、能上线**。
> 这一节是很多算法同学的盲区——但它决定了你的代码是"可用品"还是"艺术品"。

---

## 1. 项目结构：`src` layout + `pyproject.toml`

### 1.1 推荐的现代 Python 项目骨架

```
my-aigc-project/
├── pyproject.toml              # 依赖 + 工具配置（现代标准）
├── README.md
├── .gitignore
├── .python-version             # 锁定 Python 版本（供 uv/pyenv 使用）
├── src/
│   └── my_project/             # 实际的 Python 包
│       ├── __init__.py
│       ├── models/
│       ├── data/
│       ├── training/
│       └── inference/
├── tests/                      # 测试代码，与 src 同级
│   ├── test_models.py
│   └── test_data.py
├── scripts/                    # 训练/推理入口脚本
│   ├── train.py
│   └── infer.py
├── configs/                    # 配置文件
│   └── default.yaml
├── notebooks/                  # 探索性 notebook
└── experiments/                # 实验输出（gitignore）
```

**为什么用 `src/` layout？**
避免"在项目根目录误 import 未安装的包"——强制你先 `pip install -e .` 才能用，这是工业界标准。

### 1.2 最小可用的 `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "my-aigc-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.4",
    "transformers>=4.44",
    "diffusers>=0.30",
    "pydantic>=2.6",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.6",
    "mypy>=1.10",
    "pre-commit>=3.8",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "PL"]
ignore = ["PLR0913"]  # too many args - 算法函数常需要

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.mypy]
python_version = "3.10"
strict = false
ignore_missing_imports = true
```

### 1.3 用 `uv` 管理环境（2026 推荐）

```bash
pip install uv

cd my-aigc-project
uv venv
source .venv/bin/activate

uv pip install -e ".[dev]"

uv pip sync requirements.lock.txt
```

`uv` 比 `pip` 快 10×，2026 年已基本成为事实标准。

---

## 2. 测试：从 0 到 pytest

### 2.1 为什么算法代码也要测？

**AIGC 代码最脆弱的地方**是"shape 错位 / 数值微小漂移"——肉眼看不出，一跑就崩。
对这种问题，单元测试是成本最低的保险。

### 2.2 最小测试样例

```python
import pytest
import torch

from my_project.models.attention import SelfAttention


@pytest.fixture
def attn_module():
    return SelfAttention(d_model=64, n_heads=4)


def test_attention_shape(attn_module):
    x = torch.randn(2, 16, 64)
    y = attn_module(x)
    assert y.shape == x.shape


def test_attention_causal_mask(attn_module):
    torch.manual_seed(0)
    x = torch.randn(1, 8, 64)

    x_mod = x.clone()
    x_mod[0, -1] += 100

    y = attn_module(x)
    y_mod = attn_module(x_mod)

    torch.testing.assert_close(y[:, :-1], y_mod[:, :-1])


@pytest.mark.parametrize("batch,seq,dim", [(1, 8, 64), (4, 32, 128), (2, 1, 64)])
def test_attention_various_shapes(batch, seq, dim, request):
    attn = SelfAttention(d_model=dim, n_heads=4)
    x = torch.randn(batch, seq, dim)
    y = attn(x)
    assert y.shape == (batch, seq, dim)
```

运行：

```bash
pytest                                # 跑全部
pytest tests/test_models.py::test_attention_shape  # 单个
pytest -k "attention" --tb=long       # 关键字过滤
pytest --cov=src --cov-report=term    # 覆盖率
```

### 2.3 必学的 pytest 套路

| 特性 | 用途 |
|---|---|
| `@pytest.fixture` | 复用测试前置（模型/数据/tmp dir） |
| `@pytest.mark.parametrize` | 一组参数跑多个用例 |
| `tmp_path` 内置 fixture | 临时目录，自动清理 |
| `monkeypatch` 内置 fixture | 打桩替换函数/环境变量 |
| `@pytest.mark.slow` | 自定义标记，跳过慢测试 |
| `pytest -x` | 首次失败立即停止 |
| `pytest --pdb` | 失败时进入调试器 |

### 2.4 算法代码的测试策略

- **Shape 测试**：所有 `nn.Module`，至少测 forward 输出形状。
- **数值测试**：对固定 seed 的输入，输出应接近参考值（`torch.testing.assert_close(rtol=1e-4)`）。
- **等价性测试**：优化实现 vs 朴素实现结果一致（比如 FlashAttention vs 朴素 attention）。
- **梯度测试**：`torch.autograd.gradcheck` 能检查 autograd 正确性。
- **集成测试**：1 个 batch 过完整流程跑通，断言 loss 不是 NaN。

---

## 3. 调试：`breakpoint()` + `rich` + IDE

### 3.1 `breakpoint()`：比 print 强 10 倍

```python
def train_step(model, batch):
    out = model(batch["x"])
    loss = F.cross_entropy(out, batch["y"])
    if loss.isnan():
        breakpoint()
    loss.backward()
```

在需要的地方插入 `breakpoint()`，代码运行到这里会进入交互式调试器：

| 命令 | 作用 |
|---|---|
| `p x` / `pp x` | 打印变量（pp 是格式化） |
| `n` | 下一行 |
| `s` | 进入函数 |
| `c` | 继续运行 |
| `u` / `d` | 上/下一帧 |
| `l` | 显示当前源码 |
| `x.shape` | 直接当 Python 交互 |

**环境变量控制**：

```bash
PYTHONBREAKPOINT=0 python train.py

PYTHONBREAKPOINT=ipdb.set_trace python train.py
```

### 3.2 `rich` / `loguru`：美化日志

```python
from rich.console import Console
from rich.traceback import install

install(show_locals=True)
console = Console()

console.print("[bold green]Epoch 1[/bold green] loss=0.12")
console.log({"lr": 1e-4, "step": 1000})
```

错误回溯会带上**局部变量值**，定位 shape mismatch 超方便。

### 3.3 IDE Debugger（VSCode / Cursor）

`.vscode/launch.json`：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Train",
      "type": "debugpy",
      "request": "launch",
      "program": "scripts/train.py",
      "args": ["--config", "configs/default.yaml"],
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

设断点，F5 启动；能逐行看 Tensor 值、调用栈、线程状态。

---

## 4. Profiling：找到真正的瓶颈

### 4.1 `cProfile`：纯 Python 耗时分布

```python
import cProfile, pstats

with cProfile.Profile() as pr:
    run_training(...)

stats = pstats.Stats(pr).sort_stats("cumulative")
stats.print_stats(30)
```

适合找 **"哪个 Python 函数花了最多时间"**。

### 4.2 `torch.profiler`：训练的"CT 扫描"

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        if step >= 5:
            break
        with record_function("forward"):
            out = model(batch)
        with record_function("backward"):
            out.loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
prof.export_chrome_trace("trace.json")
```

生成的 `trace.json` 可以用 `chrome://tracing` 或 Perfetto UI 打开，看到每个 CUDA kernel 的时间线——这是优化训练速度的标准方法。

### 4.3 显存峰值监控

```python
import torch

torch.cuda.reset_peak_memory_stats()
run(...)
peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"peak GPU mem: {peak:.2f} GB")
```

组合 `torch.cuda.memory._dump_snapshot()` 能导出可视化的显存分配时间线。

### 4.4 `line_profiler`：逐行耗时

```bash
pip install line_profiler
kernprof -l -v train.py
```

给函数加 `@profile` 装饰器，能看到每一行的耗时——定位 Python 瓶颈的利器。

### 4.5 何时用哪个？

| 瓶颈在哪 | 工具 |
|---|---|
| Python 代码慢 | `cProfile` / `line_profiler` |
| GPU 利用率低 | `torch.profiler` + `nvidia-smi` |
| 显存不够 | `torch.cuda.memory_*` + profiler |
| 数据加载慢 | `torch.profiler` 看 DataLoader 占比 |
| 底层 CUDA kernel | NVIDIA Nsight Systems / Nsight Compute |

---

## 5. 代码质量工具链：`ruff` + `mypy` + `pre-commit`

### 5.1 `ruff`：格式化 + linting

```bash
ruff check .                   # 检查问题
ruff check --fix .             # 自动修复
ruff format .                  # 格式化（代替 black）
```

`ruff` 在 2026 年已统一了 `black` + `isort` + `flake8` + `pyupgrade` 的功能，速度快 10–100×。

### 5.2 `mypy`：静态类型检查

```bash
mypy src/
```

建议渐进式启用：

```toml
[tool.mypy]
strict = false
disallow_untyped_defs = false
check_untyped_defs = true
```

等代码成熟后再打开 `strict = true`。

### 5.3 `pre-commit`：提交前自动跑

`.pre-commit-config.yaml`：

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ["--maxkb=500"]
```

安装：

```bash
pip install pre-commit
pre-commit install
```

之后每次 `git commit` 都会自动运行这些检查，代码库永远保持整洁。

---

## 6. 版本控制的算法工程师最佳实践

### 6.1 `.gitignore` 标配

```
__pycache__/
*.pyc
.venv/
.env

wandb/
outputs/
checkpoints/
runs/
*.pt
*.safetensors
*.bin
*.ckpt

data/

*.ipynb_checkpoints
.DS_Store
.vscode/
.idea/
```

**关键**：checkpoint/数据 不要进 git——用 `dvc` / `git-lfs` / 外部存储。

### 6.2 提交信息规范

推荐 Conventional Commits：

```
feat: add LoRA adapter to attention layers
fix: correct attention mask shape for batched generation
perf: enable flash attention in SDPA path
docs: update training README
test: add gradient shape tests for DiT block
refactor: extract data augmentation into Albumentations pipeline
```

---

## 7. 实验管理：可复现性

```python
import random, os
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**但要知道**：
- `cudnn.deterministic = True` 会让训练慢 10–30%，开发阶段值得开，大规模训练时可关。
- 分布式训练时还要 `torch.distributed` 的 seed 同步。
- 有些 CUDA 操作**天生就非确定**（如 `scatter_add`），完全复现不现实——目标是"相同硬件、相同环境下相近"。

---

## 8. 实战综合模板（推荐作为训练脚本起点）

```python
"""训练脚本模板：集成配置、日志、种子、checkpoint、优雅退出。"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Config:
    lr: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    seed: int = 42
    output_dir: Path = Path("./outputs")

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        data = yaml.safe_load(path.read_text())
        return cls(**data)


def setup_logging(log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=handlers,
    )


def handle_sigterm(_signum, _frame):
    logger.warning("received SIGTERM, saving checkpoint before exit...")
    sys.exit(0)


def main():
    signal.signal(signal.SIGTERM, handle_sigterm)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(cfg.output_dir / "train.log")
    logger.info("config: %s", cfg)

    torch.manual_seed(cfg.seed)

    try:
        run_training(cfg)
    except KeyboardInterrupt:
        logger.warning("interrupted by user")
    except Exception:
        logger.exception("training crashed")
        raise


if __name__ == "__main__":
    main()
```

---

## 小结：工程师 checklist

开启一个新 AIGC 项目时，完成以下清单：

- [ ] `pyproject.toml`（不要再用 `setup.py`）
- [ ] `src/` layout
- [ ] `.gitignore` 排除 checkpoint / data / cache
- [ ] `ruff` + `mypy` 配置进 `pyproject.toml`
- [ ] `pre-commit` 钩子
- [ ] `tests/` 目录 + 至少 1 个 shape 测试
- [ ] 训练脚本包含：seed / logging / checkpoint / 信号处理
- [ ] 实验追踪：tensorboard / wandb
- [ ] README：安装、运行、数据格式、已知问题

**每一条都是从血泪教训里总结出来的**。前期多花半小时，后期省几十小时。

至此 `01-python-foundations` 模块补齐工程化实践。建议写一遍这个脚手架（自己从零搭建一个项目），你会瞬间理解什么是"工程级 Python"。
