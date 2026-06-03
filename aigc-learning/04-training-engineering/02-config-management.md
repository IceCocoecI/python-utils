# 02 · 配置管理

> 目标：用 **OmegaConf + Hydra** 取代 argparse 和硬编码超参数。
> 让你的训练脚本支持"一份代码，多种配置"——从命令行切换模型、数据、优化器，不改一行 Python。

---

## 1. 问题：你的配置管理可能是这样的

### 1.1 硬编码阶段

```python
# ❌ 改一个参数要改代码
lr = 1e-4
batch_size = 32
hidden_dim = 768
num_layers = 12
```

### 1.2 argparse 阶段

```python
# ❌ 50 个参数的 argparse = 灾难
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--hidden_dim", type=int, default=768)
parser.add_argument("--num_layers", type=int, default=12)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--warmup_steps", type=int, default=1000)
parser.add_argument("--weight_decay", type=float, default=0.01)
# ... 还有 40 个 ...
```

argparse 的问题：
- **无层次**：50 个参数铺成一层
- **无类型检查**：`--lr "hello"` 运行时才报错
- **无法组合**：不同模型需要不同参数集
- **无法复用**：配置不能保存为文件
- **无法嵌套**：model 和 optimizer 的参数混在一起

### 1.3 目标：YAML 配置 + 命令行覆盖

```yaml
# config.yaml
model:
  name: transformer
  hidden_dim: 768
  num_layers: 12
  dropout: 0.1

training:
  lr: 1e-4
  batch_size: 32
  epochs: 10
  warmup_steps: 1000

data:
  dataset: wikitext
  max_length: 512
```

```bash
# 从命令行覆盖任意参数
python train.py training.lr=3e-5 model.num_layers=6
```

这就是 **Hydra + OmegaConf** 的威力。

---

## 2. OmegaConf：配置数据结构

OmegaConf 是 Hydra 的底层配置库。可以单独使用，也可以配合 Hydra。

### 2.1 基础用法

```bash
pip install omegaconf
```

```python
from omegaconf import OmegaConf, DictConfig

cfg = OmegaConf.create({
    "model": {"name": "gpt2", "layers": 12},
    "lr": 1e-4,
})

print(cfg.model.name)        # "gpt2"（dot access）
print(cfg["lr"])              # 1e-4（dict access）
print(cfg.model.layers)      # 12

OmegaConf.to_yaml(cfg)       # 转成 YAML 字符串
OmegaConf.to_container(cfg)  # 转回普通 dict
```

### 2.2 从 YAML 文件加载

```python
cfg = OmegaConf.load("config.yaml")
print(cfg.model.hidden_dim)
```

### 2.3 合并配置（merge）

```python
defaults = OmegaConf.create({"lr": 1e-4, "bs": 32, "wd": 0.01})
overrides = OmegaConf.create({"lr": 3e-5, "bs": 64})

cfg = OmegaConf.merge(defaults, overrides)
print(cfg.lr)  # 3e-5（后者覆盖前者）
```

**核心思想**：默认配置 + 覆盖配置 → 最终配置。

### 2.4 插值（interpolation）

OmegaConf 最强大的功能之一：配置值可以引用其他配置值。

```yaml
# config.yaml
model:
  name: gpt2
  hidden_dim: 768
  ffn_dim: ${multiply:${model.hidden_dim},4}  # 需要自定义 resolver

training:
  output_dir: outputs/${model.name}/${now:%Y-%m-%d}  # 引用 model.name
  log_dir: ${training.output_dir}/logs               # 引用自身
```

```python
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("multiply", lambda x, y: int(x) * int(y))

cfg = OmegaConf.load("config.yaml")
print(cfg.training.output_dir)  # "outputs/gpt2/2026-06-03"
```

内置 resolver：
- `${oc.env:HOME}` — 读环境变量
- `${oc.env:VAR,default}` — 读环境变量，带默认值

### 2.5 设为只读

```python
OmegaConf.set_readonly(cfg, True)
cfg.lr = 0.1  # raises ReadonlyViolationError
```

训练脚本解析完配置后，建议 **立即冻结**——防止后续代码意外修改。

### 2.6 缺失值与必填字段

```yaml
model:
  name: ???           # MISSING：必须在运行时提供
  layers: 12
```

```python
cfg = OmegaConf.load("config.yaml")
print(cfg.model.name)  # raises MissingMandatoryValue
```

`???` 相当于"这个参数没有默认值，你必须显式指定"。

---

## 3. Hydra：配置驱动的应用框架

Hydra 基于 OmegaConf，在它之上提供了**命令行覆盖、配置组合、多运行、插件**等工程能力。

### 3.1 最小示例

```bash
pip install hydra-core
```

```
my_project/
├── config.yaml
└── train.py
```

```yaml
# config.yaml
lr: 1e-4
batch_size: 32
epochs: 10
model_name: resnet50
```

```python
# train.py
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    print(f"lr={cfg.lr}, bs={cfg.batch_size}, model={cfg.model_name}")
    # ... 训练逻辑 ...


if __name__ == "__main__":
    main()
```

```bash
python train.py                           # 使用默认配置
python train.py lr=3e-5 batch_size=64     # 命令行覆盖
python train.py model_name=vit_base       # 换模型
```

### 3.2 自动输出目录

Hydra 会自动创建带时间戳的输出目录：

```
outputs/
└── 2026-06-03/
    └── 14-30-25/
        ├── .hydra/
        │   ├── config.yaml      ← 最终合并后的完整配置
        │   ├── hydra.yaml       ← Hydra 自身配置
        │   └── overrides.yaml   ← 你从命令行传入的覆盖
        └── train.log
```

**这意味着每次 run 的完整配置都被自动保存了**——再也不会丢失。

### 3.3 Config Groups：组合式配置

这是 Hydra 的灵魂功能。假设你有多种模型和多种优化器：

```
my_project/
├── conf/
│   ├── config.yaml           # 主配置
│   ├── model/
│   │   ├── small.yaml
│   │   ├── base.yaml
│   │   └── large.yaml
│   └── optimizer/
│       ├── adam.yaml
│       └── sgd.yaml
└── train.py
```

```yaml
# conf/config.yaml
defaults:
  - model: base             # 默认使用 model/base.yaml
  - optimizer: adam          # 默认使用 optimizer/adam.yaml
  - _self_                   # 本文件的配置优先级最低

training:
  epochs: 10
  seed: 42
```

```yaml
# conf/model/small.yaml
name: transformer-small
hidden_dim: 256
num_layers: 6
num_heads: 4
```

```yaml
# conf/model/base.yaml
name: transformer-base
hidden_dim: 768
num_layers: 12
num_heads: 12
```

```yaml
# conf/model/large.yaml
name: transformer-large
hidden_dim: 1024
num_layers: 24
num_heads: 16
```

```yaml
# conf/optimizer/adam.yaml
name: adam
lr: 1e-4
weight_decay: 0.01
betas: [0.9, 0.999]
```

```yaml
# conf/optimizer/sgd.yaml
name: sgd
lr: 0.01
momentum: 0.9
weight_decay: 1e-4
```

```bash
# 用默认配置（base + adam）
python train.py

# 换成 large 模型 + sgd 优化器
python train.py model=large optimizer=sgd

# 换模型的同时覆盖 lr
python train.py model=small optimizer.lr=3e-5
```

### 3.4 Multirun：一条命令跑多组实验

```bash
python train.py --multirun model=small,base,large

python train.py --multirun optimizer.lr=1e-3,1e-4,1e-5 training.seed=42,123,456
```

`--multirun` 会自动做笛卡尔积，上面的命令会跑 3 × 3 = 9 组实验。

### 3.5 Hydra 的工作目录

Hydra 默认会 `os.chdir()` 到输出目录。这可能导致相对路径出错。

```python
import hydra
from pathlib import Path

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    original_cwd = hydra.utils.get_original_cwd()
    data_path = Path(original_cwd) / "data" / "train.jsonl"
```

或者在配置里禁用自动切换：

```yaml
# config.yaml
hydra:
  run:
    dir: .                  # 不切换目录
```

---

## 4. Structured Configs：用 dataclass 定义配置 schema

```python
from dataclasses import dataclass, field
from omegaconf import MISSING
import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    name: str = MISSING
    hidden_dim: int = 768
    num_layers: int = 12
    dropout: float = 0.1


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01


@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    epochs: int = 10
    seed: int = 42


cs = ConfigStore.instance()
cs.store(name="config", node=TrainingConfig)


@hydra.main(version_base=None, config_name="config")
def main(cfg: TrainingConfig) -> None:
    print(cfg.model.hidden_dim)  # 有类型提示！IDE 补全！
```

**好处**：
- IDE 自动补全和类型检查
- 运行时自动验证（传错类型会报错）
- 代码即文档

---

## 5. 实战：完整的 Hydra 训练脚本

```yaml
# conf/config.yaml
defaults:
  - model: base
  - optimizer: adam
  - _self_

training:
  epochs: 10
  batch_size: 32
  seed: 42
  log_every: 100
  save_every: 1000

data:
  name: wikitext
  max_length: 512
  num_workers: 4

wandb:
  project: my-llm
  enabled: true
```

```python
# train.py
import torch
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.training.seed)

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    model = build_model(cfg.model)
    optimizer = build_optimizer(model, cfg.optimizer)
    loader = build_dataloader(cfg.data, cfg.training.batch_size)

    for epoch in range(cfg.training.epochs):
        for step, batch in enumerate(loader):
            loss = train_step(model, batch, optimizer)
            if step % cfg.training.log_every == 0:
                print(f"epoch={epoch} step={step} loss={loss:.4f}")

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
```

```bash
python train.py model=large optimizer.lr=3e-5 training.epochs=20
```

---

## 6. 与其他方案对比

| 方案 | 层次化 | 类型检查 | CLI 覆盖 | 组合配置 | 学习成本 |
|---|---|---|---|---|---|
| 硬编码 | ❌ | ❌ | ❌ | ❌ | ★☆☆ |
| `argparse` | ❌ | 弱 | ✅ | ❌ | ★☆☆ |
| 纯 YAML + `yaml.load` | ✅ | ❌ | ❌ | ❌ | ★☆☆ |
| `ml_collections` | ✅ | ✅ | ✅ | ❌ | ★★☆ |
| **OmegaConf** | ✅ | ✅ | ❌（需自写） | ❌ | ★★☆ |
| **Hydra** | ✅ | ✅ | ✅ | ✅ | ★★★ |
| `jsonargparse` + Lightning | ✅ | ✅ | ✅ | 部分 | ★★☆ |

**推荐路径**：
- 小脚本 / 快速原型：`argparse` 够用
- 正式项目（5+ 超参）：**Hydra + OmegaConf**
- Lightning 用户：`jsonargparse`（Lightning CLI 内置）

---

## 7. AIGC 项目的配置组织模式

大型 AIGC 项目的典型配置结构：

```
conf/
├── config.yaml                # 入口：声明 defaults
├── model/
│   ├── llama_7b.yaml          # 模型架构配置
│   ├── llama_13b.yaml
│   └── qwen_7b.yaml
├── data/
│   ├── pretrain.yaml          # 预训练数据配置
│   ├── sft.yaml               # SFT 数据配置
│   └── dpo.yaml               # DPO 数据配置
├── optimizer/
│   ├── adamw.yaml
│   └── adafactor.yaml
├── scheduler/
│   ├── cosine.yaml
│   └── linear.yaml
└── experiment/
    ├── pretrain_7b.yaml       # 组合配置：model=llama_7b + data=pretrain
    └── sft_7b.yaml            # 组合配置：model=llama_7b + data=sft
```

```yaml
# conf/experiment/sft_7b.yaml
# @package _global_
defaults:
  - override /model: llama_7b
  - override /data: sft
  - override /optimizer: adamw
  - override /scheduler: cosine

training:
  epochs: 3
  batch_size: 8
  gradient_accumulation_steps: 4
```

```bash
python train.py +experiment=sft_7b   # 一键加载完整实验配置
```

---

## 8. 常见坑

### 8.1 忘记 `_self_`

```yaml
# config.yaml
defaults:
  - model: base
  # 没有 _self_ → model 配置会覆盖本文件的同名字段
```

`_self_` 控制本文件的优先级。**放在 defaults 最后**表示"本文件的值优先级最低"（最常见用法）。

### 8.2 Hydra 改了工作目录导致路径出错

```python
# ❌ 相对路径找不到文件
data = load("data/train.jsonl")

# ✅ 用 Hydra 的原始路径
data = load(Path(hydra.utils.get_original_cwd()) / "data/train.jsonl")

# ✅ 或者在配置里用绝对路径
# data.path: /home/user/project/data/train.jsonl
```

### 8.3 OmegaConf 的 DictConfig 不是 dict

```python
# ❌ 有些库需要普通 dict
some_api(cfg)  # TypeError

# ✅ 转换
some_api(OmegaConf.to_container(cfg, resolve=True))
```

### 8.4 multirun 的笛卡尔积爆炸

```bash
# 3 个参数各 5 个值 = 125 组实验
python train.py --multirun lr=1e-3,1e-4,1e-5,3e-4,3e-5 \
    bs=8,16,32,64,128 dropout=0.0,0.1,0.2,0.3,0.5
```

除非你有 125 块 GPU，否则用 Optuna 做采样搜索（下一节）。

### 8.5 YAML 的类型陷阱

```yaml
# YAML 会把这些解析成布尔值！
use_amp: yes     # True
use_amp: no      # False
use_amp: on      # True
use_amp: off     # False

# 安全做法：用 true/false
use_amp: true
use_amp: false
```

```yaml
# YAML 会把这些解析成浮点数
version: 3.10    # 3.1（不是 "3.10"！）

# 安全做法：加引号
version: "3.10"
```

### 8.6 Structured config 和 YAML 配置冲突

Structured config 定义了 schema，YAML 不能引入 schema 里没有的字段。
如果需要灵活扩展，在 dataclass 里加 `extra: dict = field(default_factory=dict)`。

---

## 小结

| 需求 | 推荐 |
|---|---|
| 5 个以内超参 | `argparse` |
| 层次化配置 + 命令行覆盖 | **Hydra** |
| 多种模型/数据集切换 | Hydra **Config Groups** |
| 类型安全 | Hydra **Structured Configs** |
| 配置引用/插值 | **OmegaConf** 插值 |
| 多组实验批量跑 | Hydra `--multirun` |

**一条黄金规则**：**配置和代码分离**——不要在 Python 文件里硬编码任何超参数。所有影响实验结果的值都应该在配置文件或命令行中指定。

下一节学习超参数搜索——让机器帮你找最优的超参组合。
