# 03 · 超参数搜索

> 目标：用 **Optuna** 和 **Ray Tune** 系统化地找到最优超参，告别"手动试 lr 试一周"的日子。
> 重点掌握 Optuna——它是 2026 年 Python 生态里超参搜索的事实标准。

---

## 1. 为什么需要超参搜索？

### 1.1 手动调参的痛苦

```
"lr=1e-4 不行，试试 1e-3"
"1e-3 炸了，3e-4 呢？"
"3e-4 看着可以，但 batch_size 是不是也要调？"
"lr 和 batch_size 有交互作用……"
→ 排列组合爆炸，两周过去了
```

### 1.2 搜索策略概览

| 策略 | 原理 | 效率 | 适用场景 |
|---|---|---|---|
| **Grid Search** | 穷举所有组合 | 低 | 参数少（≤3 个），离散值 |
| **Random Search** | 随机采样 | 中 | 任何场景的基线 |
| **Bayesian (TPE)** | 用概率模型引导采样 | 高 | 连续参数、预算有限 |
| **Evolutionary** | 遗传算法 | 中-高 | 大搜索空间 |
| **Hyperband / ASHA** | 早停 + 资源分配 | 极高 | 训练成本高时 |
| **PBT** | 群体训练 + 动态调参 | 高 | 长训练周期 |

**经验法则**：
- 参数 ≤ 3 个 → Random Search 就够了
- 参数 > 3 个 → **Bayesian (TPE) + 剪枝** 是最优选择
- 训练一次要几小时 → **一定要用剪枝**

---

## 2. Optuna：Python 超参搜索的王者

### 2.1 核心概念

```
Study       ← 一次超参搜索任务
├── Trial 0 ← 一组超参的尝试
├── Trial 1
├── Trial 2
└── ...

每个 Trial：
1. 采样超参（suggest_float / suggest_int / suggest_categorical）
2. 训练模型
3. 报告指标（return value 或 report + 中间剪枝）
```

### 2.2 最小示例

```bash
pip install optuna
```

```python
import optuna


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    num_layers = trial.suggest_int("num_layers", 2, 8)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])

    model = build_model(num_layers=num_layers, dropout=dropout)
    val_loss = train_and_evaluate(model, lr=lr, optimizer=optimizer_name)
    return val_loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print(f"Best trial: {study.best_trial.params}")
print(f"Best value: {study.best_value:.4f}")
```

### 2.3 Suggest API 详解

```python
def objective(trial: optuna.Trial) -> float:
    # 连续值（线性）
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    # 连续值（对数均匀）——学习率必须用 log=True
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    # 连续值（步进）
    weight_decay = trial.suggest_float("wd", 0.0, 0.1, step=0.01)

    # 整数
    num_layers = trial.suggest_int("num_layers", 1, 12)

    # 整数（步进）
    batch_size = trial.suggest_int("batch_size", 16, 128, step=16)

    # 分类
    activation = trial.suggest_categorical("act", ["relu", "gelu", "silu"])

    ...
```

**关键**：`log=True` 对学习率、权重衰减等跨越多个数量级的参数非常重要。不加 `log=True`，采样会集中在大值区域。

### 2.4 剪枝（Pruning）：提前终止差 trial

剪枝是 Optuna 的杀手锏：**训练到一半就能判断这组参数不行，直接跳过。**

```python
def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(20):
        train_one_epoch(model, optimizer)
        val_loss = evaluate(model)

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss


study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,     # 前 5 个 trial 不剪枝（收集基线）
        n_warmup_steps=3,       # 每个 trial 前 3 个 epoch 不剪枝
    ),
)
study.optimize(objective, n_trials=100)
```

**MedianPruner 原理**：如果当前 trial 在第 N 步的指标**差于所有已完成 trial 在第 N 步的中位数**，就剪掉。

### 2.5 Hyperband Pruner（推荐）

Hyperband 比 MedianPruner 更激进、更高效：

```python
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1,       # 最少跑 1 个 epoch
        max_resource=20,      # 最多跑 20 个 epoch
        reduction_factor=3,   # 每轮淘汰 2/3
    ),
)
```

### 2.6 可视化

```python
import optuna.visualization as vis

fig = vis.plot_optimization_history(study)
fig.show()

fig = vis.plot_param_importances(study)
fig.show()

fig = vis.plot_parallel_coordinate(study)
fig.show()

fig = vis.plot_contour(study, params=["lr", "dropout"])
fig.show()

fig = vis.plot_slice(study)
fig.show()
```

`plot_param_importances` 特别有用——告诉你**哪个参数对结果影响最大**，帮你收窄搜索空间。

### 2.7 持久化存储

```python
study = optuna.create_study(
    study_name="transformer-hpo",
    storage="sqlite:///optuna.db",    # SQLite 本地存储
    direction="minimize",
    load_if_exists=True,              # 重启时继续
)

study = optuna.create_study(
    storage="mysql://user:pass@host/db",  # MySQL 共享存储
    ...
)
```

### 2.8 分布式搜索

只需让多个 worker 指向同一个 storage：

```bash
# 终端 1
python search.py   # 自动创建 study

# 终端 2（甚至另一台机器）
python search.py   # load_if_exists=True，自动加入

# 终端 3
python search.py   # 三个 worker 并行搜索
```

Optuna 的 storage 层会自动协调——这就是分布式超参搜索，**零额外代码**。

---

## 3. Ray Tune：大规模超参搜索

Ray Tune 更适合**集群级**搜索场景（几十到几百个 GPU）。

### 3.1 基本用法

```bash
pip install "ray[tune]"
```

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler


def train_fn(config: dict) -> None:
    model = build_model(
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    for epoch in range(20):
        train_one_epoch(model, optimizer)
        val_loss = evaluate(model)
        tune.report({"val_loss": val_loss})


search_space = {
    "lr": tune.loguniform(1e-5, 1e-2),
    "dropout": tune.uniform(0.0, 0.5),
    "num_layers": tune.choice([4, 6, 8, 12]),
}

scheduler = ASHAScheduler(
    max_t=20,
    grace_period=3,
    reduction_factor=2,
)

results = tune.run(
    train_fn,
    config=search_space,
    num_samples=50,
    scheduler=scheduler,
    metric="val_loss",
    mode="min",
    resources_per_trial={"cpu": 4, "gpu": 1},
)

print(results.best_config)
```

### 3.2 ASHA vs PBT

| 调度器 | 原理 | 适用场景 |
|---|---|---|
| **ASHA** | 异步 Hyperband，早停差 trial | 通用，大部分场景首选 |
| **PBT** | 群体训练，动态调参 + 权重继承 | 需要在训练中修改 lr schedule |
| **BOHB** | Bayesian + Hyperband | 需要更智能的采样 |

### 3.3 与 Optuna 对比

| 特性 | Optuna | Ray Tune |
|---|---|---|
| 安装 | 轻量（~5 MB） | 较重（~200 MB） |
| 学习成本 | ★★☆ | ★★★ |
| 单机使用 | 极佳 | 可以但有点重 |
| 集群分布式 | 需共享 DB | 原生支持 |
| 调度策略 | 剪枝（pruning） | Scheduler（ASHA/PBT） |
| 可视化 | 内置 | TensorBoard 集成 |
| 与 PyTorch 集成 | 原生 | 原生 |

**推荐**：
- 单机 / 小规模 → **Optuna**
- 集群 / 大规模 / 已在用 Ray → **Ray Tune**

---

## 4. Hydra + Optuna 集成

Hydra 有官方 Optuna sweeper 插件：

```bash
pip install hydra-optuna-sweeper
```

```yaml
# config.yaml
defaults:
  - override hydra/sweeper: optuna

model:
  num_layers: 6
  dropout: 0.1

training:
  lr: 1e-4
  epochs: 10

hydra:
  sweeper:
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
    direction: minimize
    n_trials: 30
    params:
      training.lr:
        type: float
        low: 1e-5
        high: 1e-2
        log: true
      model.dropout:
        type: float
        low: 0.0
        high: 0.5
      model.num_layers:
        type: int
        low: 2
        high: 12
```

```bash
python train.py --multirun
```

这样你的训练脚本**不需要任何修改**——Hydra + Optuna 插件会自动采样超参、调用脚本、收集结果。

---

## 5. W&B Sweeps

如果你已经在用 W&B，它的 Sweeps 功能可以直接做超参搜索：

```python
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/loss", "goal": "minimize"},
    "parameters": {
        "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
        "batch_size": {"values": [16, 32, 64]},
        "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.5},
    },
}

sweep_id = wandb.sweep(sweep_config, project="my-project")


def train():
    wandb.init()
    cfg = wandb.config

    model = build_model(dropout=cfg.dropout)
    for epoch in range(10):
        loss = train_epoch(model, lr=cfg.lr, batch_size=cfg.batch_size)
        wandb.log({"val/loss": loss})


wandb.agent(sweep_id, function=train, count=30)
```

**好处**：搜索过程直接在 W&B 面板可视化，无需额外工具。

---

## 6. 搜索空间设计实战

### 6.1 学习率

```python
lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
```

**永远用 `log=True`**。lr 的好值通常在 1e-5 到 1e-3 之间，线性采样会严重偏向大值。

### 6.2 Batch Size

```python
batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
```

Batch size 通常取 2 的幂次（GPU 优化友好）。

### 6.3 模型架构

```python
num_layers = trial.suggest_int("num_layers", 4, 12)
hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768, 1024])
num_heads = trial.suggest_categorical("num_heads", [4, 8, 12, 16])
```

注意约束：`hidden_dim % num_heads == 0`。用条件搜索：

```python
hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768])
if hidden_dim == 256:
    num_heads = trial.suggest_categorical("num_heads_256", [4, 8])
elif hidden_dim == 512:
    num_heads = trial.suggest_categorical("num_heads_512", [4, 8, 16])
else:
    num_heads = trial.suggest_categorical("num_heads_768", [4, 8, 12])
```

### 6.4 优化器相关

```python
optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
weight_decay = trial.suggest_float("wd", 1e-6, 0.1, log=True)

if optimizer == "sgd":
    momentum = trial.suggest_float("momentum", 0.8, 0.99)
```

### 6.5 预算分配

```
总预算 = 100 GPU·hours

方案 A（不推荐）：
  50 trials × 2 hours/trial（完整训练）

方案 B（推荐）：
  200 trials × 0.5 hours/trial（短训练 + 剪枝）
  → 实际只有 ~30 trials 跑完，其余被剪掉

方案 C（更推荐）：
  第 1 轮：100 trials × 10 min（粗筛，1/5 数据 + 1/5 epochs）
  第 2 轮：前 10 名 × 2 hours（精调，完整数据）
```

---

## 7. 完整示例：用 Optuna 调 Transformer

```python
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("wd", 1e-6, 0.1, log=True)
    num_layers = trial.suggest_int("num_layers", 2, 8)
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    hidden_dim = num_heads * trial.suggest_int("head_dim", 32, 128, step=16)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_transformer(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            loss = train_step(model, batch, optimizer, device)

        model.eval()
        val_loss = evaluate(model, val_loader, device)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss


study = optuna.create_study(
    study_name="transformer-hpo",
    storage="sqlite:///hpo.db",
    direction="minimize",
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource=10, reduction_factor=3,
    ),
    load_if_exists=True,
)

study.optimize(objective, n_trials=100, timeout=3600 * 4)

print("=" * 60)
print(f"Best trial #{study.best_trial.number}")
print(f"  Value: {study.best_value:.4f}")
print(f"  Params: {study.best_trial.params}")

fig = optuna.visualization.plot_param_importances(study)
fig.write_html("param_importances.html")
```

---

## 8. 常见坑

### 8.1 搜索空间太大

```python
# ❌ 7 个参数 × 宽范围 = 搜索空间天文数字
lr = trial.suggest_float("lr", 1e-10, 1.0, log=True)    # 太宽
num_layers = trial.suggest_int("layers", 1, 100)         # 太宽
```

**先做文献调研**，确定合理范围。比如 Transformer 的 lr 一般在 1e-5 ~ 5e-3。

### 8.2 忘记 `log=True`

```python
# ❌ 线性采样 lr：99% 的采样点在 0.001~0.01，几乎不会试 1e-5
lr = trial.suggest_float("lr", 1e-5, 1e-2)

# ✅
lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
```

### 8.3 没用剪枝

```python
# ❌ 每个 trial 完整跑 50 epochs
study.optimize(objective, n_trials=100)  # 100 × 50 epochs = 5000 epochs 🤯

# ✅ 用剪枝，大部分 trial 在 5-10 epochs 就被砍掉
```

### 8.4 报告指标和优化方向不一致

```python
# ❌ direction="minimize" 但 return accuracy（应该 maximize）
study = optuna.create_study(direction="minimize")
def objective(trial):
    ...
    return accuracy  # 越大越好，但你说了 minimize！
```

### 8.5 Trial 之间有状态泄漏

```python
# ❌ 全局模型，上一个 trial 的权重影响下一个
model = build_model()

def objective(trial):
    # model 还是上一次的权重！
    optimizer = ...
    train(model, optimizer)
```

**每个 trial 必须独立**——在 objective 函数内部创建全新的模型。

### 8.6 忽略 trial 之间的方差

同样的超参、不同的随机种子，结果可能差 5-10%。

```python
def objective(trial):
    results = []
    for seed in [42, 123, 456]:
        set_seed(seed)
        val_loss = train_and_eval(trial_params, seed)
        results.append(val_loss)
    return sum(results) / len(results)  # 取平均
```

### 8.7 超参搜索后不做最终训练

搜索时通常用小数据集 / 少 epochs。找到最优超参后，**必须用完整数据和足够 epochs 做最终训练**，并跑多个 seed 确认结果。

---

## 小结

| 场景 | 推荐 |
|---|---|
| 入门 / 参数少 | Random Search |
| 单机系统化搜索 | **Optuna + HyperbandPruner** |
| 集群大规模搜索 | **Ray Tune + ASHA** |
| 已用 Hydra | Hydra Optuna Sweeper 插件 |
| 已用 W&B | W&B Sweeps |

**一条黄金规则**：**先缩小搜索空间，再增加 trial 数量**。宁可在合理范围内搜 200 次，不要在天文数字范围内搜 20 次。

下一节学习可复现性——确保你找到的"最优参数"别人也能跑出同样的结果。
