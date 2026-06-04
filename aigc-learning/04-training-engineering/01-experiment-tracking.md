# 01 · 实验追踪

> 目标：让每一次训练都变成**可查询、可对比、可回溯**的记录。
> 你不需要三个工具都精通——选一个主力（推荐 W&B），其余知道怎么用就行。

---

## 1. 为什么需要实验追踪？

没有追踪工具时，你的"实验管理"大概是这样的：

```
experiments/
├── run_v1/
├── run_v2_lr_higher/
├── run_v3_fix_bug/
├── run_v3_fix_bug_real/
├── run_final/
├── run_final_final/
└── run_final_final_v2_BEST/   ← 到底哪个是 best？
```

**实验追踪工具**帮你自动记录：
- 超参数（learning rate、batch size、模型配置……）
- 指标曲线（loss、accuracy、FID……）
- 系统信息（GPU 利用率、内存）
- 产出物（checkpoint、生成的样本图）
- 代码版本（git commit hash）

**更重要的**：支持**跨 run 对比**——一张表看 100 个实验的结果。

本章对应示例：

```bash
cd aigc-learning/04-training-engineering/examples
conda run -n aigc python wandb_train.py --epochs 1
tensorboard --logdir runs/
```

示例默认写 TensorBoard 日志，并把 `config.json`、`environment.json`、`metrics.json` 和 checkpoint 放到同一个 run 目录。安装 W&B 后可加 `--use-wandb`，建议先用 `WANDB_MODE=offline` 避免把 demo 数据上传到云端。

---

## 2. TensorBoard：最轻量的起步

TensorBoard 是 Google 开发的可视化工具，PyTorch 原生支持。

### 2.1 基本用法

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/exp-001")

for step in range(1000):
    loss = train_one_step()
    writer.add_scalar("train/loss", loss, step)
    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

writer.close()
```

启动面板：

```bash
tensorboard --logdir runs/ --port 6006
# 浏览器打开 http://localhost:6006
```

### 2.2 记录更多类型

```python
writer.add_scalars("loss", {"train": 0.5, "val": 0.7}, step)

img_grid = torchvision.utils.make_grid(images[:8])
writer.add_image("generated_samples", img_grid, step)

writer.add_histogram("layer1/weights", model.fc1.weight, step)

writer.add_text("config", str(config), 0)

writer.add_hparams(
    {"lr": 1e-4, "bs": 32, "layers": 6},
    {"hparam/best_loss": 0.12, "hparam/best_acc": 0.95},
)
```

### 2.3 与 PyTorch 训练循环集成

```python
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def train(model, loader, optimizer, epochs, device):
    writer = SummaryWriter()
    global_step = 0

    for epoch in range(epochs):
        model.train()
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            loss = nn.functional.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        val_loss = evaluate(model, val_loader, device)
        writer.add_scalar("val/loss", val_loss, epoch)

    writer.close()
```

### 2.4 优缺点

| 优点 | 缺点 |
|---|---|
| 零依赖（PyTorch 自带） | 没有云端存储，换机器就丢 |
| 本地速度快 | 多人协作困难 |
| 适合快速原型 | 大量 run 对比不方便 |

**定位**：本地调试用 TensorBoard，正式项目上 W&B 或 MLflow。

---

## 3. Weights & Biases（wandb）：AIGC 团队的主流选择

### 3.1 快速开始

```bash
pip install wandb
wandb login  # 输入 API key（从 https://wandb.ai/settings 获取）
```

```python
import wandb

wandb.init(
    project="my-aigc-project",
    name="exp-001-lr1e4",
    config={
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 10,
        "model": "resnet50",
        "optimizer": "adamw",
    },
)

for step in range(1000):
    loss, acc = train_step()
    wandb.log({"train/loss": loss, "train/acc": acc}, step=step)

wandb.finish()
```

打开 `https://wandb.ai/<your-name>/my-aigc-project` 就能看到实时曲线。

### 3.2 记录 config 的正确姿势

```python
wandb.init(config={...})

wandb.config.update({"extra_param": 42})

wandb.config["dropout"] = 0.1

print(wandb.config.learning_rate)
```

**黄金法则**：所有影响实验结果的参数都应该在 `config` 里。

### 3.3 日志进阶

```python
wandb.log({
    "train/loss": 0.5,
    "train/acc": 0.85,
    "lr": scheduler.get_last_lr()[0],
    "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
})

wandb.log({"samples": [
    wandb.Image(img, caption=f"step {step}")
    for img in generated_images[:8]
]})

columns = ["input", "prediction", "ground_truth"]
table = wandb.Table(columns=columns)
for inp, pred, gt in zip(inputs, preds, gts):
    table.add_data(inp, pred, gt)
wandb.log({"predictions": table})
```

### 3.4 Artifacts：管理数据和模型版本

```python
artifact = wandb.Artifact("model-checkpoint", type="model")
artifact.add_file("checkpoints/best.pt")
wandb.log_artifact(artifact)

artifact = wandb.use_artifact("my-project/model-checkpoint:latest")
artifact_dir = artifact.download()
```

### 3.5 Sweeps：自动超参搜索

```yaml
# sweep.yaml
method: bayes
metric:
  name: val/loss
  goal: minimize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-2
  batch_size:
    values: [16, 32, 64]
  optimizer:
    values: ["adam", "adamw", "sgd"]
```

```bash
wandb sweep sweep.yaml           # 返回 sweep_id
wandb agent <sweep_id>            # 启动 agent 跑实验
```

```python
def train():
    wandb.init()
    config = wandb.config
    model = build_model(config)
    for epoch in range(10):
        loss = train_epoch(model, config)
        wandb.log({"val/loss": loss})

sweep_id = wandb.sweep(sweep_config, project="my-project")
wandb.agent(sweep_id, function=train, count=20)
```

### 3.6 与 HuggingFace Trainer 集成

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="./output",
    report_to="wandb",          # 就这一行
    run_name="sft-llama-v1",
    logging_steps=10,
)

trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
```

Trainer 会自动记录 loss、learning rate、throughput 等所有指标到 W&B。

---

## 4. MLflow：开源自托管方案

MLflow 适合**不想把数据上传到第三方云**的团队——可以部署在自己的服务器上。

### 4.1 核心概念

```
MLflow Tracking
├── Experiment（实验，一组相关 run）
│   ├── Run 001
│   │   ├── Parameters（超参）
│   │   ├── Metrics（指标时序）
│   │   └── Artifacts（产出物）
│   ├── Run 002
│   └── ...
└── Model Registry（模型注册中心）
```

### 4.2 基本用法

```bash
pip install mlflow
mlflow server --host 0.0.0.0 --port 5000  # 启动 tracking server
```

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("aigc-training")

with mlflow.start_run(run_name="exp-001"):
    mlflow.log_params({
        "lr": 1e-4,
        "batch_size": 32,
        "model": "vit-base",
    })

    for step in range(1000):
        loss = train_step()
        mlflow.log_metric("train_loss", loss, step=step)

    mlflow.log_artifact("config.yaml")

    mlflow.pytorch.log_model(model, "model")
```

### 4.3 Model Registry

```python
mlflow.register_model(
    model_uri="runs:/<run_id>/model",
    name="my-vit-model",
)

model = mlflow.pytorch.load_model("models:/my-vit-model/Production")
```

### 4.4 与 HuggingFace Trainer 集成

```python
args = TrainingArguments(
    output_dir="./output",
    report_to="mlflow",         # 换成 mlflow
)
```

---

## 5. 三工具对比

| 特性 | TensorBoard | W&B | MLflow |
|---|---|---|---|
| 部署方式 | 本地 | SaaS / 自托管 | 自托管 |
| 学习成本 | ★☆☆ | ★★☆ | ★★☆ |
| 团队协作 | 差 | 优秀 | 良好 |
| 多 run 对比 | 基础 | 强大（表格+图表） | 良好 |
| Artifact 管理 | 无 | 内置 | 内置 |
| 超参搜索 | 无 | 内置 Sweeps | 无（需配合 Optuna） |
| HF Trainer 集成 | ✅ | ✅ | ✅ |
| 隐私/合规 | ✅ 本地 | ⚠️ 默认云端 | ✅ 自托管 |
| 免费额度 | 完全免费 | 个人免费，团队付费 | 完全免费 |

**推荐策略**：
- 个人 / 小团队：**W&B**（功能最全、体验最好）
- 企业 / 数据敏感：**MLflow**（完全自托管）
- 快速本地调试：**TensorBoard**（零配置）

---

## 6. 最佳实践

### 6.1 一个 run 至少记录什么

| 类别 | 必要内容 | 为什么 |
|---|---|---|
| Config | 模型、数据、优化器、训练轮数、seed | 判断 run 与 run 的差异 |
| Metrics | train/val 指标曲线、最终指标 | 比较实验结果 |
| Artifacts | checkpoint、样本、评估报告 | 支撑后续评估和部署 |
| Environment | Python、PyTorch、CUDA、GPU、git hash | 支撑可复现和问题定位 |

### 6.2 该记录什么

```python
essentials = {
    # 必记
    "train/loss": ...,
    "val/loss": ...,
    "val/metric": ...,          # accuracy / FID / BLEU 等任务指标
    "lr": ...,                  # 学习率变化
    "epoch": ...,
    "global_step": ...,

    # 推荐
    "gpu_mem_gb": ...,          # 显存峰值
    "throughput": ...,          # samples/sec 或 tokens/sec
    "grad_norm": ...,           # 梯度范数（检测爆炸/消失）

    # 可选
    "samples": ...,             # 生成的样本（图像/文本）
    "weight_histogram": ...,    # 参数分布
}
```

### 6.3 命名规范

```python
# ✅ 好：用 / 分组
wandb.log({"train/loss": ..., "train/acc": ..., "val/loss": ..., "val/acc": ...})

# ❌ 差：扁平命名
wandb.log({"train_loss": ..., "train_acc": ..., "val_loss": ..., "val_acc": ...})
```

用 `/` 分组的好处：W&B 和 TensorBoard 都会自动按前缀分面板。

### 6.4 Run 命名

```python
# ✅ 好：包含关键区分信息
wandb.init(name="sft-llama3-8b-lr3e5-bs64-ep3")

# ❌ 差：无意义名字
wandb.init(name="experiment_1")
```

### 6.5 团队工作流

```
1. 统一 project 命名（如 project-name-sft、project-name-pretrain）
2. 所有人把 config 完整记录（不要"我记得我用的是 1e-4"）
3. 重要 run 打 tag（"baseline"、"best-v2"、"paper-submission"）
4. 用 Artifact 管理 checkpoint（不要在微信群里传 .pt 文件）
```

---

## 7. 常见坑

### 7.1 忘记 `wandb.finish()`

多次 `wandb.init()` 不 finish 会导致 run 混乱。用 context manager 更安全：

```python
with wandb.init(project="my-project") as run:
    ...
# 自动 finish
```

### 7.2 log 频率太高

```python
# ❌ 每个 step 都 log（训练变慢 10%+）
for step in range(1_000_000):
    wandb.log({"loss": loss}, step=step)

# ✅ 每 N 步 log 一次
if step % 100 == 0:
    wandb.log({"loss": loss}, step=step)
```

### 7.3 log 的 step 不连续

W&B 要求 step 严格递增。如果你跳着 log（step=0, 100, 200...），x 轴会出现空白。
解决：要么不传 step（自动递增），要么用 `define_metric`：

```python
wandb.define_metric("train/loss", step_metric="global_step")
wandb.log({"train/loss": 0.5, "global_step": 100})
```

### 7.4 config 里放了不可序列化的对象

```python
# ❌ nn.Module 不能序列化
wandb.config["model"] = model

# ✅ 只记录可序列化的配置
wandb.config["model_name"] = "resnet50"
wandb.config["model_params"] = sum(p.numel() for p in model.parameters())
```

### 7.5 TensorBoard 目录堆积

每次运行都产生新目录。定期清理：

```bash
rm -rf runs/        # 全清
# 或者用日期命名，定期归档
```

### 7.6 MLflow 的 artifact 路径问题

`log_artifact` 会**复制**文件到 MLflow 存储。大模型 checkpoint 建议用 `log_artifact` 记录一个包含路径的文本文件，而不是直接上传几十 GB 的权重。

---

## 小结

| 场景 | 推荐方案 |
|---|---|
| 本地快速调试 | TensorBoard |
| 个人/小团队正式实验 | W&B |
| 企业/数据敏感 | MLflow（自托管） |
| HuggingFace 训练 | `report_to="wandb"` 一行搞定 |

**一条黄金规则**：**如果一个实验值得跑，就值得被记录。** 没有追踪的实验 = 没跑过。

下一节学习如何用 Hydra + OmegaConf 管理配置——再也不用在代码里硬编码超参数。
