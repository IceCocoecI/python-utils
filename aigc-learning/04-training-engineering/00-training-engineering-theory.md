
# 00 · 训练工程化理论框架

> 目标：先建立“实验系统”的抽象模型，再学习工具。
> TensorBoard、W&B、Hydra、Optuna 都只是实现手段；真正重要的是你能把训练过程拆成可记录、可组合、可比较、可复现的对象。

---

## 1. 训练工程化解决的不是“写代码”问题

普通训练脚本关心：

```
data -> model -> loss -> optimizer -> checkpoint
```

训练工程化关心的是另一层问题：

```
一次实验为什么得到这个结果？
这个结果能不能被别人找到、比较、解释、复现？
下一次实验应该改什么，为什么？
```

所以本模块的核心不是“多加几个库”，而是把训练活动组织成一个可审计系统。

---

## 2. 四个核心对象

### 2.1 Experiment：一组有共同目标的实验

Experiment 是一个研究问题或工程目标，例如：

- “Qwen-7B SFT 学习率扫描”
- “图像生成模型不同 noise schedule 对比”
- “RAG reranker 蒸馏实验”

它不是一次运行，而是一组 run 的集合。Experiment 的边界应该由问题定义，而不是由日期或个人习惯定义。

### 2.2 Run：一次不可变的训练记录

Run 是最小可比较单位。一个合格的 run 至少包含：

| 类别 | 示例 |
|---|---|
| 配置 | lr、batch size、模型结构、数据路径、seed |
| 指标 | train/loss、val/loss、accuracy、FID、BLEU |
| 产物 | checkpoint、样本、日志、评估报告 |
| 代码 | git commit、diff 状态 |
| 环境 | Python、PyTorch、CUDA、GPU 型号 |
| 数据 | 数据版本、split seed、checksum |

关键原则：**Run 一旦完成，就不要修改它的含义**。如果发现配置错了，应该创建新 run，而不是覆盖旧 run。

### 2.3 Config：实验空间中的一个点

Config 不是“参数字典”这么简单。它代表实验空间里的一个具体点：

```
config = model x data x optimizer x scheduler x training x seed x environment
```

工程上最常见的错误是只记录 optimizer 参数，却漏掉数据版本、预处理逻辑、随机种子和环境。这样得到的 run 只能看曲线，不能复现。

### 2.4 Artifact：可被后续系统引用的产物

Artifact 包括：

- checkpoint
- tokenizer / vocab
- 数据切分文件
- 生成样本
- 评估结果
- 配置快照
- 训练日志

Artifact 的关键不是“保存文件”，而是建立引用关系：

```
run A 使用 data:v3，产生 model:v12
run B 使用 model:v12，产生 eval_report:v4
```

一旦 artifact 关系清楚，模型从训练到评估再到部署的链路才可追踪。

---

## 3. 训练工程化的三条主线

### 3.1 记录：把实验变成证据

没有记录时，实验结果只是记忆。记录系统要回答：

- 这次训练用了什么配置？
- 训练过程中发生了什么？
- 结果和上一次相比好在哪里？
- checkpoint 对应哪份代码和数据？

TensorBoard、W&B、MLflow 都属于这一主线。

### 3.2 组合：把配置变成可管理空间

当实验规模变大，配置会天然分层：

```
model:
  hidden_dim
  num_layers
data:
  dataset
  max_length
optimizer:
  lr
  weight_decay
training:
  seed
  epochs
  batch_size
```

Hydra 的价值在于把配置拆成可组合的 config groups。你不再复制 10 份 YAML，而是声明：

```
model=small + data=sft + optimizer=adamw
model=large + data=dpo + optimizer=adafactor
```

这会把“改脚本”变成“选择配置组合”。

### 3.3 搜索：把调参变成带预算的优化

超参搜索不是盲目试参数，而是一个带噪声、带预算约束的优化问题：

```
目标函数：f(config) = validation_loss
成本：一次训练消耗 time/GPU/money
噪声：不同 seed、数据顺序、硬件都会影响结果
约束：显存、训练时长、模型结构合法性
```

因此搜索策略要同时考虑：

- 搜索空间是否合理
- 采样策略是否高效
- 差的 trial 能否提前停止
- top config 是否经过多 seed 验证
- 短训找到的参数能否迁移到完整训练

Optuna 和 Ray Tune 解决的是这一主线。

---

## 4. 可复现性的三个层级

中文里经常把 repeatability、reproducibility、replicability 都叫“复现”，但工程上最好区分：

| 层级 | 含义 | 典型要求 |
|---|---|---|
| Repeatability | 同一机器、同一代码、同一配置，多次运行结果一致 | seed、DataLoader generator、确定性算子 |
| Reproducibility | 不同机器或不同成员，使用同一代码和配置得到等价结果 | 环境锁定、数据版本、git hash |
| Replicability | 独立实现或不同框架也能验证同一结论 | 多 seed、统计报告、清晰方法描述 |

大多数工程项目首先追求 reproducibility，而不是 bit-level determinism。完全确定性会牺牲速度，并且在不同 GPU 架构之间仍然可能有浮点差异。

---

## 5. AIGC 训练里的特殊难点

### 5.1 训练成本高

LLM、Diffusion、多模态模型的单次训练成本高，导致“试错”本身很贵。工程策略是：

- 先用小模型或小数据验证代码路径
- 再用短训筛选搜索空间
- 最后把候选配置扩展到完整训练

不要直接在最大模型、全量数据上开始调参。

### 5.2 指标不总是稳定

AIGC 任务常见指标：

- loss
- perplexity
- win rate
- FID / CLIP score
- BLEU / ROUGE
- 人工偏好

这些指标有些和真实体验弱相关，有些噪声很大。工程上要避免只看一个 final metric，应同时记录训练曲线、样本、评估集切片和多 seed 结果。

### 5.3 数据版本影响巨大

AIGC 训练常常包含复杂数据处理：

- 去重
- 过滤
- 分桶
- prompt 模板
- tokenizer 版本
- 图像 resize/crop 策略
- SFT/DPO 数据混合比例

这些都属于实验配置。只记录模型和 optimizer 是不够的。

---

## 6. 一个合理的训练系统形态

最小可用版本：

```
project/
├── conf/
│   ├── config.yaml
│   ├── model/
│   ├── data/
│   └── optimizer/
├── train.py
├── eval.py
├── outputs/
│   └── <run_id>/
│       ├── resolved_config.yaml
│       ├── metrics.json
│       ├── environment.json
│       └── checkpoints/
└── README.md
```

成熟版本：

```
code version -> config composition -> training run
       |              |                  |
       v              v                  v
    git hash     resolved config     metrics/artifacts
       \              |                  /
        \             v                 /
         ------ experiment tracker -----
```

成熟训练系统的判断标准：

- 新人能按 README 跑出一个 baseline
- 任意 checkpoint 能追溯到配置、代码、数据和环境
- 任意线上模型能追溯到训练 run
- 重要结果有多 seed 验证
- 超参搜索有明确预算和搜索空间说明

---

## 7. 本模块文档如何对应理论

| 理论问题 | 对应章节 | 对应示例 |
|---|---|---|
| 如何记录 run？ | `01-experiment-tracking.md` | `examples/wandb_train.py` |
| 如何表达 config 空间？ | `02-config-management.md` | `examples/hydra_train.py` |
| 如何搜索 config？ | `03-hyperparameter-search.md` | `examples/optuna_search.py` |
| 如何证明结果可信？ | `04-reproducibility.md` | `examples/reproducible_train.py` |

建议学习时不要只读工具 API。每看一个工具，都问自己三个问题：

1. 它在保存哪个核心对象？
2. 它让哪个人工流程自动化了？
3. 如果没有它，我需要用什么文件和约定替代？

---

## 8. 实践原则

1. 实验先有名字和目标，再开始训练。
2. 所有影响结果的参数都进 config。
3. 每次 run 保存 resolved config，而不是只保存默认配置。
4. 指标按 `train/`、`val/`、`eval/` 分组命名。
5. checkpoint 必须能追溯到 git hash、数据版本和 config。
6. 搜索空间先窄后宽，先低成本筛选再完整训练。
7. 重要结论至少跑多个 seed。
8. 可复现不是最后补文档，而是训练脚本从第一天就要支持的能力。

---

## 小结

训练工程化的本质是把训练从“一次脚本运行”升级为“可审计的实验系统”。

本模块后续四章分别展开：

- 实验追踪：Run 如何被记录和比较
- 配置管理：Config 如何被组合和覆盖
- 超参搜索：Config 空间如何被高效探索
- 可复现性：Run 的结果如何被验证和复建
