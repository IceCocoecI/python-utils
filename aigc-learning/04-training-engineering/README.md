# 模块 04：训练与实验工程化

> 模型跑起来只是第一步。**让实验可追踪、可配置、可复现、可调优**才是 AIGC 工程师与"调包侠"的分水岭。
> 本模块教你四件事：实验追踪、配置管理、超参搜索、可复现性——掌握它们，你的训练流程从"手工作坊"升级为"工业级"。

---

## 为什么这一章重要？

每个 AIGC 工程师迟早会遇到这些场景：

| 场景 | 没有工程化的后果 |
|---|---|
| "上周那个 loss 最低的 run 用的什么参数？" | 翻 terminal 历史，找不到了 |
| "我想试 50 组超参" | 手动改代码跑 50 次，CSV 记录 |
| "同样的代码，同事跑出来结果不一样" | debug 三天，发现是随机种子 |
| "老板说用上个月的模型，那个配置是什么？" | 找不到了，只能重新调 |
| "切换 lr_scheduler 要改 5 个地方" | 改漏一个，跑了一晚上的实验报废 |

**一句话**：没有实验工程化，你的工作不可积累、不可协作、不可信赖。

---

## 学习内容

| # | 文档 | 核心话题 |
|---|---|---|
| 00 | [training-engineering-theory](./00-training-engineering-theory.md) | 实验工程化的理论模型：实验记录、配置空间、搜索空间、可复现边界 |
| 01 | [experiment-tracking](./01-experiment-tracking.md) | TensorBoard / W&B / MLflow：把每一次训练变成可查询的记录 |
| 02 | [config-management](./02-config-management.md) | OmegaConf / Hydra：告别 argparse 地狱 |
| 03 | [hyperparameter-search](./03-hyperparameter-search.md) | Optuna / Ray Tune：让机器帮你调参 |
| 04 | [reproducibility](./04-reproducibility.md) | 随机种子 / 确定性模式 / 环境锁定 / 数据版本 |

---

## 示例代码（`examples/`）

| 文件 | 说明 | 是否需要 GPU |
|---|---|---|
| [`common.py`](./examples/common.py) | 合成数据集、MLP、训练循环、环境快照等共享工具 | 否 |
| [`wandb_train.py`](./examples/wandb_train.py) | TensorBoard 实验追踪 + 可选 W&B 日志 | 否 |
| [`hydra_train.py`](./examples/hydra_train.py) | Hydra 配置驱动训练；未安装 Hydra 时自动退化为 argparse demo | 否 |
| [`optuna_search.py`](./examples/optuna_search.py) | Optuna 超参搜索；未安装 Optuna 时自动退化为随机搜索 demo | 否 |
| [`reproducible_train.py`](./examples/reproducible_train.py) | 两次同配置训练并校验指标一致性，输出复现报告 | 否 |

这些示例都使用一个小型合成二分类任务，默认不下载数据、不需要 GPU、不要求登录外部服务。

### 在当前 `aigc` 环境运行

```bash
cd aigc-learning/04-training-engineering/examples

conda run -n aigc python wandb_train.py --epochs 1
conda run -n aigc python hydra_train.py --epochs 1
conda run -n aigc python optuna_search.py --trials 2 --epochs 1
conda run -n aigc python reproducible_train.py --epochs 1
```

当前示例的核心路径只依赖 PyTorch 和 TensorBoard。若要体验完整工具链，再安装可选依赖：

```bash
pip install wandb hydra-core omegaconf optuna
```

安装后可以运行：

```bash
WANDB_MODE=offline python wandb_train.py --use-wandb --epochs 2
python hydra_train.py training.epochs=2 optimizer.lr=0.001 model=wide
python optuna_search.py --trials 8 --epochs 2
```

TensorBoard 查看命令：

```bash
tensorboard --logdir runs/
```

---

## 理论与实践怎么组织

本模块建议按三层学习：

| 层次 | 要回答的问题 | 对应材料 |
|---|---|---|
| 理论层 | 实验为什么要被抽象成 record？配置为什么是组合空间？搜索为什么要处理噪声？ | `00-training-engineering-theory.md` |
| 工具层 | TensorBoard/W&B/MLflow、Hydra/OmegaConf、Optuna/Ray Tune 各自解决什么边界问题？ | `01` ~ `04` 文档 |
| 模板层 | 如何把理论和工具落到一个可运行训练脚本？ | `examples/` |

学习顺序建议：

1. 先读 `00`，建立“实验系统”的整体模型。
2. 跑通 `examples/reproducible_train.py`，理解最小可复现实验闭环。
3. 跑 `wandb_train.py` 和 `hydra_train.py`，把追踪与配置接入训练循环。
4. 最后跑 `optuna_search.py`，理解搜索空间、预算和噪声的关系。

---

## 推荐配套资源

### 官方文档
- [TensorBoard 文档](https://www.tensorflow.org/tensorboard)
- [Weights & Biases 文档](https://docs.wandb.ai/)
- [MLflow 文档](https://mlflow.org/docs/latest/index.html)
- [Hydra 文档](https://hydra.cc/docs/intro/)
- [OmegaConf 文档](https://omegaconf.readthedocs.io/)
- [Optuna 文档](https://optuna.readthedocs.io/)
- [Ray Tune 文档](https://docs.ray.io/en/latest/tune/index.html)

### 实战参考
- [pytorch/examples](https://github.com/pytorch/examples) — 官方训练模板
- [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) — Hydra + Lightning 工程化模板
- [HuggingFace Trainer 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py) — 看工业级训练循环怎么处理配置/日志/种子

---

## 自检清单

学完本模块，你应该能：

- [ ] 解释训练工程化的四个核心对象：Experiment、Run、Config、Artifact。
- [ ] 区分可重复（repeatability）、可复现（reproducibility）、可复制（replicability）。
- [ ] 说明为什么超参搜索是带噪声、带预算约束的优化问题。
- [ ] 用 TensorBoard 查看训练曲线和中间结果。
- [ ] 用 W&B 记录一次完整训练，包含 config、metrics、artifacts。
- [ ] 解释 MLflow 的 experiment → run → metric 三层结构。
- [ ] 用 Hydra 写一个配置驱动的训练脚本，从命令行覆盖任意参数。
- [ ] 用 OmegaConf 实现配置插值（`${model.hidden_dim}`）。
- [ ] 用 Hydra config groups 管理 model/data/optimizer 三组配置。
- [ ] 用 Optuna 对一个训练任务做贝叶斯超参搜索。
- [ ] 解释 Optuna 剪枝（pruning）的原理和好处。
- [ ] 写出一个完整的随机种子设置函数（覆盖 Python / NumPy / PyTorch）。
- [ ] 解释 `torch.use_deterministic_algorithms(True)` 的作用和限制。
- [ ] 用 Docker 或 `uv.lock` 锁定训练环境。
- [ ] 描述一个从代码到数据到环境的完整可复现方案。
- [ ] 跑通 `examples/` 下四个入口脚本，并能解释每个输出文件的作用。
