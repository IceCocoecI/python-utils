# Lab 02：训练一个可复现的小模型

> 目标：完成一个最小但规范的训练闭环：数据、模型、训练、评估、日志、配置、保存、复现。

---

## 前置知识

- [../../02-deep-learning-libraries/02-pytorch-training-loop.md](../../02-deep-learning-libraries/02-pytorch-training-loop.md)
- [../../04-training-engineering/00-training-engineering-theory.md](../../04-training-engineering/00-training-engineering-theory.md)
- [../../04-training-engineering/01-experiment-tracking.md](../../04-training-engineering/01-experiment-tracking.md)
- [../../04-training-engineering/02-config-management.md](../../04-training-engineering/02-config-management.md)
- [../../04-training-engineering/04-reproducibility.md](../../04-training-engineering/04-reproducibility.md)

---

## 运行脚本

```bash
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/mlp_mnist.py --synthetic --epochs 1 --max-train-batches 3 --max-val-batches 2 --workers 0
conda run -n aigc python aigc-learning/04-training-engineering/examples/wandb_train.py --epochs 1
conda run -n aigc python aigc-learning/04-training-engineering/examples/hydra_train.py --epochs 1
conda run -n aigc python aigc-learning/04-training-engineering/examples/optuna_search.py --trials 2 --epochs 1
conda run -n aigc python aigc-learning/04-training-engineering/examples/reproducible_train.py --epochs 1
```

---

## 任务

1. 选择 synthetic MNIST 或模块 04 的合成二分类任务作为 baseline。
2. 记录模型结构、参数量、batch size、learning rate、optimizer、seed。
3. 跑一次 baseline 训练，保存 loss/accuracy。
4. 接入配置管理：至少支持从命令行覆盖 `epochs`、`lr`、`batch_size`。
5. 接入实验记录：TensorBoard 或本地 JSON/CSV 均可。
6. 重复同一配置训练两次，比较指标是否一致。
7. 改动一个超参，写出变化和判断。

---

## 验收标准

- [ ] 能从零说明训练循环的七个部件。
- [ ] 每次训练都有配置和指标记录。
- [ ] 模型权重保存路径不污染仓库根目录。
- [ ] 能解释随机种子、DataLoader worker、CUDA deterministic 的影响。
- [ ] 能指出当前实验结果的可信边界。

---

## 延伸挑战

- 增加 early stopping。
- 用 Optuna 做 8 次以上搜索。
- 用 TensorBoard 比较不同 run。
- 增加一个故意导致 NaN 的配置，并写排查记录。

