# 岗位路线：微调与对齐工程师

> 目标：能完成从数据构造到 LoRA/QLoRA、SFT、DPO 评估的微调闭环。

---

## 优先级

| 优先级 | 模块 | 学习重点 |
---|---|---|
| P0 | [01-python-foundations](../01-python-foundations/README.md) | Python 工程、类型、配置、测试 |
| P0 | [02-deep-learning-libraries](../02-deep-learning-libraries/README.md) | PyTorch 训练循环、Transformers、Trainer、Transformer |
| P0 | [03-data-and-scientific-computing](../03-data-and-scientific-computing/README.md) | JSONL、Parquet、datasets、数据清洗 |
| P0 | [04-training-engineering](../04-training-engineering/README.md) | 实验追踪、Hydra、复现、超参搜索 |
| P0 | [06-finetuning-and-alignment](../06-finetuning-and-alignment/README.md) | LoRA、QLoRA、SFT、DPO |
| P1 | [05-distributed-training](../05-distributed-training/README.md) | Accelerate、DeepSpeed、FSDP 基础 |
| P1 | [07-inference-and-deployment](../07-inference-and-deployment/README.md) | 微调后模型评估和部署 |
| P2 | [09-frontier-models](../09-frontier-models/README.md) | LLM 架构、MoE、长上下文 |
| P2 | [10-cuda-and-triton](../10-cuda-and-triton/README.md) | profiling，训练瓶颈定位 |

---

## 10 周建议

| 周 | 学习内容 | 产出 |
|---|---|---|
| 1 | Python 工程、数据格式 | 数据清洗脚本 |
| 2 | PyTorch 训练循环、显存预算 | 小模型训练 baseline |
| 3 | Transformers、datasets、Trainer | 文本分类或 causal LM toy 训练 |
| 4 | 实验追踪、配置、复现 | 可复现实验模板 |
| 5 | LoRA/PEFT | tiny model LoRA 微调 |
| 6 | 量化和 QLoRA | 显存对比记录 |
| 7 | SFT 数据和 chat template | SFT 数据 pipeline |
| 8 | DPO/RLHF 理论与 toy loss | preference 训练 demo |
| 9 | Accelerate/DeepSpeed/FSDP | 单机多进程或配置模拟 |
| 10 | 评估、部署、报告 | 微调实验报告 |

---

## 必做 Lab

- [../labs/02-train-mini-model/README.md](../labs/02-train-mini-model/README.md)
- [../labs/04-lora-sft-mini-llm/README.md](../labs/04-lora-sft-mini-llm/README.md)

---

## 验收标准

- [ ] 能从原始数据生成 SFT JSONL，并说明字段含义。
- [ ] 能配置 LoRA target modules，并说明为什么选这些层。
- [ ] 能估算 full fine-tuning、LoRA、QLoRA 的显存差异。
- [ ] 能用实验追踪工具比较多组训练配置。
- [ ] 能写出训练报告：数据、配置、指标、失败样例、下一步。

