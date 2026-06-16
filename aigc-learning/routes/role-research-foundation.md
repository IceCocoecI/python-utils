# 岗位路线：模型架构与研究基础

> 目标：能读懂主流 AIGC 架构论文，理解模型设计动机，并把理论映射到代码。

---

## 优先级

| 优先级 | 模块 | 学习重点 |
---|---|---|
| P0 | [02-deep-learning-libraries](../02-deep-learning-libraries/README.md) | 深度学习理论、Transformer from scratch |
| P0 | [09-frontier-models](../09-frontier-models/README.md) | LLM、图像、多模态、视频、语音架构 |
| P0 | [03-data-and-scientific-computing](../03-data-and-scientific-computing/README.md) | 张量、shape、图像和数据管线 |
| P1 | [06-finetuning-and-alignment](../06-finetuning-and-alignment/README.md) | 对齐理论、DPO、RLHF |
| P1 | [05-distributed-training](../05-distributed-training/README.md) | 大模型训练并行策略 |
| P1 | [10-cuda-and-triton](../10-cuda-and-triton/README.md) | FlashAttention、性能模型 |
| P2 | [04-training-engineering](../04-training-engineering/README.md) | 实验设计和复现 |
| P2 | [07-inference-and-deployment](../07-inference-and-deployment/README.md) | 推理系统对架构的约束 |
| P2 | [08-llm-applications](../08-llm-applications/README.md) | 了解应用需求对模型能力的牵引 |

---

## 12 周建议

| 周 | 学习内容 | 产出 |
|---|---|---|
| 1 | 深度学习基础理论 | 概念图：张量、梯度、优化 |
| 2 | Attention 和 Transformer | 手写最小 attention |
| 3 | RoPE、KV cache、GQA/MQA | 架构对比笔记 |
| 4 | LLaMA/Qwen/DeepSeek/MoE | LLM 架构综述 |
| 5 | 扩散模型和 DiT | 图像生成演进图 |
| 6 | 多模态模型 | CLIP/LLaVA/Qwen-VL 对比 |
| 7 | 视频和音频生成 | 选读笔记 |
| 8 | 微调和对齐理论 | SFT/DPO/RLHF 对比表 |
| 9 | 分布式训练理论 | 并行策略图 |
| 10 | 推理系统理论 | KV cache 和 batching 笔记 |
| 11 | GPU 性能模型 | FlashAttention 读书笔记 |
| 12 | 论文复现计划 | 一个可执行 mini reproduction plan |

---

## 必做 Lab

- [../labs/03-transformer-from-scratch/README.md](../labs/03-transformer-from-scratch/README.md)
- [../labs/07-profiling-and-optimization/README.md](../labs/07-profiling-and-optimization/README.md)

---

## 验收标准

- [ ] 能从论文图和 config 推断模型主要计算路径。
- [ ] 能解释一个架构改动解决了什么瓶颈。
- [ ] 能把论文中的公式对应到 PyTorch 代码。
- [ ] 能写出一份包含背景、方法、实验、局限的论文阅读笔记。
- [ ] 能设计一个小规模可复现实验验证论文中的一个 claim。

