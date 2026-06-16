# AIGC Learning Labs

> `examples/` 是单个知识点的最小脚本；`labs/` 是端到端实践任务。
> 每个 Lab 都引用已有模块和示例，不重复教程正文。

---

## Lab 列表

| Lab | 主题 | 主要产出 |
|---|---|---|
| [01-python-data-tensors](./01-python-data-tensors/README.md) | Python、数据、张量基础 | 一个清晰的数据处理和张量变换小管线 |
| [02-train-mini-model](./02-train-mini-model/README.md) | PyTorch 训练闭环 | 可复现的小模型训练实验 |
| [03-transformer-from-scratch](./03-transformer-from-scratch/README.md) | Transformer 核心机制 | 能解释 attention/RoPE/KV cache 的最小实现 |
| [04-lora-sft-mini-llm](./04-lora-sft-mini-llm/README.md) | LoRA/SFT/DPO 小闭环 | 微调数据、LoRA 训练、量化/DPO 理解 |
| [05-openai-compatible-server](./05-openai-compatible-server/README.md) | 推理服务化 | 一个 OpenAI-compatible toy server |
| [06-rag-mini-system](./06-rag-mini-system/README.md) | RAG 应用 | 本地文档问答系统设计和评估 |
| [07-profiling-and-optimization](./07-profiling-and-optimization/README.md) | 性能分析 | 一份瓶颈定位和优化报告 |

---

## 执行约定

默认使用 `aigc` conda 环境：

```bash
conda run -n aigc python <script.py>
```

每个 Lab 完成后，建议在自己的学习笔记里记录：

- 环境：Python、PyTorch、CUDA、GPU。
- 命令：完整运行命令。
- 结果：关键输出、指标、截图或日志摘要。
- 问题：失败原因和修复方法。
- 复盘：这个 Lab 对应的核心概念。

