# Portfolio Projects

> 作品集目标：把学习内容转成可以展示的工程成果。
> 每个作品都应该包含代码、运行命令、结果记录和一页报告。

---

## 推荐作品

| 作品 | 对应能力 | 对应 Lab |
|---|---|---|
| Mini Data + Tensor Pipeline | Python 工程、NumPy/einops、PyTorch Tensor | [Lab 01](../labs/01-python-data-tensors/README.md) |
| Mini Training Framework | PyTorch 训练、配置、日志、复现 | [Lab 02](../labs/02-train-mini-model/README.md) |
| Transformer Notes + Toy Implementation | Attention、RoPE、KV cache、采样 | [Lab 03](../labs/03-transformer-from-scratch/README.md) |
| LoRA SFT Experiment Report | 数据构造、LoRA、量化、DPO 理解 | [Lab 04](../labs/04-lora-sft-mini-llm/README.md) |
| OpenAI-Compatible Toy Server | 推理服务、API 协议、延迟记录 | [Lab 05](../labs/05-openai-compatible-server/README.md) |
| Local RAG System | 文档切分、检索、生成、评估 | [Lab 06](../labs/06-rag-mini-system/README.md) |
| Profiling Report | 性能 baseline、瓶颈判断、优化对比 | [Lab 07](../labs/07-profiling-and-optimization/README.md) |

---

## 每个作品必须包含

- `README.md`：项目目标、环境、运行命令、结果。
- `report.md`：实验记录和结论，可用 [project-report-template](../progress/project-report-template.md)。
- 可复现命令：从干净环境可以重新跑。
- 至少一个失败样例或局限性分析。
- 下一步优化计划。

---

## 最小作品集组合

如果目标是求职或面试展示，优先准备 4 个：

1. Mini Training Framework。
2. LoRA SFT Experiment Report。
3. OpenAI-Compatible Toy Server。
4. Local RAG System。

如果目标是推理/性能方向，再补：

5. Profiling Report。

