# 岗位路线：推理部署工程师

> 目标：能把模型稳定、高效、可观测地部署成服务，并能解释性能瓶颈。

---

## 优先级

| 优先级 | 模块 | 学习重点 |
---|---|---|
| P0 | [02-deep-learning-libraries](../02-deep-learning-libraries/README.md) | Transformer、KV cache、生成参数、显存预算 |
| P0 | [07-inference-and-deployment](../07-inference-and-deployment/README.md) | vLLM/SGLang、FastAPI、容量规划、服务化 |
| P0 | [10-cuda-and-triton](../10-cuda-and-triton/README.md) | profiling、GPU 性能模型、kernel 基础 |
| P1 | [05-distributed-training](../05-distributed-training/README.md) | TP/PP/DP、通信、并行策略 |
| P1 | [06-finetuning-and-alignment](../06-finetuning-and-alignment/README.md) | 量化、LoRA merge、模型格式 |
| P1 | [01-python-foundations](../01-python-foundations/README.md) | async、服务端 Python、类型 |
| P2 | [08-llm-applications](../08-llm-applications/README.md) | 理解上层调用模式和负载特征 |
| P2 | [09-frontier-models](../09-frontier-models/README.md) | 架构差异对推理的影响 |

---

## 8 周建议

| 周 | 学习内容 | 产出 |
|---|---|---|
| 1 | Transformer、KV cache、采样 | KV cache 显存估算 |
| 2 | 推理指标：TTFT/TPOT/吞吐/延迟 | 指标说明文档 |
| 3 | FastAPI 和 OpenAI-compatible API | toy serving |
| 4 | vLLM/SGLang/llama.cpp 对比 | 引擎选型表 |
| 5 | 量化、batching、并发 | 压测报告 |
| 6 | 多卡并行策略 | 容量规划草案 |
| 7 | torch.profiler / Nsight 工作流 | profiling trace 和分析 |
| 8 | 生产化：日志、健康检查、限流、回滚 | 部署 checklist |

---

## 必做 Lab

- [../labs/05-openai-compatible-server/README.md](../labs/05-openai-compatible-server/README.md)
- [../labs/07-profiling-and-optimization/README.md](../labs/07-profiling-and-optimization/README.md)

---

## 验收标准

- [ ] 能估算不同 batch、seq len、并发下的 KV cache 显存。
- [ ] 能解释 prefill 和 decode 的性能差异。
- [ ] 能选择合适的推理引擎和量化方案。
- [ ] 能设计基本压测方法，并读懂吞吐/延迟曲线。
- [ ] 能用 profiler 判断瓶颈来自 CPU、I/O、GPU compute 还是 GPU memory。

