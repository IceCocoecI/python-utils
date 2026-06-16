# 岗位路线：LLM 应用工程师

> 目标：能构建、评估、部署 RAG / Agent / LLM 工作流。

---

## 优先级

| 优先级 | 模块 | 学习重点 |
---|---|---|
| P0 | [01-python-foundations](../01-python-foundations/README.md) | Python 工程能力、async、类型、测试 |
| P0 | [02-deep-learning-libraries](../02-deep-learning-libraries/README.md) | Transformers 推理、Tokenizer、生成参数 |
| P0 | [07-inference-and-deployment](../07-inference-and-deployment/README.md) | FastAPI、OpenAI-compatible API、服务化 |
| P0 | [08-llm-applications](../08-llm-applications/README.md) | RAG、向量库、LangChain/LlamaIndex/LangGraph、Agent |
| P1 | [03-data-and-scientific-computing](../03-data-and-scientific-computing/README.md) | 数据格式、JSONL、Parquet、embedding 向量处理 |
| P1 | [06-finetuning-and-alignment](../06-finetuning-and-alignment/README.md) | LoRA/SFT 基础，理解什么时候该微调 |
| P2 | [09-frontier-models](../09-frontier-models/README.md) | LLM 和多模态架构概览 |
| P2 | [05-distributed-training](../05-distributed-training/README.md) | 只需理解概念，非主线 |
| P2 | [10-cuda-and-triton](../10-cuda-and-triton/README.md) | 只需理解 profiling 和性能瓶颈 |

---

## 8 周建议

| 周 | 学习内容 | 产出 |
|---|---|---|
| 1 | Python 工程、async、类型、日志、配置 | 一个带配置和日志的小 CLI |
| 2 | Transformers 推理、chat template、流式输出 | 本地或 API 模型调用封装 |
| 3 | FastAPI、OpenAI-compatible API | 一个 `/v1/chat/completions` toy server |
| 4 | RAG 基础：切分、embedding、FAISS/Chroma | 本地文档问答 |
| 5 | 检索优化：hybrid、rerank、评估 | RAG 评估报告 |
| 6 | LangChain/LlamaIndex/LangGraph | 一个可观测工作流 |
| 7 | Tool use、Agent、安全 | 一个带工具调用的 Agent |
| 8 | 部署、压测、成本控制 | 服务 README + 性能记录 |

---

## 必做 Lab

- [../labs/05-openai-compatible-server/README.md](../labs/05-openai-compatible-server/README.md)
- [../labs/06-rag-mini-system/README.md](../labs/06-rag-mini-system/README.md)

---

## 验收标准

- [ ] 能设计 RAG 的 ingestion、retrieval、generation、evaluation 四段链路。
- [ ] 能解释 chunk size、embedding model、top-k、reranker 如何影响效果。
- [ ] 能写出 OpenAI-compatible 接口，并接入前端或脚本调用。
- [ ] 能处理 prompt injection、tool abuse、敏感信息泄露等风险。
- [ ] 能用日志和 trace 定位一次线上回答质量问题。

