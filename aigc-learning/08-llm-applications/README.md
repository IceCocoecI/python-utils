# 模块 08：LLM 应用开发

> 大模型训好只是起点，把它变成用户愿意付费的产品才是终点。
> 本模块覆盖从 RAG 到 Agent 的完整 LLM 应用技术栈——这是 2024–2026 年 AIGC 领域最大的工程机会。

---

## 为什么 LLM 应用层是最大的机会？

```
┌─────────────────────────────────────────────────────────────────┐
│                      用户 / 产品界面                              │
├─────────────────────────────────────────────────────────────────┤
│  Agent 工程    │  RAG 系统    │  工作流编排    │  结构化输出     │  ← 本模块
├─────────────────────────────────────────────────────────────────┤
│  向量数据库    │  Embedding   │  Reranker     │  提示工程       │  ← 本模块
├─────────────────────────────────────────────────────────────────┤
│  推理引擎 (vLLM / SGLang / TRT-LLM)                             │  ← 模块 07
├─────────────────────────────────────────────────────────────────┤
│  大模型 (GPT-4 / Claude / Qwen / LLaMA / DeepSeek)              │  ← 模块 06/09
└─────────────────────────────────────────────────────────────────┘
```

训练一个基座模型需要千万级资金和大规模 GPU 集群，但基于已有模型构建应用——RAG、Agent、工作流——
只需要一台笔记本和 API Key。应用层是个人开发者和中小团队最能发挥价值的地方：

- **RAG（检索增强生成）**：让 LLM 访问你的私有数据，消除幻觉
- **向量数据库**：语义检索的基础设施，所有 RAG 系统的心脏
- **编排框架**：把 LLM 调用、检索、工具组合成可靠的工作流
- **Agent 工程**：让 LLM 自主规划、调用工具、完成复杂任务

---

## 学习内容

| # | 文档 | 核心话题 |
|---|---|---|
| 01 | [rag-fundamentals](./01-rag-fundamentals.md) | RAG 全流程：切分 → Embedding → 检索 → 生成 → 评估 |
| 02 | [vector-databases](./02-vector-databases.md) | FAISS / Milvus / Chroma / Qdrant / pgvector 对比与实战 |
| 03 | [orchestration-frameworks](./03-orchestration-frameworks.md) | LangChain / LlamaIndex / LangGraph / 提示工程 / 结构化输出 |
| 04 | [agent-engineering](./04-agent-engineering.md) | ReAct / Tool Use / MCP / 多智能体 / 安全与评估 |

---

## 前置知识

本模块假设你已经：

- 熟悉 Python 基础和 async/await（模块 01）
- 了解 Transformer 架构和注意力机制（模块 02）
- 会用 HuggingFace Transformers 加载模型和 tokenizer（模块 02）
- 了解 NumPy 和基本的向量运算（模块 03）

---

## 示例代码（`examples/`）

| 文件 | 说明 | 是否需要外部模型 |
|---|---|---|
| [`toy_rag.py`](./examples/toy_rag.py) | 使用本地 Markdown、哈希 embedding 和 NumPy 检索实现最小 RAG 流水线 | 否 |

运行：

```bash
conda run -n aigc python aigc-learning/08-llm-applications/examples/toy_rag.py --self-test
```

这个示例用于理解 RAG 的 ingestion、chunking、embedding、retrieval、generation 形状，不代表真实 embedding 模型效果。

---

## 推荐配套资源

| 类型 | 资源 | 说明 |
|---|---|---|
| 课程 | [LangChain 官方教程](https://python.langchain.com/docs/tutorials/) | 从 Prompt 到 Agent 的完整教程 |
| 课程 | [DeepLearning.AI RAG 课程](https://www.deeplearning.ai/short-courses/) | Andrew Ng 与各厂商合作的短课 |
| 文档 | [LlamaIndex 官方文档](https://docs.llamaindex.ai/) | RAG 框架的标杆实现 |
| 文档 | [LangGraph 文档](https://langchain-ai.github.io/langgraph/) | 状态机驱动的 Agent 框架 |
| 文档 | [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki) | Meta 的向量检索库 |
| 文档 | [Milvus 文档](https://milvus.io/docs) | 最流行的开源向量数据库 |
| 文档 | [Chroma 文档](https://docs.trychroma.com/) | 轻量级嵌入式向量数据库 |
| 文档 | [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) | 工具调用的标准范式 |
| 文档 | [MCP 协议规范](https://modelcontextprotocol.io/) | Anthropic 发起的模型上下文协议 |
| 论文 | [RAG 原始论文](https://arxiv.org/abs/2005.11401) | Lewis et al., 2020 |
| 论文 | [ReAct](https://arxiv.org/abs/2210.03629) | Yao et al., 2022 |
| 代码 | [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch) | LangChain 团队的 RAG 手把手教程 |

---

## 推荐学习顺序

```
01-rag-fundamentals ──→ 02-vector-databases ──→ 03-orchestration-frameworks ──→ 04-agent-engineering
        │                        │                          │
        │                        │                          │
   理解 RAG 为什么          掌握检索基础设施          学会把组件串成工作流
   以及完整流水线                                    最终构建自主 Agent
```

建议每一章花 2–3 天：先读教程，再动手跑示例，最后尝试用自己的数据搭建一个 mini 项目。

---

## 自检清单

学完本模块，你应该能自信地回答以下问题：

- [ ] RAG 解决了 LLM 的哪三个根本问题？
- [ ] 文本切分时 chunk size 和 overlap 该怎么选？为什么？
- [ ] Dense retrieval 和 Sparse retrieval（BM25）各自的优劣？什么时候用 hybrid？
- [ ] Reranker 在 RAG 流水线中的位置和作用是什么？
- [ ] FAISS 的 IVF 和 HNSW 索引分别适合什么场景？
- [ ] Milvus、Chroma、Qdrant 三者的定位差异？
- [ ] LangChain 的 LCEL 和 LlamaIndex 的 QueryEngine 分别解决什么问题？
- [ ] LangGraph 的 StateGraph 和普通 chain 有什么本质区别？
- [ ] OpenAI Function Calling 的 JSON Schema 怎么写？
- [ ] MCP 协议中的 Tool 和 Resource 有什么区别？
- [ ] ReAct 和 Plan-and-Execute 两种 Agent 架构各自的优缺点？
- [ ] 如何防御 prompt injection？有哪些常见攻击方式？
- [ ] 评估 RAG 系统用什么指标？Ragas 框架提供了哪些？
- [ ] 控制 LLM 应用成本的三大手段是什么？
