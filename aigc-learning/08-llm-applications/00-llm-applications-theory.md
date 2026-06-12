# 00 · LLM 应用理论框架

> 本文是模块 08 的理论底座。
> 后续 RAG、向量数据库、编排框架、Agent 工程都可以看成同一个系统问题：
> 如何稳定、可控、可评估地把 LLM 接入真实业务环境。

---

## 1. 本模块真正解决什么问题？

LLM 应用不是“调用一次模型 API”。真实产品通常要解决四件事：

| 问题 | 典型表现 | 对应技术 |
|---|---|---|
| 知识不够 | 模型不知道私有数据、最新事实、业务规则 | RAG、工具调用、数据库查询 |
| 输出不稳 | 格式飘、幻觉、引用错误、长任务中断 | 结构化输出、校验、重试、评估 |
| 流程复杂 | 一次回答要检索、计算、调用工具、多步推理 | Chain、Graph、Agent |
| 成本受限 | 延迟高、token 贵、吞吐低、上下文窗口有限 | 缓存、路由、压缩、批处理、降级策略 |

所以 LLM 应用的核心不是“让模型更聪明”，而是**构造一个可靠的上下文和行动系统**。

---

## 2. LLM 应用的基本系统模型

一个典型 LLM 应用可以拆成六层：

```text
用户输入
  ↓
意图识别 / 路由
  ↓
上下文构造：Prompt + History + Retrieved Docs + Tool Results
  ↓
模型生成：LLM / VLM / Embedding / Reranker
  ↓
输出校验：Schema / Citation / Safety / Business Rules
  ↓
反馈闭环：Logs / Eval / Human Review / Dataset
```

从工程角度看，LLM 只是中间的推理组件。真正决定应用质量的是：

- 给模型什么上下文；
- 允许模型调用哪些工具；
- 如何约束输出格式；
- 如何判断结果对不对；
- 失败时如何恢复。

---

## 3. 五个核心对象

| 对象 | 作用 | 常见实现 |
|---|---|---|
| Prompt | 指令、角色、输出约束、上下文组织方式 | prompt template、chat template |
| Context | 模型当前可见的信息窗口 | history、retrieved chunks、tool outputs |
| Retriever | 从外部知识库找相关信息 | BM25、embedding、hybrid search、reranker |
| Tool | 让模型执行环境动作 | function calling、MCP tool、HTTP API、SQL |
| Memory | 跨轮次、跨任务保留状态 | conversation summary、profile、vector memory、state graph |

常见误区是把所有东西都塞进 Prompt。更稳的做法是明确分层：

- Prompt 负责行为约束；
- Context 负责事实输入；
- Tool 负责外部动作；
- Memory 负责长期状态；
- Retriever 负责知识召回。

---

## 4. RAG 的理论链路

RAG（Retrieval-Augmented Generation）可以拆成离线和在线两条链路。

```text
离线索引：
Document → Parse → Chunk → Embed → Index

在线回答：
Query → Rewrite → Retrieve → Rerank → Build Context → Generate → Verify
```

### 4.1 Chunking 是信息压缩问题

切分文档不是简单按字数截断，而是在做信息压缩：

| 切分过大 | 切分过小 |
|---|---|
| 检索结果不精准，包含大量无关内容 | 丢上下文，答案需要的信息分散在多个 chunk |
| 上下文窗口浪费 | 召回数量增加，rerank 成本上升 |
| 容易让模型引用错位置 | 语义边界被切断 |

经验原则：

- 技术文档优先按标题、段落、代码块切。
- FAQ 和知识库优先按问答单元切。
- 长报告可以用层级切分：章节摘要 + 段落 chunk。
- chunk overlap 不是越大越好，它会制造重复召回和上下文冗余。

### 4.2 Embedding 是语义空间映射

Embedding 模型把文本映射到向量空间。相似问题应该在向量空间中距离更近。

常见距离：

| 距离 | 含义 | 适合场景 |
|---|---|---|
| Cosine | 比较方向，相当于语义角度 | 文本 embedding 最常用 |
| Inner Product | 点积，相似度和向量长度都参与 | 归一化后等价于 cosine 排序 |
| L2 | 欧氏距离 | 图像、聚类、部分 ANN 索引 |

关键约束：**索引和查询必须使用同一个 embedding 空间**。
混用不同模型、不同归一化策略、不同语言预处理，会让检索质量直接失效。

### 4.3 Retrieval 是召回与精度的权衡

检索阶段通常追求高召回，rerank 阶段追求高精度：

```text
用户问题
  ↓
粗召回：top 50~200，宁可多找
  ↓
重排：top 5~20，压掉噪声
  ↓
上下文构造：只放模型真正需要的证据
```

| 方法 | 优点 | 缺点 |
|---|---|---|
| BM25 | 关键词精确、可解释、便宜 | 语义泛化弱 |
| Dense Retrieval | 语义泛化强 | 容易漏掉专有名词、数字、代码符号 |
| Hybrid | 同时覆盖关键词和语义 | 系统复杂度更高 |
| Reranker | 精度高 | 延迟和成本更高 |

### 4.4 Generation 必须基于证据

RAG 的目标不是“让回答更像参考资料”，而是让回答**可追溯到证据**。

实用约束：

- 让模型只基于给定上下文回答。
- 输出引用来源或 chunk id。
- 当证据不足时明确拒答。
- 对数值、日期、法规、合同条款做后处理校验。

---

## 5. 向量数据库的本质

向量数据库解决三个问题：

1. 存储向量和元数据。
2. 快速近似最近邻搜索（ANN）。
3. 支持过滤、更新、分片、权限、持久化。

### 5.1 ANN 为什么是近似的？

精确搜索需要对每个向量计算距离：

```text
复杂度 ≈ O(N × D)
N = 文档数量
D = 向量维度
```

当 N 很大时，需要用 ANN 索引牺牲少量召回换取速度。

| 索引 | 核心思想 | 适合场景 |
|---|---|---|
| Flat | 暴力精确搜索 | 小数据、评估基准 |
| IVF | 先聚类，再搜最近簇 | 大规模、可控召回 |
| HNSW | 图搜索 | 低延迟在线查询 |
| PQ | 向量压缩 | 内存受限、超大规模 |

### 5.2 元数据过滤很重要

真实 RAG 系统很少只做向量相似度。通常还要过滤：

- 用户权限；
- 文档类型；
- 时间范围；
- 产品线；
- 语言；
- 地区法规。

如果权限过滤在检索后才做，可能出现“召回结果全被过滤掉”的问题。
更稳的方式是让数据库支持检索前过滤，或按权限/租户隔离索引。

---

## 6. 编排框架解决什么？

LangChain、LlamaIndex、LangGraph 这类框架解决的是“把组件稳定组合起来”的问题。

| 抽象 | 解决的问题 |
|---|---|
| Prompt Template | 统一提示词输入输出变量 |
| Retriever | 统一检索接口 |
| Chain | 固定步骤流水线 |
| Graph | 带状态和分支的工作流 |
| Tool | 模型可调用的外部能力 |
| Callback / Trace | 观测每一步输入输出 |

选型原则：

- 简单 RAG：直接写代码或用 LlamaIndex。
- 多步骤固定流程：Chain 足够。
- 有分支、循环、人工审批、失败恢复：用 Graph。
- 工具选择由模型动态决定：Agent。

---

## 7. Agent 的理论模型

Agent 可以理解成一个循环控制系统：

```text
Observe → Think / Plan → Act → Observe → ...
```

核心区别：

| 模式 | 执行路径 | 风险 |
|---|---|---|
| Chain | 人写死步骤 | 不够灵活 |
| Graph | 人定义状态和分支 | 需要设计状态模型 |
| Agent | 模型动态决定下一步 | 不稳定、成本高、难评估 |

### 7.1 ReAct

ReAct 把推理和行动交替进行：

```text
Thought → Action → Observation → Thought → ...
```

优点是简单直觉，缺点是容易循环、跑偏、调用无关工具。

### 7.2 Plan-and-Execute

先规划，再执行。适合长任务，但计划需要动态修正。

### 7.3 Reflexion

执行后自我反思，再重试。适合代码生成、数据清洗、格式修复等可验证任务。

工程上不要默认上 Agent。能用固定工作流解决的问题，优先用 Chain 或 Graph。

---

## 8. Memory 不是简单聊天记录

Memory 有三种常见形态：

| 类型 | 作用 | 风险 |
|---|---|---|
| 短期上下文 | 当前对话窗口 | 上下文过长、噪声累积 |
| 摘要记忆 | 压缩历史对话 | 摘要丢细节 |
| 长期记忆 | 用户偏好、历史事实、知识片段 | 隐私、污染、过期 |

原则：

- 事实型记忆要可更新、可删除、可追溯来源。
- 偏好型记忆要区分用户明确表达和模型推断。
- 不要把未经验证的模型输出直接写入长期记忆。

---

## 9. 安全与可靠性

LLM 应用的风险主要来自三类输入：

| 来源 | 风险 |
|---|---|
| 用户输入 | prompt injection、越权请求、恶意格式 |
| 检索文档 | 文档中隐藏恶意指令、过期知识、错误事实 |
| 工具输出 | API 错误、数据缺失、外部系统不稳定 |

常见防线：

1. 指令和数据分离：把系统规则、用户输入、检索内容明确分区。
2. 工具白名单：模型只能调用明确暴露的工具。
3. 参数校验：工具调用前校验 schema 和权限。
4. 输出校验：结构化解析、引用检查、业务规则检查。
5. 审计日志：保留 prompt、检索结果、工具调用和最终输出。

---

## 10. 评估框架

LLM 应用不能只靠人工感觉评估。至少要区分四类指标：

| 指标 | 关注点 | 示例 |
|---|---|---|
| 检索质量 | 是否找到了正确证据 | recall@k、MRR、nDCG |
| 生成质量 | 答案是否正确、完整、可读 | faithfulness、answer correctness |
| 系统质量 | 延迟、成本、稳定性 | p95 latency、tokens/request、error rate |
| 安全质量 | 是否越权、泄漏、被注入 | injection pass rate、policy violation rate |

RAG 系统常见排查顺序：

```text
答案错
  ↓
先看是否检索到了正确证据
  ↓
如果没有：修 chunk、embedding、query rewrite、hybrid、rerank
  ↓
如果有：修 prompt、上下文排序、引用约束、输出校验
```

---

## 11. 成本与延迟模型

LLM 应用的成本通常由三部分构成：

```text
总成本 = 检索成本 + 模型 token 成本 + 工具/基础设施成本
```

延迟通常由这些阶段组成：

```text
p95 latency =
  query rewrite
  + retrieval
  + rerank
  + prompt assembly
  + LLM prefill
  + LLM decode
  + output validation
```

优化方向：

- 减少无效上下文，降低 prefill 成本。
- 缓存稳定检索结果和常见回答。
- 对简单问题走小模型或规则路径。
- 对长文档先摘要再检索。
- 对 Agent 设置最大步数、最大工具调用数、超时和降级策略。

---

## 12. 从理论映射到本模块文档

| 理论问题 | 对应文档 |
|---|---|
| RAG 如何把外部知识接入 LLM？ | [01-rag-fundamentals](./01-rag-fundamentals.md) |
| 向量空间、ANN、元数据过滤如何落地？ | [02-vector-databases](./02-vector-databases.md) |
| Chain、Graph、Prompt、结构化输出如何组合？ | [03-orchestration-frameworks](./03-orchestration-frameworks.md) |
| Agent 如何规划、调用工具、处理记忆和安全？ | [04-agent-engineering](./04-agent-engineering.md) |

---

## 13. 工程判断清单

- [ ] 这个需求真的需要 LLM，还是规则/搜索/SQL 就够？
- [ ] 需要外部知识时，知识是稳定的还是频繁更新的？
- [ ] RAG 检索失败时，系统是拒答、降级，还是让模型猜？
- [ ] 检索结果有没有权限过滤和来源追踪？
- [ ] 输出是否需要结构化 schema？
- [ ] 工具调用前是否做参数和权限校验？
- [ ] Agent 是否设置最大步数、超时和成本上限？
- [ ] 是否记录每次请求的 prompt、retrieved chunks、tool calls、final answer？
- [ ] 是否有离线评估集和线上监控指标？
- [ ] 是否区分“检索错”和“生成错”两类问题？

