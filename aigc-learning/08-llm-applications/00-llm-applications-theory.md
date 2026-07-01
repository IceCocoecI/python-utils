# 00 · LLM 应用理论框架

> 本文是模块 08 的理论底座。
> 后续 RAG、向量数据库、编排框架、Agent 工程都可以看成同一个系统问题：
> 如何稳定、可控、可评估地把 LLM 接入真实业务环境。

> 本文按 `2026-06-30` 的主流工程实践组织。框架和 API 更新很快，具体代码以官方文档为准；
> 本文重点建立不会快速过时的系统判断方法。

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

### 1.1 生产级 LLM 应用六件套

| 层 | 要解决的问题 | 常见技术 |
|---|---|---|
| 输入理解 | 用户到底要问答、搜索、生成、执行还是修改？ | 意图分类、路由、权限识别 |
| 上下文构造 | 模型应该看见哪些事实、历史、检索结果和工具输出？ | RAG、memory、context compression |
| 行动系统 | 模型能调用哪些外部能力？ | function calling、MCP、HTTP API、SQL、code sandbox |
| 输出约束 | 回答是否符合 schema、引用、业务规则和安全策略？ | structured output、validators、guardrails |
| 可观测性 | 失败发生在哪一步？成本和延迟在哪里？ | tracing、logs、eval、metrics |
| 运营闭环 | 线上失败如何进入数据集并改进系统？ | human review、golden set、A/B test、feedback |

很多失败并不是模型能力不够，而是这六层中的某一层缺失：

- 没有输入路由，所有问题都走最贵模型；
- 没有权限过滤，RAG 可能泄漏跨租户数据；
- 没有输出校验，JSON 格式飘一次就让业务流程崩溃；
- 没有 tracing，线上错误无法复现；
- 没有 eval，prompt 改动靠感觉上线。

---

## 2. RAG 和 Agent 为什么是重要转向？

基座模型把大量知识压进参数，但参数有几个天然限制：

- 训练后知识不会自动更新。
- 私有业务数据通常不在训练集中。
- 模型无法直接执行数据库查询、下单、发邮件、改代码等外部动作。
- 上下文窗口有限，不能无成本装下所有资料。
- 模型输出是概率生成，不能天然保证引用、权限和业务规则正确。

RAG 和 Agent 的意义是把 LLM 从“封闭参数模型”扩展成“可接入环境的系统”：

```text
参数知识 + 检索证据 + 工具执行 + 状态管理 + 评估校验
```

这是一件重要转向，因为真实应用往往不是单轮问答，而是要在不断变化的业务环境中给出可追溯、可执行、可恢复的结果。

但难点也随之转移：

| 技术 | 解决的问题 | 新的难点 |
|---|---|---|
| RAG | 把外部知识带入上下文 | 切分、召回、重排、引用、权限、证据不足时拒答 |
| Tool Use | 让模型调用外部能力 | schema 设计、参数校验、权限控制、失败恢复 |
| Agent | 让模型动态规划和执行多步任务 | 循环失控、成本不可预测、难评估、工具误用 |
| Memory | 跨轮次保留状态 | 隐私、过期、污染、错误写入长期记忆 |

所以工程判断不是“要不要上 Agent”，而是：

1. 固定流程能解决的，用 Chain 或 Graph。
2. 需要外部事实的，先做可评估的 RAG。
3. 需要外部动作的，先做受限工具调用。
4. 只有路径确实动态、且有校验闭环时，再使用 Agent。

### 2.1 深度解读：RAG 不是长 prompt，Agent 不是万能自动化

RAG 的本质不是把更多资料塞进 prompt，而是把外部知识转化成**可验证上下文**。
一个 RAG 系统是否可靠，主要取决于三件事：

1. 是否召回了正确证据。
2. 是否把噪声和无关片段压下去。
3. 是否能让最终答案追溯到证据。

如果检索错了，模型再强也只是在错误上下文上生成流畅回答。
如果 chunk 太大，模型会被无关内容干扰；如果 chunk 太小，关键上下文会断裂。
如果没有引用、拒答和后处理校验，RAG 很容易从“减少幻觉”变成“带着引用幻觉”。

Agent 的本质也不是“让模型自由发挥”，而是把模型放进一个受控循环：

```text
Observe → Plan / Decide → Act → Verify → Update State
```

Agent 的能力来自工具、状态、反馈和校验，而不只是更长的思考文本。
它适合路径动态、需要多步探索、每一步可以被观测或验证的任务。
如果流程本来固定，例如“检索 → 重排 → 总结 → 输出 JSON”，用 Graph 或普通工作流通常更稳、更便宜、更可测。

工程上判断 RAG/Agent 是否用对，可以看失败时能否定位责任：

| 失败现象 | 优先排查 |
|---|---|
| 答案没有依据 | 检索召回、chunk、rerank、上下文构造 |
| 引用和答案不一致 | 引用约束、生成后校验、证据片段粒度 |
| Agent 循环或跑偏 | 工具白名单、步数上限、状态设计、停止条件 |
| 成本不可控 | 路由、缓存、最大工具调用次数、降级策略 |

### 2.2 Chain、Graph、Agent 的选择顺序

优先级应该是从确定性到开放性：

| 方案 | 适合 | 不适合 |
|---|---|---|
| 普通代码 / SQL / 搜索 | 规则明确、可直接计算或查询 | 需要自然语言理解和生成 |
| Prompt + LLM | 单步生成、分类、改写、摘要 | 需要外部事实或动作 |
| RAG Chain | 外部知识问答、引用、文档总结 | 需要多步工具执行 |
| Graph | 固定流程、分支、循环、人审、失败恢复 | 路径完全开放且无法预定义 |
| Agent | 动态探索、多工具、多步任务 | 高风险动作、强确定性流程、低延迟场景 |

这条顺序能减少过度工程化。能用 Graph 明确状态和分支时，不要急着让 Agent 自由规划。

---

## 3. LLM 应用的基本系统模型

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

### 3.1 失败归因比“整体分数”更重要

LLM 应用要把失败拆开看：

| 失败层 | 例子 | 修复方向 |
|---|---|---|
| 路由失败 | 简单问题走了 Agent，成本高且慢 | 意图分类、规则路由、小模型路由 |
| 检索失败 | 正确文档没召回 | chunk、embedding、hybrid、query rewrite |
| 上下文失败 | 召回了证据，但排序差、太长、引用乱 | rerank、context packing、去重、压缩 |
| 生成失败 | 证据正确但答案幻觉或格式错 | prompt、structured output、validators |
| 工具失败 | 参数错、权限不足、外部 API 失败 | schema、校验、重试、补偿事务 |
| 安全失败 | prompt injection、越权、数据泄漏 | 指令/数据隔离、权限过滤、人审 |

一个成熟系统的日志应该能支持这种归因，而不是只记录最终回答。

---

## 4. 五个核心对象

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

### 4.1 Context 不是越多越好

上下文窗口变长不等于可以无脑塞资料。长上下文会带来：

- prefill 成本上升；
- 注意力分散；
- 重要证据被噪声淹没；
- 引用更难验证；
- prompt injection 面积变大。

工程上要做 context budget：

```text
system / developer instructions
+ user query
+ conversation history
+ retrieved evidence
+ tool results
+ output schema
<= model context window 和成本预算
```

RAG 的目标是放入“足够且最相关”的证据，而不是放入“最多”的证据。

---

## 5. RAG 的理论链路

RAG（Retrieval-Augmented Generation）可以拆成离线和在线两条链路。

```text
离线索引：
Document → Parse → Chunk → Embed → Index

在线回答：
Query → Rewrite → Retrieve → Rerank → Build Context → Generate → Verify
```

### 5.1 Chunking 是信息压缩问题

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

### 5.2 Embedding 是语义空间映射

Embedding 模型把文本映射到向量空间。相似问题应该在向量空间中距离更近。

常见距离：

| 距离 | 含义 | 适合场景 |
|---|---|---|
| Cosine | 比较方向，相当于语义角度 | 文本 embedding 最常用 |
| Inner Product | 点积，相似度和向量长度都参与 | 归一化后等价于 cosine 排序 |
| L2 | 欧氏距离 | 图像、聚类、部分 ANN 索引 |

关键约束：**索引和查询必须使用同一个 embedding 空间**。
混用不同模型、不同归一化策略、不同语言预处理，会让检索质量直接失效。

### 5.3 Retrieval 是召回与精度的权衡

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

### 5.4 Generation 必须基于证据

RAG 的目标不是“让回答更像参考资料”，而是让回答**可追溯到证据**。

实用约束：

- 让模型只基于给定上下文回答。
- 输出引用来源或 chunk id。
- 当证据不足时明确拒答。
- 对数值、日期、法规、合同条款做后处理校验。

### 5.5 RAG 的权限和版本

生产 RAG 必须把权限和版本作为一等公民：

| 问题 | 风险 | 建议 |
|---|---|---|
| 检索后再过滤权限 | top-k 结果可能全被过滤，召回质量骤降 | 尽量检索前过滤，或按租户/权限分索引 |
| 文档更新后不重建索引 | 答案引用旧政策 | 记录 document_version 和 index_version |
| 删除文档但向量仍存在 | 数据合规风险 | 支持 tombstone、重建、删除审计 |
| chunk 缺少来源 | 无法追责和引用 | 保存 source、page、section、offset、hash |
| 多数据源混合 | 权威性冲突 | 给来源加 authority 和 freshness 权重 |

---

## 6. 向量数据库的本质

向量数据库解决三个问题：

1. 存储向量和元数据。
2. 快速近似最近邻搜索（ANN）。
3. 支持过滤、更新、分片、权限、持久化。

### 6.1 ANN 为什么是近似的？

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

### 6.2 元数据过滤很重要

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

## 7. 编排框架解决什么？

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

### 7.1 生产编排还要考虑什么？

| 能力 | 为什么重要 |
|---|---|
| durable execution | 长任务中断后可恢复，避免重复执行高风险工具 |
| checkpoint / state | 能审计每一步，也能从中间节点重跑 |
| human-in-the-loop | 高风险工具调用、低置信度答案需要人工确认 |
| streaming | 用户体验更好，也能早发现长任务跑偏 |
| retry / fallback | LLM/API/检索/工具都可能失败 |
| tracing | 线上失败能复现和归因 |

---

## 8. Agent 的理论模型

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

### 8.1 ReAct

ReAct 把推理和行动交替进行：

```text
Thought → Action → Observation → Thought → ...
```

优点是简单直觉，缺点是容易循环、跑偏、调用无关工具。

### 8.2 Plan-and-Execute

先规划，再执行。适合长任务，但计划需要动态修正。

### 8.3 Reflexion

执行后自我反思，再重试。适合代码生成、数据清洗、格式修复等可验证任务。

工程上不要默认上 Agent。能用固定工作流解决的问题，优先用 Chain 或 Graph。

---

## 9. Memory 不是简单聊天记录

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

## 10. 安全与可靠性

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

### 10.1 工具安全边界

给模型工具之前，先给工具分级：

| 风险 | 例子 | 控制 |
|---|---|---|
| 只读低风险 | 搜索公开文档、读取无敏感知识库 | 日志 + 速率限制 |
| 只读敏感 | 查询用户资料、财务数据、内部文档 | 权限校验 + 脱敏 + 审计 |
| 写入可回滚 | 创建草稿、写临时文件、生成工单 | 幂等 key + 人审可选 |
| 写入不可逆 | 发邮件、下单、删除数据、转账 | 强制人审 + 二次确认 + 补偿流程 |
| 代码执行 | Python/bash/browser 自动化 | 沙箱 + 网络/文件限制 + 超时 |

工具设计原则：

- schema 越窄越好，不暴露万能接口；
- 参数必须校验，不信任模型生成的 JSON；
- 高风险工具默认 dry-run；
- 写操作要有 idempotency key；
- 每次调用记录 user、tool、args、result、approval、trace_id。

---

## 11. 评估框架

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

### 11.1 最小可用评估集

上线前至少准备一套 golden set：

| 字段 | 说明 |
|---|---|
| question | 用户真实问题或高频变体 |
| expected_answer | 标准答案或评分 rubric |
| evidence_ids | 应该召回的文档/chunk |
| user_role / tenant | 权限测试 |
| freshness | 是否依赖最新数据 |
| unsafe_variant | prompt injection / 越权版本 |
| expected_behavior | 回答、拒答、调用工具或转人工 |

评估时同时看：

- retrieval recall@k；
- rerank 后 context precision；
- answer faithfulness；
- citation accuracy；
- schema valid rate；
- tool success rate；
- p95 latency；
- cost/request；
- unsafe pass rate。

### 11.2 线上监控指标

| 类别 | 指标 |
|---|---|
| 质量 | thumbs up/down、人工抽检通过率、引用命中率 |
| 检索 | empty retrieval rate、top-k 分布、reranker score 分布 |
| 模型 | input/output tokens、refusal rate、schema retry rate |
| 工具 | tool error rate、approval rate、timeout rate |
| 性能 | p50/p95/p99 latency、TTFT、queue time |
| 成本 | cost/request、cache hit rate、agent steps/request |
| 安全 | injection detected、policy violation、cross-tenant block |

---

## 12. 成本与延迟模型

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

### 12.1 成本优化的正确顺序

1. 先做日志，知道钱花在哪里。
2. 再减无效上下文，降低 prefill。
3. 加缓存，处理重复问题。
4. 做模型路由，让简单任务走小模型。
5. 对 Agent 加步数和工具调用预算。
6. 最后再考虑复杂的 prompt 压缩、蒸馏和微调。

不要在没有 token、延迟和成功率数据时盲目优化。

---

## 13. 从理论映射到本模块文档

| 理论问题 | 对应文档 |
|---|---|
| RAG 如何把外部知识接入 LLM？ | [01-rag-fundamentals](./01-rag-fundamentals.md) |
| 向量空间、ANN、元数据过滤如何落地？ | [02-vector-databases](./02-vector-databases.md) |
| Chain、Graph、Prompt、结构化输出如何组合？ | [03-orchestration-frameworks](./03-orchestration-frameworks.md) |
| Agent 如何规划、调用工具、处理记忆和安全？ | [04-agent-engineering](./04-agent-engineering.md) |

---

## 14. 工程判断清单

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
