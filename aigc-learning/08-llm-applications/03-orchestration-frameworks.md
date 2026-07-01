# 03 · 编排框架

> 单次 LLM 调用只能完成简单问答，真实产品需要把检索、推理、工具调用串成可靠的工作流。
> LangChain、LlamaIndex、LangGraph 是当前最主流的三大编排框架。
> 框架 API 变化较快，本文重点讲稳定的抽象和工程取舍；具体代码以对应版本官方文档为准。

---

## 1. 为什么需要编排框架？

一个典型的 LLM 应用远不止一次 API 调用：

```
用户提问
  → 改写查询（query rewrite）
  → 检索文档（retrieval）
  → 重排序（reranking）
  → 构建提示词（prompt assembly）
  → LLM 推理（generation）
  → 结构化输出解析（parsing）
  → 工具调用（tool use）
  → 记忆更新（memory）
  → 返回结果
```

手写这些管道代码会变成"胶水代码地狱"。编排框架提供：
- 统一的组件抽象（LLM、Retriever、Tool、Memory）
- 可组合的链式 / 图式调用
- 内置的错误处理和重试
- 可观测性（tracing、logging）

但编排框架不是越多越好。生产系统优先保证：

| 能力 | 为什么重要 |
|---|---|
| 显式状态 | 知道每一步输入、输出和失败原因 |
| 可恢复 | 长任务中断后能从 checkpoint 继续 |
| 可观测 | trace 能定位检索、模型、工具、校验哪一步出错 |
| 可测试 | 每个节点可以单测，整条链可以离线回归 |
| 可降级 | 模型/API/工具失败时有 fallback |

---

## 2. LangChain

### 2.1 核心概念

```
┌─────────────────────────────────────────────────┐
│                  LangChain                       │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Chat Model│  │ Prompt   │  │ Output   │      │
│  │          │  │ Template │  │ Parser   │      │
│  └──────────┘  └──────────┘  └──────────┘      │
│         │            │             │             │
│         └────────────┼─────────────┘             │
│                      ↓                           │
│              LCEL (链式组合)                      │
│                      ↓                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Retriever│  │ Memory   │  │ Tools    │      │
│  └──────────┘  └──────────┘  └──────────┘      │
└─────────────────────────────────────────────────┘
```

### 2.2 LCEL（LangChain Expression Language）

LCEL 是 LangChain 的核心——用 `|` 管道符把组件串起来：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位 Python 专家，回答简洁准确。"),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()

answer = chain.invoke({"question": "什么是装饰器？"})
print(answer)

# 流式输出
for chunk in chain.stream({"question": "解释 asyncio"}):
    print(chunk, end="", flush=True)

# 批量调用
answers = chain.batch([
    {"question": "什么是 GIL？"},
    {"question": "什么是协程？"},
])
```

### 2.3 Chat Model

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

gpt = ChatOpenAI(model="gpt-4o-mini")
claude = ChatAnthropic(model="claude-sonnet-4-20250514")

messages = [
    SystemMessage(content="你是 AI 助手"),
    HumanMessage(content="你好"),
]
response = gpt.invoke(messages)
print(response.content)
```

### 2.4 Prompt Template

```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# 基础模板
prompt = ChatPromptTemplate.from_template("翻译成英文：{text}")

# Few-shot 模板
examples = [
    {"input": "你好", "output": "Hello"},
    {"input": "谢谢", "output": "Thank you"},
]
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位翻译专家。"),
    few_shot_prompt,
    ("human", "{input}"),
])
```

### 2.5 RAG Chain

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local("my_index", embeddings,
                                allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
    "根据以下上下文回答问题。\n上下文：{context}\n问题：{question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("什么是注意力机制？")
```

---

## 3. LlamaIndex

LlamaIndex 的哲学是"以数据为中心"——专注于把你的数据变成 LLM 可查询的知识库。

### 3.1 核心概念

```
Documents → Nodes → Index → QueryEngine → Response
```

### 3.2 快速上手

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

documents = SimpleDirectoryReader("data/").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("什么是 RAG？")
print(response)
print(response.source_nodes)  # 溯源
```

### 3.3 自定义 Node 解析

```python
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents)

md_parser = MarkdownNodeParser()
md_nodes = md_parser.get_nodes_from_documents(documents)
```

### 3.4 LlamaIndex vs LangChain

| 维度 | LangChain | LlamaIndex |
|---|---|---|
| 定位 | 通用 LLM 编排框架 | 数据索引 + 查询框架 |
| 核心抽象 | Chain / Runnable | Index / QueryEngine |
| RAG | 需要手动组装 | 开箱即用 |
| Agent | LangGraph（强） | 有但较弱 |
| 灵活性 | 极高（啥都能组合） | RAG 场景极方便 |
| 学习曲线 | 较陡 | 较平缓 |
| 建议 | 复杂工作流 / Agent | 快速搭建 RAG 原型 |

### 3.5 什么时候不用框架？

如果应用只有：

- 一个固定 prompt；
- 一个模型调用；
- 少量规则后处理；
- 没有工具、检索、状态和多分支；

直接写普通 Python 代码更清晰。框架适合复杂度已经出现时使用，不适合为了“看起来像 AI 工程”而引入。

---

## 4. LangGraph：状态机驱动的 Agent 框架

LangGraph 是 LangChain 团队专门为**多步骤有状态工作流**开发的框架。

### 4.1 核心概念

```
StateGraph = 节点（Node）+ 边（Edge）+ 状态（State）

Node: 执行某个操作（调用 LLM、检索、工具调用）
Edge: 定义节点间的跳转逻辑
State: 在节点间共享的数据（TypedDict）
```

LangGraph 的核心价值不只是“能循环”，还包括 durable execution、checkpoint、human-in-the-loop、streaming 和可观测状态。
这使它更适合长任务、工具调用和需要人工审批的流程。

### 4.2 一个简单的 Agent

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(model="gpt-4o-mini")

def chatbot(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

app = graph.compile()

result = app.invoke({"messages": [HumanMessage(content="你好！")]})
print(result["messages"][-1].content)
```

### 4.3 带工具调用的 Agent

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import ast
import operator

@tool
def search_web(query: str) -> str:
    """搜索互联网获取最新信息"""
    return f"搜索结果：关于 '{query}' 的最新信息..."

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](eval_node(node.operand))
        raise ValueError("Only simple arithmetic expressions are allowed")

    tree = ast.parse(expression, mode="eval")
    return str(eval_node(tree))

tools = [search_web, calculate]
llm_with_tools = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def agent(state: State) -> State:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

app = graph.compile()

result = app.invoke({
    "messages": [HumanMessage(content="帮我算一下 (15 * 32) + 78")]
})
print(result["messages"][-1].content)
```

### 4.4 条件边与循环

```python
def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"      # 有工具调用 → 执行工具
    return END              # 没有 → 结束

graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    END: END,
})
```

这就是 LangGraph 的核心范式：**条件边实现循环，让 Agent 可以多轮推理**。

### 4.5 状态设计原则

LangGraph 好不好维护，很大程度取决于 State 设计。

| 字段 | 用途 |
|---|---|
| messages | 对话和工具调用轨迹 |
| intent | 当前任务类型 |
| retrieved_docs | 检索证据 |
| tool_results | 工具执行结果 |
| plan / current_step | 多步任务进度 |
| budget | 剩余 token、工具次数、时间 |
| errors | 可恢复错误和重试记录 |
| approvals | 人工审批状态 |

原则：

- 不要把所有东西都塞进 `messages`；
- 可审计状态和临时 scratchpad 分开；
- 高风险工具调用前写入 pending approval；
- 每个节点只读写自己负责的字段；
- 状态要可序列化，方便 checkpoint 和重放。

---

## 5. LangSmith：可观测性

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "my-rag-app"

# 之后所有 LangChain / LangGraph 调用自动上报 trace
# 在 https://smith.langchain.com 查看：
#   - 每次调用的输入/输出
#   - token 用量和延迟
#   - 链路中每个节点的执行详情
```

---

## 6. 提示工程模式

### 6.1 Few-Shot

```python
prompt = """根据示例回答问题。

问：Python 的 GIL 是什么？
答：GIL（Global Interpreter Lock）是 CPython 中的全局解释器锁，确保同一时刻只有一个线程执行 Python 字节码。

问：什么是装饰器？
答：装饰器是一种用 @语法 修改函数/类行为的设计模式，本质上是一个接受函数并返回函数的高阶函数。

问：{question}
答："""
```

### 6.2 Chain-of-Thought（思维链）

```python
prompt = """请一步步思考后回答问题。

问题：一个书包里有 3 个红球和 5 个蓝球，随机取 2 个球，两个都是红球的概率是多少？

让我们一步步来思考：
"""
```

### 6.3 ReAct（Reasoning + Acting）

```
思考：用户问的是最新的 Python 版本，我需要搜索一下。
行动：search("latest Python version 2026")
观察：Python 3.13 于 2024 年 10 月发布...
思考：搜索结果显示 Python 3.13，让我确认是否有更新。
行动：search("Python 3.14 release date")
观察：Python 3.14 预计 2025 年 10 月发布...
最终回答：截至 2026 年，Python 最新稳定版本是 3.14。
```

---

## 7. 结构化输出

### 7.1 OpenAI Function Calling / Tools

```python
from openai import OpenAI

client = OpenAI()

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
)

tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name)       # get_weather
print(tool_call.function.arguments)  # {"city": "北京", "unit": "celsius"}
```

> OpenAI 生态正在从 Chat Completions / Assistants 逐步迁移到 Responses API 和 Agents SDK。
> 学习重点是 tool schema、参数校验、结构化输出和审计，不要把某个接口形态当成长期不变的架构。

### 7.2 Pydantic + LangChain 结构化输出

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class MovieReview(BaseModel):
    title: str = Field(description="电影名称")
    rating: float = Field(description="评分 1-10")
    summary: str = Field(description="一句话点评")
    recommended: bool = Field(description="是否推荐")

llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm = llm.with_structured_output(MovieReview)

review = structured_llm.invoke("评价一下电影《星际穿越》")
print(f"片名: {review.title}")
print(f"评分: {review.rating}")
print(f"点评: {review.summary}")
print(f"推荐: {review.recommended}")
```

### 7.3 结构化输出失败恢复

结构化输出仍然可能失败：

| 失败 | 处理 |
|---|---|
| JSON 无法解析 | 自动重试，附上解析错误 |
| 字段缺失 | schema validator 指出缺失字段 |
| 枚举越界 | 要求模型在允许值中重选 |
| 数值越界 | 程序侧 clamp 或拒绝 |
| 业务规则不满足 | 进入修复节点或人工审核 |

生产上建议：

```text
LLM output → parse → schema validate → business validate
  ├─ pass → downstream
  └─ fail → repair prompt / retry / fallback / human review
```

不要把 Pydantic 解析成功等同于业务正确。

---

## 8. 缓存策略

### 8.1 精确匹配缓存

```python
from langchain_community.cache import InMemoryCache
from langchain_core.globals import set_llm_cache

set_llm_cache(InMemoryCache())

# 第一次调用：走 API
result1 = llm.invoke("什么是 Python？")  # ~1s
# 第二次调用同样的输入：走缓存
result2 = llm.invoke("什么是 Python？")  # ~0ms
```

### 8.2 语义缓存

```python
from langchain_community.cache import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings

set_llm_cache(RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OpenAIEmbeddings(),
    score_threshold=0.95,
))
# "什么是 Python？" 和 "Python 是什么？" 会命中同一缓存
```

---

## 9. 成本优化

### 9.1 Token 计算

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")
text = "你好，世界！Hello, World!"
tokens = enc.encode(text)
print(f"Token 数: {len(tokens)}")

# 估算成本
input_cost_per_1k = 0.00015   # gpt-4o-mini
output_cost_per_1k = 0.0006
estimated_cost = len(tokens) / 1000 * input_cost_per_1k
```

### 9.2 模型路由

```python
def route_model(query: str) -> str:
    """简单查询用便宜模型，复杂查询用强模型"""
    if len(query) < 50 and "?" not in query:
        return "gpt-4o-mini"    # 便宜
    return "gpt-4o"             # 贵但强

# 更智能的方案：用小模型给查询打复杂度分
```

### 9.3 三大省钱手段

```
1. 缓存：精确缓存 + 语义缓存，避免重复调用
2. 模型路由：简单任务用小模型，难题用大模型
3. Prompt 优化：减少冗余上下文，压缩检索结果
```

### 9.4 成本预算写进链路

复杂工作流要把预算放进 state：

```text
max_input_tokens
max_output_tokens
max_tool_calls
max_retries
timeout_seconds
max_cost_usd
```

每个节点执行前检查预算，超过预算就降级、总结当前进度或请求用户确认。
否则 Agent/Graph 很容易在异常路径上把成本放大。

---

## 10. 常见陷阱

| 陷阱 | 后果 | 解决方案 |
|---|---|---|
| 过度工程化 | 简单任务用了复杂的框架 | 评估是否真的需要 LangChain |
| Prompt Injection | 用户输入篡改系统指令 | 输入清洗、角色隔离、输出验证 |
| 上下文窗口溢出 | 检索塞了太多内容导致截断 | 限制 top-k、用 reranker 精选 |
| 不做 tracing | 出问题了不知道哪步错 | 用 LangSmith 或类似工具 |
| 同步调用导致慢 | 串行调 LLM 延迟叠加 | 用 LCEL 的 `batch` / `abatch` |
| 忽略错误处理 | LLM 返回格式不对就崩溃 | 加 retry、fallback、输出校验 |
| 框架升级不兼容 | LangChain API 频繁变化 | 锁版本、关注 changelog |
| 不做成本监控 | 月底账单吓一跳 | 记录每次调用的 token 用量 |
| 状态设计混乱 | 后续节点读不到或误读上下文 | 明确定义 State schema |
| 结构化输出只做解析 | JSON 合法但业务错误 | 增加业务 validator |
| 工具无幂等设计 | 重试导致重复下单/写入 | idempotency key + 审计 |

---

## 11. 实践任务：把 RAG Chain 改成 Graph

把一个简单 RAG 改写成显式状态图：

```text
query
  → classify_intent
  → retrieve
  → rerank
  → generate_answer
  → validate_citation
  → final / ask_clarification / refuse
```

State 至少包含：

```python
class RAGState(TypedDict):
    question: str
    intent: str
    retrieved_docs: list
    answer: str
    citations: list
    errors: list
```

记录每个节点的输入输出、耗时和 token 用量，然后用 20 条问题做回归测试。

---

## 12. 本章小结

```
框架选择：
  快速 RAG 原型 → LlamaIndex
  通用 LLM 编排 → LangChain + LCEL
  多步骤有状态 Agent → LangGraph
  生产监控 → LangSmith

核心技能：
  1. LCEL 管道组合：prompt | llm | parser
  2. 结构化输出：Function Calling + Pydantic
  3. 提示工程：Few-Shot / CoT / ReAct
  4. 成本控制：缓存 + 路由 + Prompt 优化
  5. 可观测性：tracing + logging + 成本监控
```

下一章我们进入最前沿的领域——Agent 工程。
