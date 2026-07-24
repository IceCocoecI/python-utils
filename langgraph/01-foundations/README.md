# 01 · 基础：从函数调用到第一个 StateGraph

> 目标环境：Python 3.11、LangGraph 1.0.6。  
> 本章完全离线，不需要模型、API Key 或网络。

## 学完本章能做什么

你将能够：

1. 解释 LangGraph 为什么不是“把几个函数装进列表”。
2. 用 `TypedDict` 定义图的共享状态。
3. 把普通 Python 函数注册为节点，用边连接执行顺序。
4. 区分 builder 与编译后的 graph。
5. 使用 `invoke()` 获取最终状态，使用 `stream(..., stream_mode="updates")` 观察节点增量。
6. 读懂 Mermaid 图定义，并能定位一个最小图中的状态流转。

## 1. LangGraph 解决的到底是什么问题

普通函数调用很适合固定、短小的流程：

```python
cleaned = normalize(raw)
answer = transform(cleaned)
result = format_output(answer)
```

当流程开始出现以下需求时，普通嵌套调用会逐渐变难维护：

- 中间状态需要被多个步骤共享。
- 下一步由运行时结果决定。
- 某一步可能循环、重试或暂停。
- 希望流式观察每个步骤的输出。
- 希望以后增加持久化、人工审批或子图，而不重写业务函数。

LangGraph 把这个问题表达为“状态上的图计算”：

```text
输入
  │
  ▼
START -> node_a -> node_b -> node_c -> END
           │          │
           └── 读取并更新同一个 State ──┘
```

图负责“谁在什么时候执行”，状态负责“步骤之间交换什么数据”，节点负责“这一步具体做什么”。

## 2. 最小心智模型

先记住五个角色：

| 角色 | 含义 | 最小 API |
|---|---|---|
| State | 整张图共享的数据协议 | `TypedDict` |
| Node | 接收当前状态、返回增量更新的函数 | `def node(state): ...` |
| Edge | 从一个节点到另一个节点的执行关系 | `add_edge()` |
| START / END | 图的虚拟入口与出口 | `START`、`END` |
| Compiled graph | 真正可运行的图 | `compile()` |

一次节点执行可以近似理解为：

```text
current_state
    │
    ▼
node(current_state) -> update
                         │
                         ▼
             merge(current_state, update)
                         │
                         ▼
                    next_state
```

关键点是：节点通常返回“本节点改变的字段”，不是手工复制整份状态。

## 3. 示例一：第一个线性图

文件：[examples/hello_state_graph.py](./examples/hello_state_graph.py)

运行：

```bash
conda run -n langgraph python langgraph/01-foundations/examples/hello_state_graph.py
conda run -n langgraph python langgraph/01-foundations/examples/hello_state_graph.py --self-test
```

图结构：

```text
START -> normalize -> greet -> measure -> END
```

### 3.1 用 TypedDict 描述共享状态

```python
class GreetingState(TypedDict):
    name: str
    normalized_name: str
    greeting: str
    greeting_length: int
```

`TypedDict` 描述字典中允许出现的键和值类型。它的作用主要是：

- 让 IDE 和类型检查器理解状态结构。
- 让读代码的人一眼看到图中流动的数据。
- 让 LangGraph 知道各 channel 应如何管理。

它不会像 Pydantic 一样自动做严格的运行时校验。输入是否为空等业务约束，仍应在节点或输入边界处理。

### 3.2 节点读取状态，返回局部更新

```python
def normalize_name(state: GreetingState) -> dict[str, str]:
    normalized = " ".join(state["name"].strip().split())
    if not normalized:
        raise ValueError("name must not be empty")
    return {"normalized_name": normalized}
```

这里发生了三件事：

1. 节点读取输入 channel `name`。
2. 普通 Python 代码完成清洗和校验。
3. 节点只返回 `normalized_name` 的更新。

LangGraph 不要求节点里调用 LLM。节点可以是任意确定性函数、数据库调用、工具调用或模型调用。本教程先用纯函数，是为了把“图机制”和“模型行为”分开。

### 3.3 注册节点和边

```python
builder = StateGraph(GreetingState)
builder.add_node("normalize", normalize_name)
builder.add_node("greet", build_greeting)
builder.add_node("measure", measure_greeting)

builder.add_edge(START, "normalize")
builder.add_edge("normalize", "greet")
builder.add_edge("greet", "measure")
builder.add_edge("measure", END)
```

节点名是图内标识符。边使用节点名，而不是直接传函数。这样图结构可以被检查、可视化和追踪。

### 3.4 builder 不能直接当作应用运行

```python
graph = builder.compile()
result = graph.invoke({"name": "Ada Lovelace"})
```

`StateGraph(...)` 返回的是构建器。`compile()` 会检查图结构并生成可执行对象。后续的 `invoke`、`stream`、持久化配置等都发生在编译后的对象上。

可以把二者类比为：

```text
builder        = 流程设计图
compiled graph = 可执行程序
```

### 3.5 为什么输入只传了 name

示例调用时只传：

```python
graph.invoke({"name": "Ada Lovelace"})
```

`normalized_name`、`greeting` 和 `greeting_length` 会由后续节点创建。`TypedDict` 的完整状态描述不意味着入口必须手工填满所有中间字段。第 02 章会进一步用独立 input/output schema 把入口和出口协议写得更精确。

## 4. 示例二：invoke、stream 与图检查

文件：[examples/execution_modes.py](./examples/execution_modes.py)

运行：

```bash
conda run -n langgraph python langgraph/01-foundations/examples/execution_modes.py
conda run -n langgraph python langgraph/01-foundations/examples/execution_modes.py --self-test
```

图结构：

```text
START -> normalize -> tokenize -> summarize -> END
```

### 4.1 invoke 返回最终合并状态

```python
result = graph.invoke({"text": text})
```

节点依次执行后，`invoke()` 返回最终状态。适合：

- 调用者只关心最终结果。
- 工作流很快，不需要实时展示进度。
- 测试中需要断言最终状态。

### 4.2 stream_mode="updates" 返回节点增量

```python
for update in graph.stream({"text": text}, stream_mode="updates"):
    print(update)
```

输出形状类似：

```python
{"normalize": {"normalized": "langgraph keeps state visible"}}
{"tokenize": {"tokens": ["langgraph", "keeps", "state", "visible"]}}
{"summarize": {"summary": "count=4; first=langgraph; last=visible"}}
```

外层键是刚执行的节点名，内层字典是该节点返回的增量。这对学习和调试非常有用：如果最终结果错误，可以先确定错误从哪个节点开始出现。

`updates` 不是每一步的完整状态快照。需要完整状态时可以探索 `stream_mode="values"`，但生产系统中完整状态可能很大，频繁传输会增加开销。

### 4.3 不联网也能查看 Mermaid 定义

```python
mermaid = graph.get_graph().draw_mermaid()
```

这会返回 Mermaid 文本，不要求联网。它适合代码审查和文档记录。渲染成图片可能需要额外依赖或服务，本章不依赖该能力。

## 5. 核心 API 速查

| API | 作用 | 常见误解 |
|---|---|---|
| `StateGraph(State)` | 创建图构建器 | 它还不是可执行图 |
| `add_node(name, fn)` | 注册节点 | 节点名应稳定且唯一 |
| `add_edge(a, b)` | 添加固定边 | 不负责条件判断 |
| `compile()` | 检查并编译图 | 通常在构建完成后调用一次 |
| `invoke(input)` | 同步运行并返回最终输出 | 不是逐节点输出 |
| `stream(input, stream_mode="updates")` | 流式返回节点更新 | update 不等于完整 state |
| `get_graph()` | 获取结构表示 | 不会执行工作流 |

## 6. 常见坑

### 6.1 节点直接修改传入的 state

不推荐：

```python
def node(state):
    state["result"] = "done"
    return state
```

推荐返回局部更新：

```python
def node(state):
    return {"result": "done"}
```

局部更新更容易测试，也能让 reducer、并行执行和追踪保持清晰。

### 6.2 忘记连接 START 或 END

没有入口，图不知道从哪里开始；没有合理出口，图可能无法结束。最小图至少要形成一条从 `START` 可达 `END` 的路径。

### 6.3 把节点执行顺序写进节点函数

节点不应直接调用“下一个节点”。执行关系应该写在边中，否则图结构无法反映真实控制流。

### 6.4 以为 TypedDict 会自动校验外部输入

`TypedDict` 是类型协议，不是完整的运行时校验器。空字符串、数值范围和跨字段约束仍需显式处理。

### 6.5 在节点中依赖随机数或当前时间，却不记录它们

这会让测试与恢复执行变得不可重复。学习阶段优先使用纯函数；必须使用外部不确定性时，要考虑如何注入、记录和测试。

## 7. 练习

1. 在 `hello_state_graph.py` 中增加 `title` 输入，并生成 `Hello, Dr. Ada Lovelace!`。
2. 给线性图增加一个节点，将问候语转成大写；用 `stream` 确认它位于 `greet` 与 `measure` 之间。
3. 将 `execution_modes.py` 的摘要改成同时输出最长 token。
4. 分别运行 `stream_mode="updates"` 和 `stream_mode="values"`，比较事件形状。
5. 故意删除一条边，观察编译或运行时行为，并解释为什么图无法到达预期节点。

## 8. 自检

- [ ] 我能用一句话区分 State、Node 和 Edge。
- [ ] 我知道节点返回的是状态更新，而不是必须返回完整状态。
- [ ] 我能解释 `START` 和 `END` 为什么不是普通业务节点。
- [ ] 我能区分 builder 与 compiled graph。
- [ ] 我能说明 `invoke()` 和 `stream(..., stream_mode="updates")` 的输出差异。
- [ ] 我能在不运行外部模型的情况下写出并测试一个三节点图。
- [ ] 我能从流式更新中判断哪个节点产生了错误数据。

## 9. 下一章

本章把状态当作普通共享字典。下一章将回答两个更重要的问题：

1. 入口、内部状态和对外输出是否必须使用同一份 schema？
2. 多个节点连续更新同一字段时，为什么有时覆盖、有时需要累积或按消息 ID 合并？
