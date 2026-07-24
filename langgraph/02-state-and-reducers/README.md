# 02 · 状态与 Reducer：把数据契约和更新语义说清楚

> 目标环境：Python 3.11、LangGraph 1.0.6。  
> 本章完全离线，不需要模型、API Key 或网络。

## 学完本章能做什么

你将能够：

1. 区分输入 schema、图内部 schema 和输出 schema。
2. 使用 `TypedDict` 表达稳定、可读的状态协议。
3. 解释 state 中的一个字段为什么可以看作一个 channel。
4. 区分默认覆盖更新与 reducer 合并更新。
5. 使用 `Annotated[list[T], operator.add]` 累积列表。
6. 使用 `add_messages` 追加、反序列化和按消息 ID 更新消息。
7. 避免“返回完整历史导致重复累积”等常见 reducer 错误。

## 1. 为什么状态设计比节点代码更重要

节点只是工作流中的局部计算，状态才是所有节点共同遵守的协议。如果状态字段含义模糊，图会很快出现以下问题：

- 某个节点期待字符串，前一个节点却写入了列表。
- 内部调试信息被意外暴露给调用者。
- 多个节点写同一个列表，旧内容莫名消失。
- 消息历史无限重复，因为节点返回了“完整历史”而不是“新消息”。
- 并行分支同时写同一个字段，却没有定义如何合并。

因此，设计图时不要先问“有几个节点”，先问：

1. 调用者允许提供什么？
2. 节点之间需要共享什么？
3. 调用者最终应该看到什么？
4. 每个字段收到新值时，是覆盖旧值，还是合并旧值？

## 2. 状态 channel 的心智模型

把每个 state 字段看成一条有独立更新规则的 channel：

```text
State
├── query       : str          -> 默认覆盖
├── answer      : str          -> 默认覆盖
├── steps       : list[str]    -> operator.add 累积
└── messages    : list[Message]-> add_messages 合并
```

节点返回一个 update 后，LangGraph 针对 update 中的每个键分别处理：

```text
没有 reducer：new_state[key] = update[key]

存在 reducer：new_state[key] = reducer(old_state[key], update[key])
```

Reducer 不是“节点执行完之后再跑的节点”，而是字段级状态更新规则。

## 3. 示例一：输入、内部与输出 schema

文件：[examples/input_output_schema.py](./examples/input_output_schema.py)

运行：

```bash
conda run -n langgraph python langgraph/02-state-and-reducers/examples/input_output_schema.py
conda run -n langgraph python langgraph/02-state-and-reducers/examples/input_output_schema.py --self-test
```

### 3.1 三层数据契约

示例定义了三个 `TypedDict`：

```python
class InputState(TypedDict):
    raw_text: str

class OutputState(TypedDict):
    result: str
    word_count: int

class OverallState(InputState, OutputState):
    normalized: str
```

它们分别回答：

| Schema | 谁使用 | 包含什么 |
|---|---|---|
| `InputState` | 图的调用者 | 原始文本 |
| `OverallState` | 图内部所有节点 | 原始文本、中间规范化文本、统计和结果 |
| `OutputState` | 图的调用者 | 最终结果和词数 |

数据流如下：

```text
调用输入                         图内部                          调用输出
{raw_text} -> normalize -> {raw_text, normalized, ...} -> {result, word_count}
                                      ▲
                                      └─ 中间字段不必暴露
```

### 3.2 在 StateGraph 中声明边界

```python
builder = StateGraph(
    OverallState,
    input_schema=InputState,
    output_schema=OutputState,
)
```

第一个位置参数仍是完整内部状态。`input_schema` 限定入口协议，`output_schema` 限定正常完成后返回的字段。

调用：

```python
result = graph.invoke({"raw_text": "LangGraph makes contracts explicit"})
```

得到：

```python
{
    "result": "LANGGRAPH MAKES CONTRACTS EXPLICIT",
    "word_count": 4,
}
```

`raw_text` 和 `normalized` 参与了执行，但不会出现在正常输出中。

### 3.3 为什么节点参数可以使用更窄的类型

第一个节点只需要 `raw_text`：

```python
def normalize(state: InputState) -> dict[str, str]:
    ...
```

后续节点需要内部字段：

```python
def count_words(state: OverallState) -> dict[str, int]:
    ...
```

给函数标注它真正依赖的最小协议，可以减少无意耦合。节点不应因为“状态里还有别的字段”就读取与职责无关的数据。

### 3.4 output_schema 不是安全边界的全部

输出过滤能让公共 API 更干净，但内部 state 仍可能被 checkpointer、trace 或日志保存。密码、令牌等敏感值不应仅依赖 `output_schema` 隐藏；还要考虑是否应进入状态、是否被持久化以及日志脱敏。

## 4. TypedDict 的职责和边界

`TypedDict` 的优势：

- Python 标准类型系统的一部分，依赖轻。
- 非常适合表达“字段名 -> 字段类型”的图状态。
- IDE、mypy、pyright 能检查常见键名和类型错误。
- 与 LangGraph 的 channel/reducer 注解自然配合。

它不负责：

- 自动清洗输入。
- 严格运行时校验。
- 自动补齐所有缺失字段。
- 表达复杂跨字段业务约束。

因此常见组合是：外部 API 边界用 Pydantic 或显式校验，图内部用简洁的 `TypedDict` 表达状态。

## 5. 默认更新：新值覆盖旧值

如果字段没有 reducer，后一次更新覆盖前一次更新。示例二中的 `latest` 就是这样：

```text
""
  -> "draft about reducers"
  -> "reviewed: draft about reducers"
  -> "published: reviewed: draft about reducers"
```

适合覆盖的字段通常包括：

- 当前状态或阶段名。
- 最新草稿。
- 最新评分。
- 最终答案。
- 当前重试次数，由节点返回完整新值。

覆盖不是数据丢失 bug；它是没有 reducer 时的明确语义。

## 6. 示例二：普通 reducer 与 add_messages

文件：[examples/reducers_and_messages.py](./examples/reducers_and_messages.py)

运行：

```bash
conda run -n langgraph python langgraph/02-state-and-reducers/examples/reducers_and_messages.py
conda run -n langgraph python langgraph/02-state-and-reducers/examples/reducers_and_messages.py --self-test
```

示例在同一张图中对比三种字段：

```python
class WorkflowState(TypedDict):
    topic: str
    latest: str
    steps: Annotated[list[str], add]
    messages: Annotated[list[AnyMessage], add_messages]
```

### 6.1 Annotated 把类型与更新规则放在一起

```python
steps: Annotated[list[str], add]
```

这里：

- `list[str]` 是字段的数据类型。
- `operator.add` 是 reducer。

每个节点只返回本次新增步骤：

```python
return {"steps": ["draft"]}
```

三个节点执行后得到：

```python
["draft", "review", "publish"]
```

不要在节点里写成：

```python
return {"steps": state["steps"] + ["draft"]}
```

因为 reducer 还会再做一次 `old + update`，旧内容会重复。

### 6.2 add_messages 不只是 list.add

消息状态使用：

```python
messages: Annotated[list[AnyMessage], add_messages]
```

`add_messages` 具备消息领域语义：

1. 新 ID 的消息通常追加到列表。
2. 相同 ID 的新消息会更新已有消息，而不是重复追加。
3. 它能处理 LangChain message 对象，并支持常见消息字典的反序列化。

示例先放入用户消息：

```python
HumanMessage(content="Write about reducers", id="request")
```

三个节点都返回 ID 为 `answer` 的 AI 消息：

```text
draft(answer) -> reviewed(answer) -> published(answer)
```

最终消息数是 2，而不是 4：

```text
request: HumanMessage
answer : AIMessage with final published content
```

这正是编辑、人工修正或重放消息时需要的行为。

### 6.3 三种更新放在一起看

| 字段 | 注解 | 节点每次返回 | 最终行为 |
|---|---|---|---|
| `latest` | `str` | 完整新字符串 | 后值覆盖前值 |
| `steps` | `Annotated[list[str], add]` | 本次新增列表 | 所有步骤依次累积 |
| `messages` | `Annotated[list[AnyMessage], add_messages]` | 新增或替换消息 | 新 ID 追加，同 ID 更新 |

## 7. 如何选择 reducer

先问业务语义，不要先问“哪个 API 最方便”：

| 问题 | 推荐语义 |
|---|---|
| 只需要最新值吗？ | 默认覆盖 |
| 需要保留所有事件且顺序明确吗？ | 列表追加 reducer |
| 需要累计数字吗？ | 加法 reducer，但明确输入是增量还是总量 |
| 需要按键合并字典吗？ | 编写有明确定义的自定义 reducer |
| 是对话消息，且可能编辑/替换吗？ | `add_messages` |

一个好的 reducer 应尽量满足：

- 输入输出类型稳定。
- 合并语义容易解释。
- 不原地修改旧值。
- 在可能并行的场景中考虑顺序和结合律。

## 8. 常见坑

### 8.1 在 reducer 字段中返回完整历史

旧值已经由 LangGraph 持有。节点应返回增量，否则会重复：

```text
old = [a, b]
update = [a, b, c]
reducer(old, update) = [a, b, a, b, c]
```

### 8.2 在节点中原地 append

```python
state["steps"].append("review")
return {"steps": state["steps"]}
```

这同时引入共享可变对象和重复合并问题。应返回一个新的增量列表。

### 8.3 把 add_messages 当成纯追加

相同消息 ID 会替换对应消息。如果希望保留两个独立事件，要给它们不同 ID；如果希望编辑旧消息，则必须稳定复用 ID。

### 8.4 消息没有稳定 ID，却期待覆盖

没有相同 ID，reducer 无法知道两条消息代表同一个逻辑对象，结果通常是追加。

### 8.5 多个并行节点写无 reducer 的同一字段

LangGraph 无法凭空猜测如何合并冲突更新。并行章节中会看到，这类情况通常需要 reducer、拆分 channel，或重新设计汇聚节点。

### 8.6 把“状态字段多”误认为“上下文越完整越好”

状态越大，持久化、序列化和调试成本越高。只保留后续节点确实需要的数据；临时局部变量留在节点内部。

## 9. 练习

1. 给 `input_output_schema.py` 增加内部字段 `unique_words`，但不要让它出现在最终输出。
2. 给 `OutputState` 增加 `is_empty`，并为普通输入和空输入补断言。
3. 在 reducer 示例中给 `steps` 的每一项增加序号，确认节点仍只返回增量。
4. 将三个 AI 消息改成不同 ID，观察最终消息数量。
5. 编写一个 reducer，将两个 `set[str]` 合并，并验证重复标签只保留一次。
6. 故意返回 `state["steps"] + ["publish"]`，用输出解释重复发生的原因，然后修正。

## 10. 自检

- [ ] 我能解释输入、内部和输出 schema 分别服务谁。
- [ ] 我知道 `output_schema` 会过滤正常返回字段，但不能替代敏感数据治理。
- [ ] 我能说出没有 reducer 时字段如何更新。
- [ ] 我能解释 `Annotated[list[str], add]` 中两部分分别表示什么。
- [ ] 我知道 reducer 字段的节点通常应返回增量。
- [ ] 我能解释 `add_messages` 与普通列表拼接的关键区别。
- [ ] 我知道消息 ID 何时应稳定复用，何时应生成新值。
- [ ] 我能为一个新字段写出明确的覆盖或合并语义。

## 11. 下一章

状态和更新规则明确之后，下一步是让执行路径不再固定。第 03 章将用条件边实现分支，用条件边与 `Command` 实现有界循环，并讨论如何避免无限循环。
