# LangGraph 1.x 速查表

这份文件用于写代码时快速回忆，不替代各章原理说明。

## 1. 最小 StateGraph

```python
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph

class State(TypedDict):
    text: str

def normalize(state: State) -> dict:
    return {"text": state["text"].strip().lower()}

builder = StateGraph(State)
builder.add_node("normalize", normalize)
builder.add_edge(START, "normalize")
builder.add_edge("normalize", END)
graph = builder.compile()

result = graph.invoke({"text": " Hello "})
```

## 2. 独立输入/内部/输出 schema

```python
class InputState(TypedDict):
    query: str

class OverallState(TypedDict):
    query: str
    normalized: str
    answer: str

class OutputState(TypedDict):
    answer: str

builder = StateGraph(
    OverallState,
    input_schema=InputState,
    output_schema=OutputState,
)
```

## 3. Reducer

```python
import operator
from typing import Annotated

class State(TypedDict):
    trace: Annotated[list[str], operator.add]
    score: float  # 默认后写覆盖
```

消息 reducer：

```python
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

## 4. 条件路由

```python
from typing import Literal

def route(state: State) -> Literal["search", "answer"]:
    return "search" if state["needs_search"] else "answer"

builder.add_conditional_edges(
    "classify",
    route,
    {"search": "search", "answer": "answer"},
)
```

## 5. Command：更新并跳转

```python
from typing import Literal
from langgraph.types import Command

def decide(state: State) -> Command[Literal["retry", "finish"]]:
    if state["attempts"] < 2:
        return Command(update={"attempts": state["attempts"] + 1}, goto="retry")
    return Command(goto="finish")
```

## 6. 运行时 Context

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime

@dataclass(frozen=True)
class Context:
    user_id: str
    locale: str = "zh-CN"

def node(state: State, runtime: Runtime[Context]) -> dict:
    return {"owner": runtime.context.user_id}

builder = StateGraph(State, context_schema=Context)
graph = builder.compile()
result = graph.invoke(input_state, context=Context(user_id="u-1"))
```

## 7. Checkpoint 与 thread

```python
from langgraph.checkpoint.memory import InMemorySaver

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "thread-1"}}

first = graph.invoke(first_input, config)
second = graph.invoke(second_input, config)
snapshot = graph.get_state(config)
history = list(graph.get_state_history(config))
```

## 8. Store 长期记忆

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

def remember(state: State, runtime: Runtime[Context]) -> dict:
    namespace = ("users", runtime.context.user_id, "preferences")
    runtime.store.put(namespace, "reply_style", {"value": "concise"})
    item = runtime.store.get(namespace, "reply_style")
    return {"preference": item.value["value"]}

graph = builder.compile(store=store)
```

## 9. Interrupt 与恢复

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

def approval(state: State) -> dict:
    approved = interrupt({"question": "Approve?", "amount": state["amount"]})
    return {"approved": bool(approved)}

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "approval-1"}}

paused = graph.invoke(input_state, config)
assert paused["__interrupt__"]
finished = graph.invoke(Command(resume=True), config)
```

## 10. ToolNode 循环

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

builder.add_node("tools", ToolNode([add]))
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")
```

## 11. 动态 fan-out

```python
import operator
from typing import Annotated
from langgraph.types import Send

class State(TypedDict):
    subjects: list[str]
    summaries: Annotated[list[str], operator.add]

def fan_out(state: State) -> list[Send]:
    return [Send("worker", {"subject": item}) for item in state["subjects"]]

builder.add_conditional_edges("plan", fan_out, ["worker"])
```

## 12. Stream

```python
for event in graph.stream(input_state, stream_mode="updates"):
    print(event)

for mode, event in graph.stream(
    input_state,
    stream_mode=["updates", "custom"],
):
    print(mode, event)
```

节点发送自定义事件：

```python
def node(state: State, runtime: Runtime[Context]) -> dict:
    runtime.stream_writer({"progress": 0.5})
    return {"done": True}
```

## 13. RetryPolicy

```python
from langgraph.types import RetryPolicy

builder.add_node(
    "unstable_io",
    unstable_io,
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=0.1,
        retry_on=ConnectionError,
    ),
)
```

## 14. 常用调试接口

```python
print(graph.get_graph().draw_mermaid())
print(graph.get_state(config).values)
print(graph.get_state(config).next)

for snapshot in graph.get_state_history(config):
    print(snapshot.metadata, snapshot.next, snapshot.values)
```

## 15. 高频错误

```text
InvalidUpdateError
  -> 同一超级步多个节点写同一字段，缺少 reducer

GraphRecursionError
  -> 循环没有终止条件，或 recursion_limit 太低

No checkpointer / missing thread_id
  -> interrupt、状态历史或线程记忆需要 checkpointer + thread_id

恢复后从头执行
  -> 错把完整 state 当新输入；应使用 Command(resume=...)

import langgraph.graph 失败
  -> 本地 langgraph/__init__.py 遮蔽了官方包
```
