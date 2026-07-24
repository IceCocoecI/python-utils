# 07 · 工具与 Agent 循环：把模型决策和工具执行接成闭环

> 目标环境：Python 3.11、LangGraph 1.0.6。  
> 本章完全离线，不需要模型、API Key 或网络，也不使用已弃用的 `create_react_agent`。

## 学完本章能做什么

你将能够：

1. 把 Agent 解释为一条可观察、可测试的消息循环，而不是“会自主思考的黑盒”。
2. 读懂 `AIMessage.tool_calls` 与 `ToolMessage.tool_call_id` 之间的协议。
3. 使用 `ToolNode` 执行工具，并用 `tools_condition` 决定继续调用工具还是结束。
4. 在没有真实模型的情况下构造确定性 tool loop 测试。
5. 处理一次模型消息中的多个 tool call。
6. 说清工具参数校验、权限、错误和副作用的安全边界。

## 1. 为什么工具调用需要一张图

一次工具调用至少包含四个动作：

```text
用户问题
   |
   v
模型决定调用什么工具，并给出结构化参数
   |
   v
程序执行工具，把结果写成 ToolMessage
   |
   v
模型读取工具结果，决定继续调用还是生成最终回答
```

真实任务可能不止调用一次工具。例如：

```text
"先算 7 * 6，再加 8"

model -> multiply -> model -> add -> model -> final answer
```

如果只写一次 `model.invoke()` 和一次函数调用，第二次工具选择、错误回传、消息关联和终止条件都会散落在命令式代码里。LangGraph 把这条循环显式表达为：

```text
                 tool_calls 非空
               +-----------------+
               |                 v
START -> model +-------------> tools
          |                       |
          | 无 tool_calls         |
          v                       |
         END <--------------------+
                  回到 model
```

这里的“模型”不一定是网络 LLM。本章用确定性函数生成 `AIMessage`，因此每条路径都可复现；换成真实模型时，图结构不需要改变。

## 2. Agent 循环的最小心智模型

先记住四种角色：

| 角色 | 职责 | 本章使用的类型/API |
|---|---|---|
| 消息状态 | 保存用户、模型和工具之间的协议记录 | `MessagesState` |
| 模型节点 | 生成普通回答或 `tool_calls` | `AIMessage` |
| 工具节点 | 校验参数、调用工具、生成 `ToolMessage` | `ToolNode` |
| 路由条件 | 检查最后一条 AI 消息是否要求工具 | `tools_condition` |

一次完整循环不是“工具函数把值直接返回给模型函数”，而是消息状态不断增加：

```text
HumanMessage
AIMessage(tool_calls=[multiply])
ToolMessage(name=multiply, tool_call_id=call-multiply, content="42")
AIMessage(tool_calls=[add])
ToolMessage(name=add, tool_call_id=call-add, content="50")
AIMessage(content="Final result: 50")
```

`tool_call_id` 很重要。一个 AI 消息可以同时请求多个工具，工具结果必须能对应回原请求。

## 3. 核心 API

| API | 作用 | 需要注意 |
|---|---|---|
| `@tool` | 从 Python 函数生成带名称、描述和参数 schema 的工具 | 类型标注和 docstring 会影响工具协议 |
| `MessagesState` | 提供使用 `add_messages` reducer 的 `messages` 字段 | 节点只返回新消息，不返回完整历史 |
| `AIMessage(tool_calls=...)` | 表示模型希望程序执行工具 | 每个 call 应有稳定的 `id`、`name`、`args` |
| `ToolNode(tools)` | 执行最后一条 AI 消息中的工具调用 | 默认捕获调用协议错误，但工具函数体异常通常继续抛出 |
| `tools_condition(state)` | 有 tool call 时返回 `"tools"`，否则返回 `END` | 默认读取名为 `messages` 的字段 |
| `ToolMessage` | 将工具结果送回消息循环 | `tool_call_id` 必须关联请求 |

本章没有使用 `create_react_agent`。手工搭建循环的价值是能看见每条边、每条消息和每个测试断言；高层封装适合交付应用，不适合第一次理解底层协议。

## 4. 示例一：确定性的两步工具循环

文件：[examples/deterministic_tool_loop.py](./examples/deterministic_tool_loop.py)

运行：

```bash
conda run -n langgraph python langgraph/07-tools-and-agents/examples/deterministic_tool_loop.py
conda run -n langgraph python langgraph/07-tools-and-agents/examples/deterministic_tool_loop.py --self-test
```

预期消息序列：

```text
human: Multiply 7 by 6, then add 8.
ai -> tool calls: multiply({'left': 7, 'right': 6})
tool multiply: 42
ai -> tool calls: add({'left': 42, 'right': 8})
tool add: 50
ai: Final result: 50
```

### 4.1 扩展 MessagesState

```python
class AgentState(MessagesState):
    left: int
    right: int
    increment: int
```

`MessagesState` 已经定义了带 `add_messages` reducer 的 `messages`。示例额外保存三个确定性输入，让“模拟模型”不需要从自然语言中脆弱地解析数字。

这也说明一个重要边界：

- 对话协议放在 `messages`。
- 工作流真正需要的结构化业务值可以放独立字段。
- 不要为了“像聊天”而让所有业务数据都只能从字符串重新解析。

### 4.2 工具首先是普通、可单测的 Python 函数

```python
@tool
def multiply(left: int, right: int) -> int:
    """Multiply two integers."""
    return left * right
```

`@tool` 会根据参数类型生成 schema。真实模型看到的不是 Python 源码，而是工具名称、描述和参数定义。因此：

1. 名称应稳定、明确。
2. docstring 要解释“做什么”，不要只重复函数名。
3. 参数类型要尽量收窄。
4. 工具内部仍要做业务校验、权限检查和超时控制。

### 4.3 模型节点只负责产生消息

示例中的 `deterministic_model` 根据已有 `ToolMessage` 数量决定下一步：

```python
if not tool_messages:
    return {"messages": [AIMessage(tool_calls=[multiply_call])]}

if latest_tool.name == "multiply":
    return {"messages": [AIMessage(tool_calls=[add_call])]}

return {"messages": [AIMessage(content=f"Final result: {total}")]}
```

节点没有直接调用 `multiply()` 或 `add()`。它只生成工具调用协议。执行权仍在程序控制的 `ToolNode` 中。

替换成真实模型时，这个节点通常变成：

```python
def call_model(state):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

图的路由和工具节点保持不变。

### 4.4 ToolNode 执行协议，而不是参与决策

```python
builder.add_node("tools", ToolNode([multiply, add]))
```

`ToolNode` 会：

1. 读取最后一条 AI 消息中的 `tool_calls`。
2. 按名称找到注册工具。
3. 使用 schema 校验和传递参数。
4. 执行工具。
5. 返回与 call ID 对应的 `ToolMessage`。

它不会判断“是否应该调用工具”。这项决策属于模型节点和路由。

### 4.5 tools_condition 形成闭环

```python
builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")
```

`tools_condition` 检查最后一条消息：

- 有 `tool_calls`：返回 `"tools"`。
- 没有 `tool_calls`：返回 `END`。

因此最终普通 `AIMessage` 是循环终止信号。不要再额外用“消息数量达到某个值”猜测是否完成。

### 4.6 自测验证路径，不只验证最终数字

示例断言：

```python
assert [message.type for message in messages] == [
    "human", "ai", "tool", "ai", "tool", "ai"
]
```

并使用 `stream_mode="updates"` 断言节点路径：

```text
model -> tools -> model -> tools -> model
```

只断言最终结果等于 50 不够，因为一个绕过 ToolNode、直接在模型节点计算 50 的错误实现也能通过。

## 5. 示例二：一次消息中的多个工具调用

文件：[examples/parallel_tool_calls.py](./examples/parallel_tool_calls.py)

运行：

```bash
conda run -n langgraph python langgraph/07-tools-and-agents/examples/parallel_tool_calls.py
conda run -n langgraph python langgraph/07-tools-and-agents/examples/parallel_tool_calls.py --self-test
```

示例让模型一次请求：

```python
AIMessage(
    tool_calls=[
        {"name": "word_count", "args": {"text": text}, ...},
        {"name": "unique_words", "args": {"text": text}, ...},
    ]
)
```

`ToolNode` 返回两条 `ToolMessage`，然后模型节点按工具名汇总：

```text
words=4; unique=graph,state,tools
```

这里有两个重要结论：

1. 一次 AI turn 不等于只能调用一个工具。
2. 不要依赖工具结果的偶然完成顺序；应按 `name` 或 `tool_call_id` 建立映射。

如果多个工具会写同一业务资源，还需要在工具层处理并发控制、幂等键或事务；ToolNode 不会自动解决业务冲突。

### 5.1 不能用工具名唯一关联重复调用

本例中两个工具名不同，因此按 `name` 汇总足够直观。如果同一条 AI 消息两次调用同名工具，
`{message.name: message.content}` 会覆盖其中一个结果。通用实现应以唯一的
`ToolMessage.tool_call_id` 建立映射，再按原始 `AIMessage.tool_calls` 的 ID 取回结果。

## 6. 确定性模型与真实模型的替换边界

本章模拟模型不是为了伪装成 LLM，而是为了隔离两个问题：

| 问题 | 确定性示例负责验证 | 接入真实模型后新增风险 |
|---|---|---|
| 图控制流 | model/tools 是否正确循环和结束 | 模型可能选错工具或不结束 |
| 消息协议 | tool call 与 ToolMessage 是否关联 | 厂商模型输出兼容性 |
| 工具实现 | 参数、结果、错误是否可测试 | 外部服务超时、限流、费用 |
| 安全 | 程序是否保留执行控制权 | prompt injection、越权调用 |

可靠做法是先让本章的离线 loop 全部通过，再只替换 `deterministic_model`，而不是同时更换图、工具和模型。

## 7. 工具安全边界

### 7.1 默认错误处理的准确边界

文件：[examples/tool_error_handling.py](./examples/tool_error_handling.py)

运行：

```bash
conda run -n langgraph python langgraph/07-tools-and-agents/examples/tool_error_handling.py
conda run -n langgraph python langgraph/07-tools-and-agents/examples/tool_error_handling.py --self-test
```

LangGraph 1.0.6 中，`ToolNode(tools)` 的默认 handler 有意区分两类错误：

| 错误来源 | 默认行为 | 典型例子 |
|---|---|---|
| 工具调用协议/参数校验 | 返回 `status="error"` 的 `ToolMessage` | 缺少必填参数、参数类型不符合 schema |
| 工具函数体执行 | 异常继续抛给图调用者 | 工具内部主动抛出 `ValueError`、外部服务失败 |

这意味着“ToolNode 默认吞掉所有工具异常”并不正确。若业务确实要把函数体异常反馈给模型，
必须明确配置捕获范围，例如：

```python
ToolNode([guarded_divide], handle_tool_errors=True)
```

`True` 会捕获所有工具执行异常。生产代码通常应进一步收窄到预期异常类型或使用 callable
生成脱敏错误文本；未知编程错误继续抛出，避免 Agent 把系统缺陷当成普通可恢复结果。

示例 self-test 同时验证：缺参数默认生成错误消息、工具体 `ValueError` 默认传播、显式
`handle_tool_errors=True` 后该异常才转换为错误 `ToolMessage`。

模型产生的工具参数是不可信输入。即使模型来自内部服务，也要像处理外部 API 请求一样处理：

- 使用严格参数 schema 和范围校验。
- 在工具内部检查调用者身份和权限。
- 高风险操作进入第 08 章的人工审批流程。
- 对支付、发信、创建工单等副作用使用幂等键。
- 设置超时、重试范围和并发限制。
- 只把必要错误信息返回给模型，敏感堆栈进入受控日志。
- 限制 Agent 最大步数，避免无终止工具循环。

“模型选择了这个工具”不是授权依据。

## 8. 常见坑

### 8.1 模型节点直接执行工具

这样会让图看不到工具执行步骤，也难以统一处理错误、审批和审计。模型节点应生成 tool call，ToolNode 执行。

### 8.2 工具结果使用普通 AIMessage

工具结果应是带 `tool_call_id` 的 `ToolMessage`。否则模型无法可靠地把多个结果对应到多个请求。

### 8.3 返回完整 messages 历史

`MessagesState` 使用 reducer。节点应返回本次新增消息：

```python
return {"messages": [response]}
```

如果返回 `state["messages"] + [response]`，旧消息会被 reducer 再合并，造成重复。

### 8.4 把 tools_condition 接到错误节点

条件应发生在产生 `AIMessage` 的模型节点之后。工具执行后通常固定回到模型节点。

### 8.5 认为 ToolNode 会做业务授权

ToolNode 负责调用协议，不理解租户、余额、审批等级或数据范围。授权必须在工具、上下文或专门的守卫节点中实现。

### 8.6 没有终止保护

真实模型可能持续产生 tool call。生产调用应配置合理的递归限制，并在 state 中记录业务步数或预算。

### 8.7 用最终文本替代路径测试

最终文本正确不代表工具真的被调用。测试至少应断言工具名、参数、ToolMessage 和节点更新顺序。

## 9. 练习

1. 给示例一增加 `subtract` 工具，把任务改成 `(7 * 6 + 8) - 5`，并扩展消息路径断言。
2. 让 `multiply` 拒绝绝对值大于 10,000 的参数：先验证默认 ToolNode 会抛出异常，再配置 `handle_tool_errors=ValueError`，观察错误 `ToolMessage`。
3. 修改示例二，让一个工具返回结构化 JSON，并在模型节点中安全解析。
4. 给工具调用状态增加 `tool_budget`，每次模型 turn 递减，到 0 时强制生成结束消息。
5. 写一个权限守卫：只有 context 中角色为 `finance_admin` 才允许调用转账工具。
6. 把确定性模型替换成你自己的模型客户端，但保持离线 self-test 使用原模型函数。

## 10. 自检

- [ ] 我能画出 `model -> tools -> model` 的完整闭环。
- [ ] 我能解释 `AIMessage.tool_calls`、`ToolMessage` 和 `tool_call_id` 的关系。
- [ ] 我知道 `ToolNode` 负责执行，`tools_condition` 负责路由。
- [ ] 我知道为什么节点只返回新增消息。
- [ ] 我能测试一次连续两步工具调用，而不依赖网络模型。
- [ ] 我知道一次 AI 消息可以包含多个工具调用。
- [ ] 我不会把模型产生的参数当成已经授权的参数。
- [ ] 我能区分默认会转换为 ToolMessage 的调用错误和默认会传播的工具函数体异常。
- [ ] 我能说出最终答案断言之外至少三项路径断言。

## 11. 本章结论

Agent 的核心不是某个高层构造函数，而是一份明确协议：模型产生结构化调用，程序执行受控工具，结果作为消息回到模型，条件边决定循环或结束。

下一章将在这条循环中加入人工控制：当工具具有高风险副作用时，图如何暂停、保存状态、接收审批并从原位置恢复。
