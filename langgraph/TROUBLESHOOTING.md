# 排障手册

先确认命令从仓库根目录执行，并使用正确环境：

```bash
conda run -n langgraph python -c "import sys, importlib.metadata as m; print(sys.executable); print(m.version('langgraph'))"
```

## 1. `ModuleNotFoundError: No module named 'langgraph.graph'`

检查实际导入位置：

```bash
conda run -n langgraph python -c "import langgraph; print(list(langgraph.__path__))"
```

正常结果应包含 Conda 环境的 `site-packages/langgraph`。若只看到本地课程目录，检查是否误建了 `langgraph/__init__.py` 或把课程目录设置成了错误的 source root。

## 2. 默认环境能运行 Python，但导入不了 LangGraph

仓库默认 shell 可能激活的是其他 Conda 环境。使用：

```bash
conda activate langgraph
```

或直接：

```bash
conda run -n langgraph python langgraph/01-foundations/examples/hello_state_graph.py
```

## 3. `InvalidUpdateError` / 同一字段并发写入

症状：从一个节点 fan-out 到多个节点后，多个节点都返回同一 key。

原因：字段没有定义 reducer，LangGraph 不知道应该覆盖、拼接还是聚合。

修复：

```python
import operator
from typing import Annotated

class State(TypedDict):
    results: Annotated[list[str], operator.add]
```

不要靠节点原地 `append` 共享列表，这会破坏状态更新模型。

## 4. `GraphRecursionError`

通常不是“把 recursion_limit 调大”就结束。先检查：

- router 是否存在到 `END` 的分支；
- attempts/remaining_steps 是否真的更新；
- 判断条件是否使用更新后的字段；
- 工具循环中的模拟模型是否在收到 `ToolMessage` 后停止发 tool call。

确认循环合理后才调整：

```python
config = {"recursion_limit": 50}
graph.invoke(input_state, config)
```

## 5. 使用 checkpointer 时提示缺少配置键

至少提供 `thread_id`：

```python
config = {"configurable": {"thread_id": "thread-1"}}
graph.invoke(input_state, config)
```

不同会话必须使用不同 thread id；不要把固定字符串用于所有用户。

## 6. 同一个 thread 的消息重复

常见原因：第二轮输入包含了完整历史，而 checkpointer 又已经保存历史。

第二轮只提交新增输入：

```python
graph.invoke({"messages": [new_human_message]}, config)
```

不要再次提交第一轮的所有消息。

## 7. Interrupt 能暂停，但无法恢复

检查三件事：

1. 图编译时是否提供 checkpointer；
2. 恢复时是否使用相同 `thread_id`；
3. 是否传入 `Command(resume=value)`。

正确形式：

```python
graph.invoke(Command(resume={"approved": True}), config)
```

错误形式：

```python
graph.invoke(full_state_with_approval, config)
```

后一种是新输入，可能从 START 重跑。

## 8. Interrupt 之前的日志或副作用执行两次

Interrupt 恢复时会重新进入所在节点。将副作用：

- 移到 `interrupt()` 之后；或
- 拆为下一个独立节点；或
- 使用幂等键和外部去重。

不要把打印次数直接等同于图从头重跑，可通过 state history 和 trace 字段确认实际路径。

## 9. Store 中找不到记忆

检查：

- 编译时是否 `compile(store=store)`；
- 节点是否通过 `runtime.store` 访问；
- 写入和读取 namespace 是否完全一致；
- `context.user_id` 是否一致；
- 是否错误地为每次 invoke 新建了一个 Store 实例。

教学中的 `InMemoryStore` 在进程退出后会清空，这是预期行为。

## 10. ToolNode 执行后图无限循环

模拟或真实模型收到 `ToolMessage` 后必须生成不带 `tool_calls` 的最终 `AIMessage`。若每次都发同一个 tool call，`tools_condition` 会持续回到工具节点。

同时给循环设置硬上限，不把停止完全交给模型。

## 11. ToolNode 报工具名称或参数错误

检查生成的 tool call：

```python
AIMessage(
    content="",
    tool_calls=[{
        "name": "add",
        "args": {"a": 2, "b": 3},
        "id": "call-1",
        "type": "tool_call",
    }],
)
```

名称要和注册工具一致，参数要满足工具 schema，每个调用需要唯一 id。

## 12. 时间旅行重复了外部操作

Checkpoint 只能恢复图状态，不能回滚外部系统。时间旅行、retry 和崩溃恢复都可能重放节点。给邮件、支付、工单写入等操作增加业务幂等键，并将结果写回 state。

## 13. Stream 看不到想要的信息

选择合适模式：

| 模式 | 看到什么 |
|---|---|
| `values` | 每一步合并后的完整状态 |
| `updates` | 每个节点产生的增量更新 |
| `custom` | 节点通过 stream writer 发出的事件 |
| `messages` | 模型消息/token 事件，依赖模型集成 |
| `debug` | 更详细的调试事件 |

多模式流返回 `(mode, data)`。子图事件还需要 `subgraphs=True`。

## 14. 示例失败时的最小诊断顺序

```bash
conda run -n langgraph python --version
conda run -n langgraph python -c "import importlib.metadata as m; print(m.version('langgraph')); print(m.version('langchain-core'))"
conda run -n langgraph python langgraph/scripts/smoke_test.py --verbose
conda run -n langgraph python -m unittest discover -s langgraph/tests -v
```

若只一个示例失败，直接运行该脚本的 `--self-test`，保留完整 traceback，再对照对应章节的“常见坑”。
