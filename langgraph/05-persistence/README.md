# 05 · Persistence：线程状态、历史快照与时间旅行

> 目标环境：Python 3.11、LangGraph 1.0.6。  
> 本章完全离线，使用 `InMemorySaver`，不需要数据库、模型或 API Key。

## 学完本章能做什么

你将能够：

1. 解释为什么编译图本身不会自动保留上一次调用的 State。
2. 使用 `InMemorySaver` 和 `thread_id` 为图启用 checkpoint。
3. 让同一 thread 跨多次调用延续状态，同时隔离不同 thread。
4. 使用 `get_state()` 读取最新或指定历史快照。
5. 使用 `get_state_history()` 理解 checkpoint 时间线。
6. 从历史 checkpoint 创建新分支，而不是破坏性覆盖历史。
7. 使用 `update_state()` 创建人工修正 checkpoint。
8. 识别 reducer、外部副作用和持久化之间的常见陷阱。

## 1. 问题：一次 invoke 结束后，下一次从哪里开始

没有 checkpointer 时，每次调用都是独立执行：

```text
invoke #1: input A -> graph -> result A
invoke #2: input B -> graph -> result B
```

第二次调用不会自动看到第一次的最终 State。这对一次性计算没有问题，但以下场景需要延续执行历史：

- 多轮对话要保留消息和轮次。
- 长任务失败后要查看最近保存的状态。
- 人工审批需要暂停后恢复。
- 调试时要查看某一步之前的 State。
- 希望从旧状态尝试另一条路径，而不删除原历史。

LangGraph 使用 **checkpointer** 在图的执行边界保存 State 和执行位置。

## 2. 边界：checkpoint 保存什么，不保存什么

先回顾五个概念：

| 概念 | 本章中的职责 |
|---|---|
| State | 被节点更新的业务数据，是 checkpoint 的主要内容 |
| Runtime Context | 每次调用重新注入的身份和依赖，不自动进入 checkpoint |
| Checkpoint | 某个 thread 在某个执行时刻的 State、待执行节点和元数据 |
| Store | 跨 thread 的长期业务记忆，本章尚未使用 |
| 模型上下文窗口 | 实际传给模型的有限消息；checkpoint 并不会自动帮你裁剪它 |

Checkpoint 不是通用数据库备份。它主要记录图执行所需的状态与位置，不会自动撤销已经发送的邮件、扣款、文件写入或其他外部副作用。

## 3. 心智模型：checkpointer、thread 与 checkpoint

三个术语不要混用：

```text
checkpointer
  └── thread_id = "conversation-42"
        ├── checkpoint A
        ├── checkpoint B
        ├── checkpoint C
        └── checkpoint D (latest)
```

- **checkpointer**：保存和读取 checkpoint 的后端实现。
- **thread**：一条逻辑执行时间线，由 `thread_id` 标识。
- **checkpoint**：时间线中的一个具体快照，有自己的 `checkpoint_id`。

调用只带 `thread_id` 时，LangGraph 从该 thread 的最新 checkpoint 继续。配置中同时包含 `checkpoint_id` 时，可以精确读取或从某个历史点继续。

## 4. 最小持久化配置

### 4.1 创建 saver 并在 compile 时注入

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

`InMemorySaver` 将数据保存在当前 Python 进程内，适合学习和单元测试：

- 无需额外服务。
- 速度快，便于检查。
- 进程退出后全部丢失。
- 不适合多进程共享或生产级耐久存储。

旧资料常使用 `MemorySaver`。在 LangGraph 1.0.6 中它仍是有效别名，但新代码优先使用含义更清晰的 `InMemorySaver`。

### 4.2 每次调用提供 thread_id

```python
config = {
    "configurable": {
        "thread_id": "conversation-42",
    }
}

result = graph.invoke(input_state, config)
```

启用 checkpointer 后，`thread_id` 是必需的执行配置。它位于 `config`，不属于 State，也不是 Runtime Context。

### 4.3 thread_id 不是 user_id

一个用户可能同时拥有多个会话：

```text
user alice
├── thread project-a
├── thread project-b
└── thread support-ticket-9
```

若直接把 `user_id` 当唯一 `thread_id`，本来独立的会话可能意外合并。用户身份适合放在可信 Runtime Context；会话标识适合放在 `config.configurable.thread_id`。

## 5. 示例一：同 thread 延续，不同 thread 隔离

文件：[examples/thread_persistence.py](./examples/thread_persistence.py)

运行：

```bash
conda run -n langgraph python langgraph/05-persistence/examples/thread_persistence.py
conda run -n langgraph python langgraph/05-persistence/examples/thread_persistence.py --thread-id my-demo
conda run -n langgraph python langgraph/05-persistence/examples/thread_persistence.py --self-test
```

图结构：

```text
START -> respond -> END
```

### 5.1 State 使用 reducer 累积消息

```python
class ConversationState(TypedDict, total=False):
    messages: Annotated[list[str], add]
    turn: int
    last_response: str
```

第一次调用 `thread-alice`：

```python
graph.invoke(
    {"messages": ["user: hello"]},
    {"configurable": {"thread_id": "thread-alice"}},
)
```

最终保存：

```text
messages = [user hello, assistant turn 1]
turn = 1
```

第二次对同一 thread 只提供一条新用户消息。checkpointer 取回旧 State，`add` reducer 将新消息追加进去，节点再追加助手消息：

```text
messages = [
  user hello,
  assistant turn 1,
  user remember this,
  assistant turn 2,
]
turn = 2
```

### 5.2 为什么必须复用编译图和 saver

示例先构建一次：

```python
graph = build_graph()
```

然后用同一个 `graph` 执行多次调用。若每个请求都重新创建 `InMemorySaver`，新的 saver 看不到旧内存数据，看起来就像“checkpoint 失效”。

生产中即便使用数据库 saver，也应明确管理连接和应用生命周期，而不是在每个节点中随意重建后端。

### 5.3 不同 thread 完全隔离

自检随后调用 `thread-bob`。它从 turn 1 开始，无法看到 `thread-alice` 的消息。

这说明 checkpoint 的默认隔离维度是 thread，而不是用户。第 06 章会用 Store 演示“同一用户跨 thread 共享长期偏好”。

## 6. 读取当前 StateSnapshot

```python
snapshot = graph.get_state(config)
```

`get_state()` 返回 `StateSnapshot`，不是普通 State 字典。常用属性包括：

| 属性 | 含义 |
|---|---|
| `values` | 该 checkpoint 中的 State 值 |
| `next` | 接下来待执行的节点元组；正常结束通常为空元组 |
| `config` | 可精确定位此 checkpoint 的配置，包含 `checkpoint_id` |
| `metadata` | step、source、writes 等执行元数据 |
| `created_at` | checkpoint 创建时间 |
| `parent_config` | 父 checkpoint 配置，用于理解分支关系 |
| `tasks` | 与该执行步相关的任务信息，包括错误或中断信息 |

读取 State：

```python
current_state = snapshot.values
```

判断是否仍有后续节点：

```python
if snapshot.next:
    print("execution has pending work")
```

不要把 `snapshot.next == ()` 简化成“业务一定成功”。业务错误也可能被节点转换成正常结束 State，仍需检查你自己的状态字段。

## 7. 示例二：历史、分支与人工修正

文件：[examples/history_and_time_travel.py](./examples/history_and_time_travel.py)

运行：

```bash
conda run -n langgraph python langgraph/05-persistence/examples/history_and_time_travel.py
conda run -n langgraph python langgraph/05-persistence/examples/history_and_time_travel.py --self-test
```

示例执行两次增量：

```text
10 + 2 = 12
12 + 5 = 17
```

然后检查历史，从 value 12 的完成 checkpoint 分叉，再执行 `+100`，得到 112。

### 7.1 get_state_history 返回多个 checkpoint

```python
history = list(graph.get_state_history(config))
```

返回顺序是从新到旧。一次 `invoke()` 可能产生多个 checkpoint，因为图会在输入和 superstep 边界保存执行状态。因此：

```text
调用次数 != checkpoint 数量
```

不要写“历史列表第 3 项一定是某业务步骤”这样的脆弱代码。应检查：

- `snapshot.values`
- `snapshot.next`
- `snapshot.metadata`
- 业务自己的稳定标识

示例选择 `value == 12` 且 `next == ()` 的完成快照，而不是依赖列表下标。

### 7.2 时间旅行是分支，不是删除式回滚

从历史点继续：

```python
checkpoint_at_12 = ...
branch = graph.invoke(
    {"delta": 100},
    checkpoint_at_12.config,
)
```

结果为 112，但 value 12 的旧 checkpoint 仍能通过原 `checkpoint_id` 读取。

更准确的心智模型是：

```text
10 -> 12 -> 17        原路径
       \
        -> 112        从历史点创建的新路径
```

“时间旅行”不会自动恢复数据库、消息队列或文件系统到过去。它只让图从历史 State 和执行位置继续。

### 7.3 update_state 创建新 checkpoint

```python
corrected_config = graph.update_state(
    config,
    {
        "value": 40,
        "audit": ["manual correction -> 40"],
    },
)
```

重要细节：

1. `update_state` 不会原地改写旧 checkpoint，而是创建新 checkpoint。
2. 它返回定位新 checkpoint 的 config，应保留这个返回值。
3. State 的 reducer 仍然生效。示例中 `audit` 使用 `add`，因此修正记录会追加，而不是替换整份列表。
4. 对存在待执行节点的图，`as_node` 可能影响 LangGraph 推断下一步；本例在图完成后修正，保持最小用法。

读取修正后的精确状态：

```python
snapshot = graph.get_state(corrected_config)
```

从它开始新调用：

```python
result = graph.invoke({"delta": 2}, corrected_config)
```

得到 42。

## 8. Reducer 与跨调用输入

Checkpointer 恢复旧 State 后，本次新输入也会按照 State 的 channel 规则合并：

```text
saved State + new input + node updates -> new State
```

因此，带 reducer 的字段只应传“本次增量”：

推荐：

```python
graph.invoke({"messages": [new_user_message]}, config)
```

容易重复：

```python
old = graph.get_state(config).values
old["messages"].append(new_user_message)
graph.invoke(old, config)  # 完整旧列表又经过 reducer 合并
```

不要把完整历史 State 当作一次新的增量输入。需要人工修正时优先使用 `update_state()`，并清楚每个字段的 reducer 语义。

## 9. 持久化与可重复执行

Checkpoint 让工作流具备恢复基础，但“可恢复”不自动等于“所有操作恰好执行一次”。节点可能因恢复、重试或时间旅行再次运行。

外部副作用应考虑：

- 使用稳定业务 ID 做幂等键。
- 把读取和写入分开，记录操作结果。
- 在外部系统检查操作是否已完成。
- 不要仅凭节点曾经开始执行，就假设外部写入成功。
- Human-in-the-loop 的 `interrupt()` 之前避免不可重复副作用，因为节点恢复时会从开头重跑。

学习阶段的纯函数示例没有外部副作用，因此可以安全重放。

## 10. InMemorySaver 的边界与生产后端

本章名称是 Persistence，但 `InMemorySaver` 只是实现 checkpoint 协议的内存后端：

```text
同一进程内：可以恢复
进程重启后：数据消失
多个进程间：默认不共享
```

SQLite 和 PostgreSQL saver 由额外包提供，不包含在核心 `langgraph` 包中。选择生产后端时还要考虑：

- 数据库连接生命周期。
- 多实例并发。
- checkpoint 清理和保留期。
- 序列化兼容性。
- 加密、访问控制和敏感数据。
- 同步与异步 saver 是否匹配应用调用方式。

教程核心示例保持离线，不强制安装这些后端。

## 11. 常见坑

### 11.1 启用 checkpointer 却不传 thread_id

没有稳定 thread，checkpointer 无法知道状态属于哪条时间线。每次调用都应提供：

```python
{"configurable": {"thread_id": "..."}}
```

### 11.2 每次请求重建 InMemorySaver

内存 saver 的历史只存在于该实例中。重建实例等于换了一个空后端。

### 11.3 把 user_id 直接当唯一 thread_id

这会把一个用户的多个会话混成一条 State 时间线。thread 表示会话或工作流实例，不等同于用户。

### 11.4 把完整旧 State 再次作为输入

带 reducer 的字段会重复合并。正常新一轮只传本次输入增量；人工修改使用 `update_state`。

### 11.5 认为一次 invoke 只产生一个 checkpoint

Checkpoint 对应执行边界，不对应 API 调用计数。检查 `next` 和 metadata，而不是依赖历史下标。

### 11.6 认为时间旅行会删除未来

历史 checkpoint 仍保留。时间旅行创建分支，不是 `git reset --hard` 式覆盖。

### 11.7 忘记 update_state 也走 reducer

对累积字段传入列表时，它通常会追加。先确认 schema 中每个字段的更新规则。

### 11.8 把密码、token 或客户端放进 State

State 可能被 checkpoint、trace 和日志保存。运行依赖放 Runtime Context，敏感业务数据则要最小化、脱敏和设置保留策略。

### 11.9 把 InMemorySaver 当作生产持久数据库

它在进程退出后丢失全部数据，只用于学习、原型和测试。

## 12. 练习

1. 给 `thread_persistence.py` 增加第三个 thread，验证三个会话互不影响。
2. 将 `messages` reducer 暂时移除，观察第二轮输入为什么覆盖旧历史。
3. 在 history 示例中找到 value 17 的完成 checkpoint，从它分叉执行 `-7`。
4. 使用 `update_state` 把 value 修正为 0，再执行 `+3`；解释 audit 为什么追加。
5. 给 State 增加稳定的 `operation_id`，思考如何用它让外部副作用幂等。
6. 给线程调用同时传 Runtime Context，验证同 thread State 会延续，但 context 必须每次重新提供。

## 13. 自检

先不看答案：

1. `thread_id`、`checkpoint_id` 和 `user_id` 分别标识什么？
2. `get_state()` 返回普通 State 吗？
3. 为什么两次 invoke 可能产生多于两个 checkpoint？
4. 从历史 checkpoint 调用图会删除后来的历史吗？
5. `update_state()` 会绕过 reducer 吗？
6. `InMemorySaver` 能否在 Python 进程重启后恢复？
7. checkpoint 是否会自动撤销已发送的邮件？

答案：

1. thread 是执行时间线，checkpoint 是时间线中的具体快照，user 是业务身份。
2. 不是；返回 `StateSnapshot`，State 在其 `values` 中。
3. 因为 checkpoint 在输入和 superstep 等执行边界保存，不按调用次数一一对应。
4. 不会；它从历史状态创建新的执行分支。
5. 不会；更新仍按 State channel/reducer 规则合并。
6. 不能；它只保存在当前进程内存中。
7. 不会；外部副作用需要业务自己的幂等和补偿设计。

运行机器自检：

```bash
conda run -n langgraph python langgraph/05-persistence/examples/thread_persistence.py --self-test
conda run -n langgraph python langgraph/05-persistence/examples/history_and_time_travel.py --self-test
```

## 14. 本章总结

- Checkpointer 保存 State 和执行位置，thread_id 隔离逻辑时间线。
- 同一 thread 可以跨调用延续 State，不同 thread 默认互相隔离。
- `StateSnapshot` 同时包含 values、next、config、metadata 和任务信息。
- 一次 invoke 可能产生多个 checkpoint，历史默认从新到旧。
- 时间旅行是从旧 checkpoint 分叉，不是删除式回滚。
- `update_state` 创建新 checkpoint，并继续遵循 reducer。
- `InMemorySaver` 适合离线学习，不提供进程重启后的耐久性。
- Checkpoint 不替代外部副作用的幂等、事务与补偿设计。

下一章将加入 Store，把“某条 thread 的执行状态”与“某个用户跨 thread 的长期记忆”彻底分开。
