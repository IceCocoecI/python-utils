# LangGraph 概念地图

这份文档回答“为什么这样设计”。API 的最短写法见 [CHEATSHEET.md](./CHEATSHEET.md)，逐步代码见各章节。

## 1. LangGraph 真正解决什么问题

普通函数足以表达确定性的短流程：

```python
result = format_result(search(parse(user_input)))
```

当流程同时出现下面几类要求时，命令式代码会迅速变得难以恢复和观测：

- 模型可能选择不同工具或反复调用工具；
- 工作流存在循环、动态并行和多个参与者；
- 进程中断后需要从已完成步骤恢复；
- 人需要在中间查看、批准或修改状态；
- 对话需要线程级短期记忆和用户级长期记忆；
- 调用者需要流式看到中间事件；
- 团队需要测试“走了哪条路径”，而不只是最终文本。

LangGraph 把这些要求压缩为四个核心元素：

```text
State: 现在知道什么
Node:  这一小步做什么
Edge:  下一步去哪
Runtime: 如何执行、保存、观察和恢复
```

它是编排层，不负责替你决定 prompt、模型、检索器或业务规则。

## 2. 一次图执行发生了什么

假设两个并行节点都从同一个 state 快照开始：

```text
superstep N 的 State
       | copied view
       +------------------+
       v                  v
    node_a              node_b
       | {items:[A]}       | {items:[B]}
       +---------+---------+
                 v
          reducer(items)
                 |
                 v
        superstep N+1 State
```

关键点：

1. 节点返回的是“更新”，不是原地修改后的完整状态。
2. 同一超级步中的节点通常看不到彼此刚产生的更新。
3. 多个更新写同一字段时，必须有 reducer；否则会产生并发更新错误。
4. 边只决定调度，不传数据；数据通过 state channel 流动。
5. checkpoint 保存的是某个执行时刻的状态和调度元数据，不只是聊天记录。

这解释了为什么 `state["items"].append(...)` 是危险写法，也解释了为什么 reducer 不只是“列表拼接小技巧”。

## 3. State：业务事实，不是对象仓库

好的 state 字段具备三个属性：

- 可序列化：checkpoint 后端能够保存；
- 可解释：看到快照就能理解工作流为什么走到这里；
- 有明确合并语义：覆盖、追加、去重、取最大值或其他规则。

常见字段类型：

| 类别 | 示例 | 更新语义 |
|---|---|---|
| 输入事实 | `question`, `ticket_id` | 通常只写一次 |
| 中间决策 | `intent`, `risk_level` | 后写覆盖 |
| 累积记录 | `messages`, `trace`, `evidence` | reducer 合并 |
| 控制计数 | `attempts`, `remaining_steps` | 显式计算后覆盖 |
| 最终结果 | `answer`, `resolution` | 最终节点覆盖 |

不要放入 state：数据库连接、HTTP client、模型实例、锁、文件句柄和函数。它们属于 context、闭包或应用容器。

## 4. Reducer：字段级代数

未标注 reducer 的字段使用覆盖语义。标注形式：

```python
class State(TypedDict):
    trace: Annotated[list[str], operator.add]
```

Reducer 的本质是：给定旧值 `a` 和更新 `b`，得到新值 `r(a, b)`。并行场景中，好的 reducer 应尽可能满足：

- 封闭性：输入输出类型一致；
- 结合律：`r(r(a,b),c) == r(a,r(b,c))`；
- 明确的顺序需求：若不满足交换律，就不要假设并行结果天然有业务顺序；
- 不修改输入：返回新对象，避免跨节点共享可变引用。

`add_messages` 不等于简单列表相加。它按 message id 追加或替换消息，因此可以更新已有消息，也能接受 LangChain message 对象或兼容字典。

## 5. Edge、Router、Command 与 Send

四者解决不同层次的问题：

| 机制 | 适合 | 特点 |
|---|---|---|
| 普通 edge | 固定顺序 | 拓扑最清晰 |
| conditional edges | 根据 state 选择下一节点 | 路由函数应纯净、易测 |
| `Command` | 节点既更新 state 又决定跳转 | 适合决策与结果不可分的节点 |
| `Send` | 运行时动态创建多个节点调用 | 适合 map-reduce / fan-out |

循环必须有可证明的终止条件，例如最大尝试次数。不要只期待模型“最终会停”。

## 6. Context 与 configurable

1.x 推荐用 `context_schema` 描述运行时上下文：

```python
builder = StateGraph(State, context_schema=Context)
result = graph.invoke(input_state, context=Context(user_id="u-1"))
```

节点通过 `Runtime[Context]` 读取。Context 的适合内容：

- user/tenant identity；
- 权限和 feature flags；
- 模型或服务依赖；
- 一次调用固定的阈值；
- Store 访问所需的命名空间信息。

`configurable.thread_id` 仍用于 checkpoint 线程标识。它不是业务上下文的通用垃圾桶：

```text
context.user_id              -> 这个请求代表谁
configurable.thread_id       -> checkpoint 写到哪条执行线程
```

同一用户可以有多个 thread；同一 thread 通常不应跨用户复用。

## 7. Checkpoint 与短期记忆

编译图时提供 checkpointer：

```text
graph.compile(checkpointer=saver)
                  |
                  v
每个 superstep 保存 state + next tasks + metadata
```

再次使用相同 `thread_id` 调用时，输入更新会接到该线程当前状态上。这形成线程级短期记忆。

Checkpoint 还支持：

- `get_state`：读取当前快照；
- `get_state_history`：查看历史；
- `update_state`：人工修正状态；
- 以历史 checkpoint config 再执行：分叉/时间旅行；
- interrupt 后恢复。

时间旅行通常创建新的执行分支，不会真的抹掉历史。外部副作用也不会自动回滚。

## 8. Store 与长期记忆

Store 与 checkpoint 的差异：

| 维度 | Checkpointer | Store |
|---|---|---|
| 主键思想 | thread + checkpoint | namespace + key |
| 主要目的 | 恢复执行、线程状态 | 跨线程共享信息 |
| 写入方式 | 运行时自动 | 节点显式读写 |
| 典型数据 | 当前对话、步骤、interrupt | 用户偏好、长期事实 |

推荐 namespace 至少包含数据类型和身份：

```python
namespace = ("users", user_id, "preferences")
```

长期记忆需要治理：来源、置信度、更新时间、TTL、用户删除、敏感信息和冲突合并。把所有聊天内容无条件写入 Store 不是记忆系统。

## 9. Interrupt 与恢复

节点调用 `interrupt(payload)` 后：

```text
执行到节点 -> 保存 checkpoint -> 返回 __interrupt__
                                      |
                         Command(resume=value)
                                      |
                                      v
                         从节点中断位置重新进入
```

恢复时必须传 `Command(resume=...)` 和同一个 `thread_id`。把完整 state 再次传给 `invoke` 是一个新输入，可能从 `START` 重跑并重复副作用。

Interrupt 所在节点会从头重新执行，直到 `interrupt()` 返回恢复值。因此 interrupt 之前的代码必须确定且可重放；不可避免的副作用应放到 interrupt 之后或独立幂等节点。

## 10. 工具 Agent 的循环

典型低层循环：

```text
        +----------------------+
        |                      |
        v                      |
     model node --tool_calls--> ToolNode
        |
        +--no tool calls------------> END
```

必须分开考虑：

1. 模型是否“想调用”工具；
2. tool call 的名称和参数是否合法；
3. 当前用户是否有权限；
4. 工具是否有外部副作用；
5. 结果是否需要截断、脱敏或转换；
6. 工具失败后是重试、回到模型还是人工处理。

`ToolNode` 执行协议，不替代业务授权和风险控制。

## 11. 子图、多智能体与动态并行

不要把“多个节点”都叫多智能体。常见模式：

| 模式 | 何时使用 | 主要代价 |
|---|---|---|
| 单 Agent + tools | 一个决策者即可 | 工具和上下文可能变多 |
| Router | 分类后进入固定专家 | 路由错误会选错专家 |
| Supervisor | 中央角色反复委派 | 额外模型调用和上下文传递 |
| Handoff | 当前 Agent 把控制权交给另一个 | 状态所有权更难定义 |
| 并行 fan-out | 子任务可独立处理 | reducer、顺序、部分失败 |
| Subgraph | 子流程需要封装或独立持久化 | schema 映射和观测更复杂 |

先问“是否真的存在独立上下文、工具权限或生命周期”，再决定是否拆成 Agent。

## 12. Durable execution 不等于事务

Checkpoint 可以让计算重放，但外部世界不会跟着回滚：

```text
节点调用支付 API成功 -> 进程在 checkpoint 前崩溃
恢复后节点重放       -> 可能再次支付
```

可靠节点应采用一种或多种策略：

- 幂等键：以 workflow/thread/task id 去重；
- 读后写检查：先查询外部结果；
- outbox / saga：显式记录事务阶段和补偿动作；
- 把非确定性和副作用拆进单独 task；
- 对高风险动作使用 interrupt 审批。

## 13. 如何测试图

测试金字塔：

```text
少量真实模型/外部后端集成测试
          ^
图行为测试：最终 state、路径、interrupt、恢复、线程隔离
          ^
纯函数测试：reducer、router、validator、namespace
```

最稳定的断言对象是结构化 state 和事件，不是自然语言逐字匹配。真实模型测试应与离线协议测试分开。

## 14. 选择 LangGraph 的判断表

| 问题 | 是 | 否 |
|---|---|---|
| 有循环、动态路由或多个参与者吗？ | 继续评估 | 普通 pipeline 可能足够 |
| 需要 checkpoint、暂停恢复或时间旅行吗？ | LangGraph 很合适 | 普通函数仍可能足够 |
| 需要线程级/跨线程记忆吗？ | LangGraph 提供统一运行时 | 可用普通数据库封装 |
| 需要精确控制模型、工具和业务步骤吗？ | 低层 Graph API 合适 | 高层 Agent 可能更省事 |
| 团队愿意维护 state schema、路由和恢复测试吗？ | 可以工程化落地 | 不要为了“Agent”标签增加复杂度 |

学习每一章时始终追问三件事：它解决了什么瓶颈；它引入什么失败模式；如何观测、验证并控制这些失败模式。
