# 08 · Human-in-the-loop：暂停、审阅、修改与正确恢复

> 目标环境：Python 3.11、LangGraph 1.0.6。  
> 本章完全离线，重点验证恢复语义，不调用模型或外部服务。

## 学完本章能做什么

你将能够：

1. 解释为什么中断必须配合 checkpointer 和 `thread_id`。
2. 使用 `interrupt(value)` 暂停节点，并用 `Command(resume=value)` 恢复。
3. 从返回值和 `get_state()` 中读取中断信息与下一节点。
4. 使用 `update_state()` 在静态断点修改 checkpoint 中的状态。
5. 区分“恢复旧运行”和“把完整状态作为新输入重新运行”。
6. 设计可重放、无重复副作用的人工审批节点。

## 1. 为什么 Agent 需要人工介入

工具循环能自动完成很多任务，但以下决策不应只交给模型：

- 转账、退款、删除和发布等不可逆操作。
- 模型置信度低或证据冲突。
- 需要用户补充缺失参数。
- 生成内容必须经过法务、运营或专家审阅。
- 需要人在执行前修改状态，而不是只回答“同意/拒绝”。

Human-in-the-loop 不是在终端里临时写一个 `input()`。真正需要的是：

```text
运行到安全暂停点
      |
      v
持久化当前状态、待执行任务和中断载荷
      |
      v
进程可以结束，人工稍后查看和提交决定
      |
      v
使用同一个 thread_id 从中断点继续
```

如果状态只保存在 Python 调用栈里，服务重启或请求返回后就无法恢复。

## 2. 最小心智模型

一次动态中断可以理解为两阶段执行：

```text
阶段一
START -> prepare -> approval node
                        |
                        +-- interrupt(payload)
                                |
                                v
                       checkpoint + 返回控制权

阶段二
Command(resume=decision)
            |
            v
重新进入 approval node，interrupt() 返回 decision
            |
            v
route -> execute / reject -> END
```

这里有一个必须记住的规则：

> 动态 `interrupt()` 恢复时，包含它的节点会从函数开头重新执行；不是从 Python 函数的下一行恢复调用栈。

因此 `interrupt()` 之前的代码必须可安全重放。发送邮件、扣款等副作用应放在审批后的独立节点，或使用幂等键保护。

## 3. 三类操作不要混淆

| 操作 | 用途 | 输入形状 |
|---|---|---|
| `interrupt(payload)` | 节点主动请求外部输入 | payload 会暴露给调用者 |
| `Command(resume=value)` | 给暂停中的动态 interrupt 提供返回值 | value 成为 `interrupt()` 的返回值 |
| `update_state(config, update)` | 直接修改已 checkpoint 的 state channel | update 按 channel/reducer 规则合并 |
| `invoke(None, config)` | 从静态断点或已保存的下一步继续 | `None` 表示没有提交一份新的图输入 |

最危险的误解是：

```python
# 错误的“恢复”思路
graph.invoke(full_updated_state, same_config)
```

非 `None` 输入通常代表一次新的图输入，会从入口重新调度。它可能重跑模型、重新生成候选项、重复外部调用，然后再次撞上同一个中断点。

## 4. 中断为什么需要 checkpointer 和 thread_id

编译时配置：

```python
graph = builder.compile(checkpointer=InMemorySaver())
```

调用时配置：

```python
config = {"configurable": {"thread_id": "approval-123"}}
```

二者分工如下：

- checkpointer 决定状态、任务和 checkpoint 保存在哪里。
- `thread_id` 决定本次调用属于哪条执行线程。
- 恢复时必须使用同一个 compiled graph/checkpointer 和同一个 `thread_id`。

本章使用 `InMemorySaver`，仅适合教学和单进程测试。进程退出后数据丢失；生产系统需要持久化实现，并设计权限、加密、TTL 和迁移。

## 5. 示例一：动态审批与 Command(resume)

文件：[examples/interrupt_approval.py](./examples/interrupt_approval.py)

运行：

```bash
conda run -n langgraph python langgraph/08-human-in-the-loop/examples/interrupt_approval.py
conda run -n langgraph python langgraph/08-human-in-the-loop/examples/interrupt_approval.py --reject
conda run -n langgraph python langgraph/08-human-in-the-loop/examples/interrupt_approval.py --self-test
```

图结构：

```text
START -> prepare -> approval
                        |
                  interrupt(payload)
                        |
                Command(resume=decision)
                        |
                   route_decision
                     /       \
                execute     reject
                     \       /
                        END
```

### 5.1 prepare 只准备计划，不执行副作用

```python
def prepare_request(state):
    plan = f"transfer {state['amount']} credits for {state['request']}"
    return {"plan": plan, "audit": ["request prepared"]}
```

审批前可以进行确定性计算、风险分析和参数规范化，但不应真正转账。这样即使后续暂停数小时，也不存在“先执行再审批”的逻辑漏洞。

### 5.2 interrupt 的载荷是给人的审阅材料

```python
decision = interrupt(
    {
        "question": "Approve this transfer?",
        "request": state["request"],
        "amount": state["amount"],
        "plan": state["plan"],
    }
)
```

第一次执行到这里时：

1. 当前任务与状态被 checkpoint。
2. 图暂停。
3. `invoke()` 返回特殊的 `__interrupt__` 数据。
4. `interrupt()` 暂时没有普通返回值，节点后续代码不执行。

载荷应足够让审批者做决定，但不要包含密码、令牌或无关个人数据。

### 5.3 读取中断和保存状态

```python
paused = graph.invoke(initial_state, config)
payload = paused["__interrupt__"][0].value
snapshot = graph.get_state(config)
```

本例自测断言：

```python
assert snapshot.next == ("approval",)
assert snapshot.values["audit"] == ["request prepared"]
```

`next` 表示待继续的节点。暂停时 `approval` 尚未产生正常状态更新，所以 audit 只有 prepare 的记录。

### 5.4 Command(resume) 把值送回 interrupt

```python
final = graph.invoke(
    Command(resume={"approved": True, "reviewer": "casey"}),
    config,
)
```

恢复时 `approval` 节点重新从函数开头运行。当代码再次到达同一个 `interrupt()` 调用时，它返回 resume 字典，随后节点严格验证字段再写入审批状态：

```python
if "approved" not in decision or type(decision["approved"]) is not bool:
    raise TypeError("approved must be exactly true or false")

reviewer = decision.get("reviewer")
if not isinstance(reviewer, str) or not reviewer.strip():
    raise TypeError("reviewer must be a non-empty string")
```

不能写成 `bool(decision.get("approved"))`：非空字符串 `"false"` 的 Python truthiness 是
`True`，会把拒绝错误地解释为批准；整数 `1` 也不是这个协议允许的布尔决定。

Resume value 是不可信输入。本例中的 `reviewer` 仍只是为了讲解协议而放在 payload 中的演示
标签，严格校验格式并不能证明身份。真实系统应从认证层提供的可信 context 获取审批人，并验证：

- reviewer 是否是已认证身份；
- 是否有该金额等级的审批权限；
- 任务是否已过期或已被别人处理；
- 决定是否带签名、审计 ID 和备注。

### 5.5 审批后再路由到副作用节点

```python
def route_decision(state):
    return "execute" if state["approved"] else "reject"
```

真正的执行位于单独的 `execute_transfer` 节点。这样恢复重跑 `approval` 时不会重复执行转账。

### 5.6 自测覆盖同意和拒绝两条线程

示例使用不同 `thread_id` 测试：

- 同意后进入 `execute`，audit 记录 prepare、approve、execute。
- 拒绝后进入 `reject`，结果明确为 cancelled。
- 字符串 `"false"`、整数 `1`、缺字段和空 reviewer 都被拒绝。
- 完成后 `get_state(config).next == ()`。

HITL 测试不能只验证“返回中断”。还要验证 resume 后路径、最终状态和线程隔离。

## 6. 示例二：静态断点与 update_state

文件：[examples/update_state_review.py](./examples/update_state_review.py)

运行：

```bash
conda run -n langgraph python langgraph/08-human-in-the-loop/examples/update_state_review.py
conda run -n langgraph python langgraph/08-human-in-the-loop/examples/update_state_review.py --self-test
```

图结构：

```text
START -> draft -> [interrupt_before publish] -> publish -> END
                         |
                         +-- human edits checkpointed draft
```

### 6.1 静态断点在编译时声明

```python
return builder.compile(
    checkpointer=InMemorySaver(),
    interrupt_before=["publish"],
)
```

这类断点不需要节点内部调用 `interrupt()`。图在执行 `publish` 前暂停，适合开发调试或固定的“发布前审阅”步骤。

动态中断适合由业务状态决定何时暂停；静态断点适合拓扑上固定的位置。生产业务通常更偏向动态 interrupt，因为中断载荷和恢复值更明确。

### 6.2 get_state 确认暂停位置

首次调用后：

```python
snapshot = graph.get_state(config)
assert snapshot.next == ("publish",)
```

此时 `draft` 已完成，`publish` 尚未运行。这正是人工编辑草稿的安全窗口。

### 6.3 update_state 修改 channel，而不是绕过状态系统

```python
graph.update_state(
    config,
    {"draft": edited_draft, "reviewed_by": reviewer},
)
```

`update_state` 会创建新的 checkpoint，并按对应 channel 的 reducer 语义合并更新。它不是直接修改某个 Python 字典引用。

对于带 reducer 的字段要尤其小心：传入的是“增量”还是“替换值”由 reducer 决定。消息字段通常要使用稳定 message ID 才能更新旧消息。

### 6.4 invoke(None, config) 从待执行节点继续

```python
final = graph.invoke(None, config)
```

`None` 表示没有提交新的入口输入。运行时读取 checkpoint 中的 pending task，执行 `publish`。自测确认发布内容使用人工编辑后的 draft，并且没有重新生成初稿。

## 7. 动态中断与静态断点怎么选

| 需求 | 推荐机制 |
|---|---|
| 节点需要向人展示问题并接收一个返回值 | `interrupt()` + `Command(resume=...)` |
| 是否暂停取决于金额、置信度或权限 | 节点中的条件 + `interrupt()` |
| 固定在某节点前调试或审阅状态 | `interrupt_before` |
| 人需要修改多个已有 state 字段 | 暂停后 `update_state()` |
| 工具内部需要批准后才能继续 | 动态 interrupt，或工具返回 Command 到审批流程 |

两种机制可以组合，但每增加一个暂停点，就要补充恢复测试和幂等设计。

## 8. 中断节点的重放与副作用

错误示例：

```python
def approval_node(state):
    send_email("approval requested")
    decision = interrupt(...)
    return {"approved": decision}
```

恢复时节点从开头重跑，邮件可能发送第二次。更可靠的方式包括：

1. 把通知放在中断前的独立节点。
2. 使用 `thread_id + action_id` 作为幂等键。
3. 让外部系统支持去重。
4. 把已完成副作用的凭证写入持久状态或事务日志。

同样，动态 interrupt 的调用顺序应保持稳定。不要在恢复时根据变化的列表顺序产生不同数量或顺序的 interrupt，否则 resume 值可能对应错误的暂停点。

## 9. 常见坑

### 9.1 没有 checkpointer

动态 interrupt 需要保存执行位置。没有 checkpointer，图无法可靠恢复。

### 9.2 恢复时更换 thread_id

新 ID 指向另一条线程，找不到原 checkpoint。业务系统应保存并传回稳定的 workflow/thread 标识。

### 9.3 把完整状态重新 invoke

这通常从 `START` 开始一次新运行，不是恢复。动态中断使用 `Command(resume=...)`；静态断点继续使用 `None`。

### 9.4 在 interrupt 前执行不可重复副作用

中断节点会重放。副作用应移到审批后的节点或使用幂等保护。

### 9.5 不验证 resume 值

人机界面、队列消费者或 API 都可能传错类型、过期决定或越权身份。恢复值必须按外部输入验证。

### 9.6 把 InMemorySaver 当生产持久化

它重启即丢失，且不解决多实例共享。生产系统要选择持久后端并执行初始化、迁移和连接管理。

### 9.7 只测试首次暂停

必须测试暂停、读取 payload、恢复、同意/拒绝分支、最终 next 为空，以及重复提交的业务策略。

## 10. 练习

1. 给审批载荷增加 `risk_level`，仅当金额超过阈值时调用 `interrupt()`。
2. 给 resume 字典增加必填 `comment`，拒绝缺少说明的拒绝操作。
3. 在示例二中更新一条带固定 ID 的 AI 草稿消息，而不是普通字符串字段。
4. 增加“退回修改”分支，让流程回到 draft，但设置最大修改次数。
5. 模拟服务重启：改用持久 checkpointer，在新 graph 实例中用同一 thread_id 恢复。
6. 给审批操作设计幂等 action ID，并解释重复 resume 应返回旧结果还是报错。

## 11. 自检

- [ ] 我能解释 checkpointer 与 thread_id 的不同职责。
- [ ] 我知道 `interrupt()` 的 payload 在哪里读取。
- [ ] 我能用 `Command(resume=value)` 让 interrupt 返回 value。
- [ ] 我知道动态中断节点恢复时会从函数开头重跑。
- [ ] 我能解释为什么副作用不能裸放在 interrupt 前。
- [ ] 我能用 `update_state()` 修改 checkpointed state。
- [ ] 我知道静态断点应以 `invoke(None, config)` 继续。
- [ ] 我不会把完整状态作为新输入冒充恢复。

## 12. 本章结论

HITL 的关键不是“程序停了一下”，而是执行位置、状态、待处理任务和人工决定都有明确协议。正确恢复依赖 checkpointer、稳定 thread ID、`Command(resume)` 或 `invoke(None)`，并要求所有副作用具备重放意识。

下一章将讨论运行中的另一个工程问题：如何流式观察进度、限定重试，并保证重试不会重复真实副作用。
