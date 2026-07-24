# Lab 01：可恢复客服工作流

这个 Lab 用一个确定性的客服系统串起整套课程。它没有真实 LLM，因此每条路径都可重复、可断言；把规则分类节点替换成模型后，图的状态、恢复、记忆和安全边界仍然成立。

## 1. 学习目标

你将看到这些机制如何协同，而不是单独调用：

| 能力 | 在项目中的落点 |
|---|---|
| State + reducer | 消息、trace 和并行调查证据分别定义合并语义 |
| Runtime Context | owner、tenant、当前 actor、角色、locale 和自动退款阈值 |
| 条件路由 | FAQ、账单、退款三条业务路径 |
| 原生子图 | FAQ 搜索与答复封装成独立子流程 |
| `Send` 动态并行 | 账单的 payment/invoice/duplicates 三项调查 |
| Checkpoint | `thread_id` 保存对话、调度状态和中断点 |
| Store | 用户 profile、跨线程历史、退款幂等记录 |
| Human-in-the-loop | 高于阈值的退款暂停，等待人工批准 |
| Streaming | 以 `updates` 观察节点增量和 interrupt 事件 |
| 可重放副作用 | ticket id 作为幂等键，规范化 payload 指纹检测冲突 |

## 2. 业务流程

```text
START
  |
  v
begin_ticket  <---------------- 绑定 owner/tenant，清空本轮临时状态
  |
  v
load_profile  <---------------- Runtime Context(user_id)
  |
  v
classify_intent
  |
  +-- faq ----> [FAQ subgraph] -----------------------+
  |             search_faq -> draft_faq_answer        |
  |                                                   |
  +-- billing -> plan_billing_checks                  |
  |                    |                              |
  |           Send x 3 in parallel                    |
  |          /         |          \                   |
  |     payment     invoice    duplicates             |
  |          \         |          /                   |
  |             synthesize_billing -------------------+
  |                                                   |
  +-- refund -> assess_refund                         |
                    |                                 |
                    +-- invalid -> clarify ----------+
                    |                                 |
                    +-- low --> execute_refund -------+
                    |                                 |
                    +-- high -> interrupt(approval)   |
                                   |                  |
                           Command(resume=decision)   |
                                   |                  |
                          execute / decline ----------+
                                                       |
                                                       v
                                              compose_response
                                                       |
                                                       v
                                              persist_case_memory
                                                       |
                                                      END
```

## 3. 文件结构

```text
01-support-agent/
├── README.md
├── TODO.md
├── run_demo.py
└── support_agent/
    ├── __init__.py
    ├── schemas.py   # state、context、reducer、输入构造
    ├── nodes.py     # 纯业务节点、interrupt、Store 副作用
    └── graph.py     # 授权网关、子图和主图拓扑
```

顶层课程目录不建 `__init__.py`，但 Lab 内部使用不同名字的 `support_agent` 包，不会遮蔽官方 `langgraph`。

## 4. 运行三个场景

从 `python-utils/` 根目录执行。

FAQ 子图：

```bash
conda run -n langgraph python langgraph/labs/01-support-agent/run_demo.py \
  --query "How do I reset my password?" \
  --ticket-id faq-001
```

账单动态并行：

```bash
conda run -n langgraph python langgraph/labs/01-support-agent/run_demo.py \
  --query "I see a duplicate invoice charge" \
  --ticket-id bill-001 \
  --stream
```

高风险退款，人工拒绝：

```bash
conda run -n langgraph python langgraph/labs/01-support-agent/run_demo.py \
  --query "Please refund 1200" \
  --ticket-id refund-001 \
  --stream \
  --no-approve
```

离线自检：

```bash
conda run -n langgraph python langgraph/labs/01-support-agent/run_demo.py --self-test
```

## 5. State 为什么这样设计

`SupportState` 没有把所有列表都简单地标成 `operator.add`：

| 字段 | 合并方式 | 原因 |
|---|---|---|
| `messages` | `add_messages` | 按 message id 追加或替换 |
| `trace` | 列表相加 | 保留跨节点、跨轮执行轨迹 |
| `evidence` | `merge_evidence` | 并行合并，并按 ticket/check 替换重复调查 |
| `profile` | 覆盖 | 每次从 Store 读取当前 profile |
| `intent/risk/resolution` | 覆盖 | 代表当前工单的最新决策 |
| `checks` | 覆盖 | planner 一次产生完整调查计划 |

`merge_evidence` 用 `(ticket_id, check)` 作为逻辑主键。这样图因恢复或重试再次执行同一调查时，不会无限追加重复证据；不同 ticket 的证据仍能在线程历史中共存。

每次 `new_ticket()` 都生成一个 `turn_id`。Human/AI message id 使用 turn id，而不是 ticket id：重新提交同一个 input 对象仍是幂等替换，针对同一 ticket 的新一轮对话则会产生新 turn 并追加两条消息。

`begin_ticket` 在每次新输入后显式清空 `intent`、FAQ 临时值、checks、amount、risk、approval 和 resolution。消息、trace 与按 ticket 标识的历史 evidence 继续累积。不能只期待覆盖型 reducer 自动“忘掉”旧字段：新输入没有携带的 state key 默认仍会从 checkpoint 继承。

## 6. Context、Checkpoint、Store 的分工

一次调用同时带 owner、actor 和线程标识。普通客户调用时 actor 默认就是 owner：

```python
context = SupportContext(user_id="user-1", tenant_id="tenant-a")
config = {"configurable": {"thread_id": "conversation-9"}}
graph.invoke(input_state, config, context=context)
```

它们不应互换：

```text
tenant_id=tenant-a, user_id=user-1
  -> 案件 owner，也决定 Store namespace

actor_id=None, role=customer
  -> 当前认证操作人默认是 user-1

thread_id=conversation-9
  -> Checkpointer namespace，决定恢复哪条执行线程
```

`begin_ticket` 第一次执行时把 `(tenant_id, user_id)` 写入 checkpoint，以后同 thread 的每次新输入都必须匹配。`SupportInput` 不包含 owner 字段，普通图输入无法改写它。返回的 `SupportApp` 还会在把新输入交给 checkpointer **之前**读取 snapshot 并校验，因此失败的越权输入不会先污染 checkpoint。

`get_state()` 和 `get_state_history()` 同样要求显式传入 Context 并校验 owner/tenant。网关不代理 `update_state()`、`bulk_update_state()` 等底层写接口，防止调用者绕过输入 schema 和 `begin_ticket`。`get_graph()` 只返回静态拓扑，不读取任何 thread 数据。

这个离线 Lab 的网关只公开已经在当前 `InMemorySaver` 基线上验证过的同步 `invoke/stream`。不要把同步 checkpointer 包装成看似可用的异步接口；需要异步服务时，应选择并验证对应的 async checkpointer/store，再为 `ainvoke/astream` 单独实现同一套授权测试。

Context 必须由认证后的宿主应用构造，不能直接反序列化用户提交的 JSON。这个 Lab 演示的是认证完成后的授权边界，不包含登录、JWT 验签或组织成员目录。

生产服务还应由服务端分配 thread id，或在图外维护原子的 `thread_id -> tenant/owner` ACL。checkpoint 中的 owner 绑定是纵深防御，不替代认证系统、首次建线程授权或跨进程并发控制。

Store 使用三个 namespace：

```text
("tenants", tenant_id, "users", user_id, "profile")
("tenants", tenant_id, "users", user_id, "support_history")
("tenants", tenant_id, "operations", user_id, "refunds")
```

一个用户可以开多个对话线程，这些线程共享同一 tenant 下的用户 profile；相同 user id 在不同 tenant 也彼此隔离。

## 7. FAQ 子图与 reducer 边界

FAQ 使用原生 compiled subgraph 作为主图节点。一个容易忽略的坑是：如果父图和子图共享同一个累积型 `trace` 字段，子图会接收父 trace，完成后返回完整 trace，父 reducer 再次相加，从而重复前缀。

本项目用独立 `FaqState.faq_trace` 隔离子图内部轨迹，并在新 ticket 开始时清空它：

```text
parent.trace -> 不进入 faq_trace reducer
subgraph.faq_trace -> 子图内部累积
begin_ticket -> faq_trace=[]
subgraph.faq_trace -> 只包含当前轮子图轨迹
record_faq_trace -> 把当前轮子图轨迹合并回 parent.trace
```

这不是为了多写一个字段，而是在明确子图的输入/输出契约。生产中还可以用适配节点显式转换两个完全不同的 schema。

## 8. 账单分支为什么是真并行

`dispatch_billing_checks` 根据运行时列表返回三个 `Send`：

```python
return [
    Send("run_billing_check", {"current_check": check, ...})
    for check in state["checks"]
]
```

每个 worker 读取自己的局部输入，并返回一个 `evidence` 更新。它们位于同一个执行阶段，最后由 reducer 合并，然后 `synthesize_billing` 才运行。

不能依赖 worker 的完成顺序表达业务顺序。合成节点按 `check` 排序，测试断言集合和数量，不断言并行日志的偶然顺序。

## 9. 高风险退款如何恢复

首次执行使用客户身份：

```python
customer = SupportContext(user_id="customer-1", role="customer")
paused = graph.invoke(initial, config, context=customer)
```

超过 `auto_refund_limit` 时，`request_refund_approval` 调用 `interrupt(payload)`。运行时保存 checkpoint，结果包含 `__interrupt__`。

恢复时，案件 owner 不变，但当前 actor 必须是可信的 reviewer/admin：

```python
reviewer = SupportContext(
    user_id="customer-1",       # 案件 owner
    actor_id="reviewer-alice",  # 当前认证员工
    role="reviewer",
)
finished = graph.invoke(
    Command(resume={"approved": True}),
    config,
    context=reviewer,
)
```

resume payload 只携带决策，并且 `approved` 必须是真正的 `bool`；字符串 `"false"` 不会被 truthiness 当成批准。reviewer 姓名和角色只取自可信 Context，payload 中伪造的 `reviewer` 字段不会成为审计身份。customer 角色不能批准，即使它提交 `approved=True`。

这里没有把 `paused` 当作新输入。相同 `thread_id` + `Command(resume=...)` 才会从中断任务恢复。Context 不持久化，因此恢复调用仍显式传入。`SupportApp` 必须在调用底层 graph 之前完成 resume 授权，因为 LangGraph 会先把 resume value 写入任务 scratchpad，再从中断节点开头重放；只在 `interrupt()` 返回后才检查权限已经太迟。

`interrupt()` 之前没有不可重复副作用。恢复时中断节点会从头进入，取得 resume value 后才返回更新。

## 10. 金额验证与幂等性

退款金额不使用二进制 `float` 做业务判断。解析器先把唯一、正数、有限且最多两位小数的金额规范化为十进制字符串，再用 `Decimal` 与阈值比较。

这些输入都不会自动退款：

- 没有金额；
- 同一句里有多个数值，无法确定哪一个是金额；
- `NaN`、`Infinity`、零、负数或超过两位小数；
- 带逗号的数值，例如 `1,200`，因为不同 locale 对逗号含义不同。

它们进入 clarification 分支，只生成“未执行”答复。`return/refund policy` 咨询进入 FAQ，不进入退款分支。自动退款阈值为负数、NaN 或无穷时也不会 fail-open，而是强制人工审核。

即使使用 checkpoint，仍可能发生：

```text
退款服务成功 -> 进程在保存下一 checkpoint 前崩溃 -> 节点被重放
```

`execute_refund` 先用 `ticket_id` 查询 Store 中的 operation，同时为 owner、tenant、ticket、币种和规范化金额计算 SHA-256 请求指纹：

- 已存在且指纹相同：使用 Store 中的 operation id、金额和状态生成答复，记录 `refund:deduplicated`；
- 已存在但指纹不同：不执行、不覆盖旧记录，返回 `refund:idempotency-conflict`；
- 不存在：执行确定性 mock 操作并写入幂等记录。

`InMemoryStore` 的 `get -> put` 不是 compare-and-set。两个并发 worker 首次处理相同 key 时仍可能竞态，跨进程更没有锁。生产中必须让退款服务或数据库用唯一约束/原子幂等接口做最终裁决，并配合 outbox/saga；本例只展示 payload-bound idempotency 的必要形状，不宣称实现了分布式事务。

## 11. 流式事件应该怎么看

`--stream` 使用 `stream_mode="updates"`，输出每个节点的增量：

```text
{'classify_intent': {'intent': 'refund', ...}}
{'assess_refund': {'amount': '1200', 'risk_level': 'high', ...}}
{'__interrupt__': (...)}
```

这比只打印最终 state 更适合回答：哪个节点改变了 risk、为什么暂停、恢复后执行了哪条分支。生产观测还应关联 request id、thread id、node、latency、retry count 和外部 operation id，同时对消息与审批 payload 脱敏。

## 12. 测试什么

运行：

```bash
conda run -n langgraph python -m unittest discover -s langgraph/tests -v
```

测试不比较整段自然语言，而是断言：

- 三种 intent 都走到正确分支；
- FAQ profile 来自对应用户；
- 三个 `Send` 结果完整合并；
- 高额退款先暂停，resume 后不重跑 START；
- owner/tenant 不匹配时，新输入与 resume 都在写 checkpoint 前拒绝；
- customer 不能批准，审批值必须为 bool，reviewer 来自 Context；
- 缺失、歧义、非法金额不会触发退款，错误阈值强制人工审核；
- 每个新 ticket 清空临时状态，同 ticket 的不同 turn 仍累积消息；
- 同 user 跨 thread 共享 Store，不同 user 隔离；
- 相同 payload 复用幂等 operation，不同 payload 报冲突。

## 13. 这是 toy，不是生产客服平台

为了让编排机制可观察，项目有意简化：

- intent 和 FAQ 使用关键词规则，不是真实模型/RAG；
- 金额解析有意拒绝 locale 相关格式，没有币种转换；
- `InMemorySaver` 与 `InMemoryStore` 重启即丢失；
- 没有身份认证提供方，安全性依赖宿主正确构造可信 Context；
- 内存幂等记录没有跨线程、跨进程原子 compare-and-set；
- 没有 PII 脱敏、内容安全、速率限制或审计后端；
- 退款只是 Store 写入，不访问真实支付系统；
- 没有分布式 worker、并发压测和 schema migration。

生产升级顺序建议：先替换持久化后端并保留全部行为测试，再接真实检索/模型，最后接高风险工具；不要反过来。

## 14. 验收问题

1. 为什么 `user_id` 不放在 checkpoint config 里当作唯一身份来源？
2. 为什么 `evidence` 不能使用普通覆盖语义？
3. 为什么并行 worker 的 trace 顺序不能当业务事实？
4. 为什么 resume 授权必须发生在底层 graph 消费 `Command` 之前？
5. 为什么 reviewer 不能从 resume payload 获取？
6. 如果 refund API 成功但 Store 写入失败，当前幂等方案还缺什么？
7. 子图共享累积字段为什么会重复，当前 schema 如何阻止它？
8. 把 classify 换成 LLM 后，哪些测试仍应保持不变？

能运行、解释并完成 [TODO.md](./TODO.md) 中至少两个改造后，才算真正完成这个 Lab。
