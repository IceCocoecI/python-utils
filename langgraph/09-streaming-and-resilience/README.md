# 09 · 流式与可靠性：看见执行、控制重试、保护副作用

> 目标环境：Python 3.11、LangGraph 1.0.6。  
> 本章完全离线，覆盖 Graph API 与 Functional API。

## 学完本章能做什么

你将能够：

1. 区分最终状态、`updates` 流和 `custom` 流。
2. 使用 `get_stream_writer()` 从节点发出业务进度事件。
3. 同时订阅多个 stream mode，并读懂 `(mode, chunk)` 事件形状。
4. 为节点配置有边界的 `RetryPolicy`。
5. 解释为什么 checkpoint 和 retry 都不能自动撤销外部副作用。
6. 使用幂等键避免“远端成功、响应丢失”导致重复操作。
7. 使用 `@task` 与 `@entrypoint` 编写一个 Functional API 工作流。
8. 根据流程形状在 Graph API、Functional API 和普通 Python 之间取舍。

## 1. 可靠工作流不只需要最终答案

一个长任务只在最后返回 `done`，调用者无法回答：

- 现在运行到哪个节点？
- 是模型慢、工具慢，还是进入了重试？
- 用户界面的进度条应该显示什么？
- 哪个节点第一次产生了错误数据？
- 失败前是否已经执行了扣款、发信或写库？

因此需要把三个问题分开：

```text
观测：运行时发生了什么？              -> stream
恢复：暂时性失败是否再试？              -> RetryPolicy
副作用：再试是否会重复真实业务操作？     -> idempotency
```

流式事件不会让失败自动恢复；重试不会让副作用自动回滚；checkpoint 也不会替你撤销外部系统已经完成的操作。

## 2. 流式执行的心智模型

普通 `invoke()`：

```text
input -> [整张图运行] -> final state
```

`stream()`：

```text
input -> node A -> event -> node B -> event -> node C -> event
```

事件不是“print 的另一种写法”，而是调用者与工作流之间的数据协议。不同模式回答不同问题：

| Stream mode | 主要内容 | 适合场景 |
|---|---|---|
| `updates` | 每个节点返回的状态增量 | 调试路径、增量 UI、测试节点输出 |
| `values` | 每一步合并后的完整状态 | 小状态教学、检查全局演化 |
| `custom` | 节点主动发出的任意业务事件 | 进度、阶段提示、外部任务状态 |
| `messages` | 模型消息/token 相关事件 | LLM token 流式输出 |
| `debug` | 更详细的运行时调试信息 | 深度诊断，不宜直接暴露给终端用户 |

本章聚焦不依赖模型的 `updates` 和 `custom`。

## 3. 示例一：混合 updates 与 custom 事件

文件：[examples/streaming_modes.py](./examples/streaming_modes.py)

运行：

```bash
conda run -n langgraph python langgraph/09-streaming-and-resilience/examples/streaming_modes.py
conda run -n langgraph python langgraph/09-streaming-and-resilience/examples/streaming_modes.py --self-test
```

图结构：

```text
START -> normalize -> tokenize -> summarize -> END
             |            |             |
             +-- custom progress -------+
             +-- normal state updates --+
```

### 3.1 updates 来自节点返回值

```python
def tokenize(state):
    return {"words": state["normalized"].split()}
```

使用 `stream_mode="updates"` 时，该节点产生：

```python
{"tokenize": {"words": ["state", "streams", "state"]}}
```

外层是节点名，内层是该节点返回的增量。它不是完整 state，所以不会重复携带原始文本和其他未变化字段。

### 3.2 custom 来自运行时 writer

```python
def tokenize(state):
    writer = get_stream_writer()
    writer({"event": "progress", "step": "tokenize", "percent": 60})
    return {"words": state["normalized"].split()}
```

writer 发出的对象不会自动写入 state。它是旁路事件：

```text
State channel: words = [...]
Custom stream: {event: progress, percent: 60}
```

这条边界很重要。瞬时 UI 进度通常不必污染 checkpoint state；真正影响恢复和业务结果的数据仍应返回到 state。

### 3.3 同时订阅多个模式

```python
for mode, chunk in graph.stream(
    initial_state,
    stream_mode=["updates", "custom"],
):
    ...
```

指定一个 mode 时通常直接得到 chunk；指定 mode 列表时，每个事件带 mode 标签：

```text
custom  -> progress 25
updates -> normalize update
custom  -> progress 60
updates -> tokenize update
custom  -> progress 100
updates -> summarize update
```

调用者应按 `mode` 分派，而不是通过猜测字典键判断事件类型。

### 3.4 事件 schema 也要设计

本例统一使用：

```python
{"event": "progress", "step": "normalize", "percent": 25}
```

生产系统还可以增加：

- `version`：事件 schema 版本。
- `workflow_id`：关联运行。
- `sequence`：业务序号；不要依赖网络到达顺序。
- `message`：面向用户的可本地化信息。
- `details`：仅允许白名单字段，避免泄露 state。

### 3.5 自测为什么断言事件交替顺序

示例断言 mode 序列、progress 百分比和每个节点的 update。这样可以发现：

- 忘记订阅 custom；
- writer 放错节点；
- 节点返回字段形状变化；
- 拓扑顺序意外改变。

对于并行图，不应断言不同并行节点的固定完成顺序；应按节点名或业务 ID 汇总。顺序断言只适合本例的线性图。

## 4. RetryPolicy 的心智模型

节点级 retry 可以近似理解为：

```text
call node
  |
  +-- success ----------------------> merge update
  |
  +-- matching exception -> wait -> call node again
  |
  +-- non-matching / attempts used -> raise failure
```

`RetryPolicy` 的常用字段：

| 字段 | 含义 |
|---|---|
| `max_attempts` | 包含首次调用在内的最大尝试次数 |
| `initial_interval` | 首次重试前等待时间 |
| `backoff_factor` | 后续间隔倍增因子 |
| `max_interval` | 单次等待上限 |
| `jitter` | 是否加入随机抖动，避免大量任务同时重试 |
| `retry_on` | 哪些异常类型或谓词结果允许重试 |

不要对所有 `Exception` 无差别重试。参数错误、权限拒绝、数据校验失败通常不会因为等待一次而消失。

## 5. 示例二：重试与幂等副作用

文件：[examples/retry_idempotency.py](./examples/retry_idempotency.py)

运行：

```bash
conda run -n langgraph python langgraph/09-streaming-and-resilience/examples/retry_idempotency.py
conda run -n langgraph python langgraph/09-streaming-and-resilience/examples/retry_idempotency.py --self-test
```

示例模拟最棘手的一类失败：

```text
第一次请求到远端
    |
    +-- 远端已经成功记录扣款
    |
    +-- 响应在网络中丢失，客户端抛异常
    |
LangGraph 重试同一节点
```

如果第二次请求创建新扣款，就会重复收费。

### 5.1 幂等键标识一次业务意图

本例使用 `operation_id="order-42"`。账本先检查：

```python
if operation_id in self.records:
    record = self.records[operation_id]
    if record["amount"] != amount:
        raise IdempotencyConflict(...)
    return record["receipt"], True
```

同一个 operation ID 和同一金额再次提交时，返回第一次的 receipt，不创建第二条记录。相同 ID 如果携带不同金额，则是业务冲突，不能伪装成一次成功的重复请求。

幂等键必须代表“同一次业务操作”，不能每次 retry 都重新生成随机 UUID。常见组成是租户 ID、业务订单 ID 和操作类型。

### 5.2 故障发生在外部提交之后

```python
self.records[operation_id] = {
    "amount": amount,
    "receipt": receipt,
}
raise TransientAfterCommit("response lost after remote commit")
```

节点没有正常 return，因此 LangGraph 没有合并该节点的 state update；但外部账本已经改变。这说明：

> 图状态的原子性不等于外部系统事务的原子性。

Checkpoint 只能记录 LangGraph 运行状态，无法穿越网络自动回滚支付系统、邮件服务或数据库中的独立事务。

### 5.3 只重试明确的瞬时异常

```python
retry = RetryPolicy(
    max_attempts=3,
    retry_on=TransientAfterCommit,
    ...,
)

builder.add_node("charge", charge_once, retry_policy=retry)
```

本例把等待时间压到 0.01 秒，适合离线测试。生产值要根据服务 SLA、超时和总任务预算设置，并加入 jitter。

### 5.4 自测同时断言尝试数和提交数

```python
assert ledger.attempts == 2
assert len(ledger.records) == 1
assert result["deduplicated"] is True
```

这三个断言分别证明：

1. retry 确实发生。
2. 外部副作用只提交一次。
3. 第二次调用复用了既有结果。

只断言最终 receipt 存在，无法发现重复扣款。

自测还会用同一个 `operation_id` 提交不同金额，并断言：

```python
try:
    ledger.charge("order-42", 1_200)
except IdempotencyConflict:
    pass
else:
    raise AssertionError("conflicting request must fail")
```

`RetryPolicy` 只重试 `TransientAfterCommit`，不会重试这种确定性的请求冲突。

## 6. 幂等不是简单“去重字符串”

本例选择“同一幂等键、不同参数时报冲突”。真实系统还要定义：

- 幂等记录保留多久？过期后重放如何处理？
- 并发收到两个相同 key 时，谁获得锁或唯一约束？
- 结果生成中、成功、失败分别如何记录？
- 调用方超时后，如何查询已有操作状态？

推荐把幂等性落在最接近副作用的数据边界，例如支付 API 的 idempotency key 或数据库唯一约束，而不只存在当前 Python 进程的字典中。

本例的 `IdempotentLedger` 只用于展示语义，不是生产存储。

## 7. Functional API 的心智模型

Graph API 强调显式状态和拓扑：

```text
StateGraph + nodes + edges + reducers
```

Functional API 保留普通 Python 控制流：

```text
@entrypoint workflow
    |
    +-- @task A -> future
    +-- @task B -> future
    +-- ordinary if / for / variables
```

两者共享 LangGraph 的运行时能力，但表达方式不同。

| 需求 | Graph API 更自然 | Functional API 更自然 |
|---|---|---|
| 审查复杂分支、循环和角色拓扑 | 是 | 否 |
| 共享 state/reducer 很重要 | 是 | 需要手工组织返回值 |
| 已有清晰 Python 函数流程 | 可能过重 | 是 |
| 任务 fan-out/future | 可以用 Send | 可以直接创建多个 task future |
| 需要可视化业务图 | 是 | 较弱 |

## 8. 示例三：entrypoint 与 task

文件：[examples/functional_workflow.py](./examples/functional_workflow.py)

运行：

```bash
conda run -n langgraph python langgraph/09-streaming-and-resilience/examples/functional_workflow.py
conda run -n langgraph python langgraph/09-streaming-and-resilience/examples/functional_workflow.py --self-test
```

### 8.1 task 是可观察、可重试的工作单元

```python
@task
def normalize(text: str) -> str:
    return " ".join(text.casefold().split())
```

Task 只能在 entrypoint 或 LangGraph 运行时内部调用。调用 task 得到 future，而不是立即得到普通返回值：

```python
normalized = normalize(text).result()
```

### 8.2 entrypoint 是工作流入口

```python
@entrypoint()
def analyze_text(inputs: WorkflowInput) -> WorkflowOutput:
    ...
```

装饰后的对象支持 `invoke()`、`stream()` 等运行方法。函数只接收一个业务输入；多参数可以放进 TypedDict。

### 8.3 先创建 futures，再取结果

```python
count_future = count_words(normalized)
long_words_future = select_long_words((normalized, minimum_length))

return {
    "word_count": count_future.result(),
    "long_words": long_words_future.result(),
}
```

两个 task 在依赖满足后都被调度。不要写成“创建一个 future 后立刻 `.result()`，再创建下一个”，否则会无意中把独立任务串行化。

### 8.4 Functional API 仍能流式观察

示例对 entrypoint 使用 `stream_mode="updates"` 并断言存在最终 `analyze_text` update。Task 也可以调用 `get_stream_writer()` 发 custom 事件，API 与 Graph 节点一致。

## 9. 常见坑

### 9.1 把 custom progress 写进持久 state

每 1% 更新一次 state 会制造大量 checkpoint 和数据传输。只用于 UI 的瞬时进度优先使用 custom stream。

### 9.2 把完整 state 当 custom 事件发送

这会泄露敏感字段并放大带宽。事件应有独立、最小、可版本化 schema。

### 9.3 对所有异常无限重试

重试必须有异常白名单、次数、间隔和总预算。权限错误和坏参数应快速失败。

### 9.4 认为 retry 会回滚外部副作用

节点抛异常前已经完成的外部操作仍然存在。必须使用幂等键、事务/outbox 或可查询操作状态。

### 9.5 每次重试生成新幂等键

这会让服务把每次尝试当成新操作。key 应来自稳定业务意图。

### 9.6 在 Functional API 中立即等待每个 task

独立任务要先创建 futures，再收集结果，否则失去并行调度机会。

### 9.7 为简单函数强行建图

如果流程只是两个无状态、毫秒级函数且不需要持久化、流式或重试，普通 Python 更简单。

## 10. 练习

1. 给 streaming 示例增加 `warning` custom 事件：当唯一词比例低于 0.5 时发出。
2. 同时订阅 `values`，比较它与 `updates` 的体积和字段形状。
3. 修改 retry 示例，让前两次都在提交前失败，第三次成功，并断言提交数仍为 1。
4. 增加同 key 不同 amount 的冲突检查，禁止静默返回旧 receipt。
5. 给 RetryPolicy 增加第二种不可重试异常，证明它只尝试一次。
6. 给 Functional API 示例的 task 发 custom 进度事件。
7. 把 Functional API 示例改成 Graph API，并比较测试可读性和拓扑可见性。

## 11. 自检

- [ ] 我能解释 `invoke()` 与 `stream()` 的用途差异。
- [ ] 我能区分 `updates`、`values` 和 `custom`。
- [ ] 我知道多 stream mode 的事件为什么带 mode 标签。
- [ ] 我能为 custom 事件设计独立 schema。
- [ ] 我知道 RetryPolicy 只应覆盖瞬时异常。
- [ ] 我能解释“远端已成功、响应丢失”的重复副作用风险。
- [ ] 我知道 checkpoint 不会自动回滚外部系统。
- [ ] 我能为一次业务意图设计稳定幂等键。
- [ ] 我能区分 `@task` 与 `@entrypoint`。
- [ ] 我能说明什么时候 Functional API 比 StateGraph 更合适。

## 12. 本章结论

可靠性来自边界清晰：stream 负责观测，RetryPolicy 负责有界重试，外部系统的幂等协议负责副作用安全。Functional API 提供了另一种编排表达，但不会取消这些工程约束。

下一章把规模继续扩大：如何用原生子图划分模块，用 `Send` 动态 fan-out，并让 supervisor 协调多个独立 worker。
