# 04 · Runtime Context：把运行依赖与业务状态分开

> 目标环境：Python 3.11、LangGraph 1.0.6。  
> 本章完全离线，不需要模型、API Key 或网络。

## 学完本章能做什么

你将能够：

1. 解释 State 与 Runtime Context 分别解决什么问题。
2. 使用 `context_schema` 声明运行上下文类型。
3. 在节点中通过 `Runtime[Context]` 读取用户身份、租户、区域和依赖。
4. 在 `invoke(..., context=...)` 中为每次调用注入不同上下文。
5. 区分 `context` 与 `config`、checkpoint、Store、模型上下文窗口。
6. 避免把数据库客户端、权限和 API 配置复制进 State。

## 1. 问题：不是所有数据都应该在 State 中流动

前几章只使用 State，足以表达节点之间不断变化的业务数据。但真实应用还会依赖另一类信息：

- 当前用户是谁、属于哪个租户。
- 用户使用什么语言和区域设置。
- 本次请求允许执行哪些操作。
- 节点应该使用哪个数据库仓库、价格查询器或测试替身。
- 当前部署环境的稳定配置是什么。

这些信息通常满足两个特征：

1. 在一次图调用期间保持稳定。
2. 节点需要读取它们，但不应该把它们当作业务进度反复更新。

如果把所有内容都塞进 State，会出现几个问题：

- 每个节点都能意外修改用户身份或依赖对象。
- checkpointer 可能尝试保存数据库客户端、函数或连接池。
- 对外输出、调试状态和 checkpoint 被无关配置污染。
- 测试必须构造一份庞大的“万能状态”。
- 权限来源与模型生成的普通业务文本混在一起，边界不清晰。

LangGraph 1.x 使用 **Runtime Context** 表达这种“随调用注入、在节点中只读使用”的数据。

## 2. 边界：五种容易混淆的“上下文”

在继续写代码前，先把五个概念分开：

| 概念 | 典型内容 | 生命周期 | 谁负责更新 | 是否自动进入 checkpoint |
|---|---|---|---|---|
| State | 当前输入、中间结果、步骤、消息 | 一次执行；有 checkpointer 时可跨调用 | 节点通过 update | 是 |
| Runtime Context | 用户、租户、语言、依赖、运行设置 | 单次 `invoke/stream` | 调用方提供，节点只读 | 否 |
| Checkpoint | 某个 thread 的 State 快照和执行位置 | 多次调用、多个 superstep | checkpointer 自动管理 | 它本身就是持久化记录 |
| Store | 用户偏好、长期事实、跨 thread 资料 | 应用级 | 业务代码显式 `put/get/search` | 不属于 checkpoint |
| 模型上下文窗口 | 实际发给模型的消息、检索片段、系统提示 | 一次模型调用 | 应用组装 | 取决于你是否另行放进 State |

一句话记忆：

```text
State    = 这张图现在处理到哪里、产生了什么
Context  = 这次运行是在谁的环境中、可以使用什么依赖
Checkpoint = 某个 thread 的 State 历史快照
Store    = 应用显式保存的跨 thread 长期资料
模型上下文 = 此刻真正送入模型的有限信息
```

本章只处理第二项。第 05 章加入 checkpoint，第 06 章再加入 Store。

## 3. 心智模型：两条输入通道

一次调用可以看作有两条不同的数据通道：

```text
graph.invoke(
    input={...},              ─────► State channel ─► 节点更新 ─► 输出
    context=RequestContext(...) ───► Runtime context ────────► 只读依赖
)
```

图中的节点可以同时接收二者：

```python
def node(state: MyState, runtime: Runtime[MyContext]) -> dict:
    business_input = state["query"]
    user_id = runtime.context.user_id
    return {"answer": f"{user_id}: {business_input}"}
```

State 会随着节点 update 改变；同一次调用中的 context 保持不变。只有节点显式把 context 中的值复制到返回字典时，它才会进入 State。

## 4. 核心 API

### 4.1 用普通类型定义 Context

推荐先用不可变 dataclass：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class RequestContext:
    user_id: str
    locale: str
    permissions: frozenset[str]
```

`frozen=True` 不是 LangGraph 的硬性要求，但它能把“本次运行只读”变成 Python 层面的约束。

Context 类型是接口说明，不是自动转换器。若节点期待 `RequestContext`，调用方就应传 `RequestContext(...)`，不要假设 LangGraph 会把任意字典自动转换成 dataclass。

### 4.2 使用 context_schema 构建图

```python
builder = StateGraph(
    AccessState,
    context_schema=RequestContext,
)
```

在 LangGraph 1.0 中应使用 `context_schema`。旧教程中的 `config_schema` 已弃用，不要在新代码中继续传播。

### 4.3 在节点签名中声明 Runtime

```python
from langgraph.runtime import Runtime

def authorize(
    state: AccessState,
    runtime: Runtime[RequestContext],
) -> dict[str, bool]:
    return {
        "allowed": state["action"] in runtime.context.permissions,
    }
```

`Runtime[RequestContext]` 让 IDE 和类型检查器知道 `runtime.context` 的结构。节点如果不需要 context，仍然可以只接收 `state`。

`Runtime` 还暴露 Store 和自定义流 writer 等运行能力；本章只关注 `runtime.context`，避免一次引入过多概念。

### 4.4 调用时显式传 context

```python
result = graph.invoke(
    {"action": "delete", "resource": "report-42"},
    context=RequestContext(
        user_id="alice",
        locale="zh-CN",
        permissions=frozenset({"read"}),
    ),
)
```

每次调用都可以传不同 context。同一张编译图不必为每个用户重建。

### 4.5 context 不是 invoke 的 config

下面两个参数职责不同：

```python
graph.invoke(
    input_state,
    config={"configurable": {"thread_id": "thread-1"}},
    context=RequestContext(user_id="alice", ...),
)
```

| 参数 | 用途 |
|---|---|
| `config` | 执行配置，例如 `thread_id`、recursion limit、tags |
| `context` | 节点读取的强类型业务运行上下文 |

不要因为旧 API 曾叫 `config_schema`，就把二者继续混为一谈。

## 5. 示例一：身份、权限与语言

文件：[examples/request_context.py](./examples/request_context.py)

运行：

```bash
conda run -n langgraph python langgraph/04-runtime-context/examples/request_context.py
conda run -n langgraph python langgraph/04-runtime-context/examples/request_context.py --admin --locale en-US
conda run -n langgraph python langgraph/04-runtime-context/examples/request_context.py --self-test
```

图结构：

```text
START -> authorize -> render -> END
```

### 5.1 State 只保留业务流程数据

```python
class AccessState(TypedDict):
    action: Action
    resource: str
    allowed: bool
    response: str
```

State 回答的是：请求执行什么动作、目标资源是什么、决策和响应是什么。

它不包含 `permissions`。权限不是节点计算过程中的可变草稿，而是应用认证和授权层提供的可信运行事实。

### 5.2 多个节点读取同一个 context

`authorize` 读取权限：

```python
allowed = state["action"] in runtime.context.permissions
```

`render` 读取用户和语言：

```python
if runtime.context.locale == "zh-CN":
    ...
```

两者得到同一个调用上下文，但都没有修改它。

### 5.3 相同输入，不同运行环境

相同输入：

```python
{"action": "delete", "resource": "report-42"}
```

viewer context 得到拒绝，admin context 得到允许。这不是隐式全局状态，而是调用者显式传入的差异，因此容易测试和追踪。

### 5.4 身份必须来自可信边界

示例从命令行构造 context，只是为了离线演示。真实服务中，`user_id`、`tenant_id` 和权限应该来自已经验证的会话、token 或网关，而不是直接信任用户 prompt：

```text
HTTP authentication -> trusted RequestContext -> graph.invoke(..., context=...)
user prompt -------------------------------------> State
```

模型说“我是管理员”不等于它真的获得管理员 context。

## 6. 示例二：依赖注入与可测试性

文件：[examples/dependency_injection.py](./examples/dependency_injection.py)

运行：

```bash
conda run -n langgraph python langgraph/04-runtime-context/examples/dependency_injection.py
conda run -n langgraph python langgraph/04-runtime-context/examples/dependency_injection.py --sku pen --quantity 3
conda run -n langgraph python langgraph/04-runtime-context/examples/dependency_injection.py --self-test
```

示例把价格查询函数和租户税率放进 `CheckoutContext`：

```python
@dataclass(frozen=True)
class CheckoutContext:
    tenant_id: str
    currency: str
    tax_basis_points: int
    lookup_price_cents: Callable[[str], int]
```

### 6.1 为什么查询器不应进入 State

价格查询器可能是真实数据库仓库、HTTP 客户端，也可能是测试替身。它不是业务结果，通常也不可安全序列化。

如果把它放入 State：

- 输出可能携带内部对象。
- 启用 checkpoint 后可能序列化失败。
- reducer 和状态 schema 被运行基础设施污染。
- 测试需要在每次 state update 中小心保留它。

放入 Runtime Context 后，节点只需声明依赖：

```python
unit_price = runtime.context.lookup_price_cents(state["sku"])
```

### 6.2 测试时替换依赖

自检为零售和批发租户注入不同的本地价格表：

```python
retail = CheckoutContext(
    tenant_id="retail",
    lookup_price_cents=make_price_lookup({"book": 2500}),
    ...,
)

wholesale = CheckoutContext(
    tenant_id="wholesale",
    lookup_price_cents=make_price_lookup({"book": 2000}),
    ...,
)
```

测试不需要 monkeypatch 全局变量，也不需要访问网络。生产代码可以注入真实实现，节点逻辑保持不变。

## 7. 生命周期与设计原则

### 7.1 每次调用重新提供

Runtime Context 是 run-scoped。新的 `invoke` 或 `stream` 应提供本次运行需要的 context。不要期待上一次调用的 context 自动延续。

第 05 章会展示：即使同一 `thread_id` 的 State 能通过 checkpointer 延续，context 仍由每次调用显式提供。

### 7.2 默认按只读设计

Context 中适合放：

- 已认证的用户和租户标识。
- 语言、区域、功能开关。
- 数据仓库、客户端、clock、ID 生成器等依赖。
- 一次请求期间稳定的策略配置。

不适合放：

- 需要由节点逐步更新的草稿。
- 重试次数、当前步骤、最终结果。
- 需要跨会话保留的用户偏好。
- 希望通过 checkpoint 回放的业务事实。

后三类分别属于 State、checkpoint 或 Store。

### 7.3 Context 未自动保存，不等于绝对保密

LangGraph checkpointer 不会像保存 State 那样自动保存 runtime context，但你的日志、异常、trace 或业务代码仍可能记录它。API Key、token 等敏感信息仍需遵守最小暴露和脱敏原则。

## 8. 常见坑

### 8.1 继续使用 config_schema

不推荐：

```python
StateGraph(MyState, config_schema=MyContext)
```

LangGraph 1.0 中改用：

```python
StateGraph(MyState, context_schema=MyContext)
```

### 8.2 把 context 当成会话记忆

Context 不会因为 `thread_id` 相同就自动保留。会话 State 延续属于 checkpointer，跨 thread 长期记忆属于 Store。

### 8.3 节点修改共享 context

可变 context 会让后续节点依赖执行顺序，破坏可预测性。优先使用 `@dataclass(frozen=True)`、不可变集合和只读接口。

### 8.4 把用户 prompt 当作权限来源

权限、租户和用户 ID 必须来自可信应用边界。模型输出和普通 State 不能提升权限。

### 8.5 使用模块级全局变量替代注入

全局客户端看似简单，但会让并发租户配置、测试隔离和资源生命周期变得困难。显式 context 让依赖可见、可替换。

### 8.6 把所有配置都复制进 State

只有后续业务逻辑真正需要更新、持久化或回放的值才应进入 State。运行依赖留在 Context，长期事实进入 Store。

### 8.7 认为 schema 会自动实例化 context

`context_schema=MyContext` 声明类型契约，不保证把任意字典转换成 `MyContext`。调用处传入与节点约定一致的对象。

## 9. 练习

1. 给 `request_context.py` 增加 `department`，让只有同部门用户可以读取资源。
2. 增加 `audit` 节点，只把必要的 `user_id` 和授权结果复制进 State；观察输出字段如何变化。
3. 给 `dependency_injection.py` 注入一个固定 clock，在 summary 中输出确定性时间。
4. 写两个 context，分别使用成功查询器和抛出异常的查询器，验证节点错误传播。
5. 将 context dataclass 改为可变对象，尝试在第一个节点修改它，再解释这种设计为什么危险。

## 10. 自检

先不看答案：

1. `thread_id` 应放在 context 还是 config？
2. 节点生成的草稿应放在 State 还是 context？
3. 数据库客户端适合进入 checkpoint 吗？
4. 同一张图能否为不同调用传入不同 context？
5. context 中的值会自动出现在最终 State 吗？
6. 长期用户偏好是否应只放在 runtime context？

答案：

1. 放在 `config={"configurable": {"thread_id": ...}}`；它是执行配置。
2. State；草稿会被节点更新，并可能需要持久化或回放。
3. 不适合；客户端属于运行依赖，放在 context。
4. 可以；context 是每次调用提供的。
5. 不会，除非节点显式返回它们作为 State update。
6. 不应；context 只服务当前调用，长期偏好应放 Store。

运行机器自检：

```bash
conda run -n langgraph python langgraph/04-runtime-context/examples/request_context.py --self-test
conda run -n langgraph python langgraph/04-runtime-context/examples/dependency_injection.py --self-test
```

看到两个 `self-test passed`，说明本章示例在本地通过。

## 11. 本章总结

- State 保存会变化、需要在节点间流动的业务数据。
- Runtime Context 保存单次调用稳定的身份、租户、区域和依赖。
- LangGraph 1.0 使用 `context_schema` 与 `Runtime[Context]`。
- context 通过 `invoke(..., context=...)` 显式提供，不会自动成为 State 或长期记忆。
- `config`、context、checkpoint、Store 与模型上下文窗口各有独立职责。
- 显式依赖注入能避免全局状态，让离线测试和多租户边界更清楚。

下一章将引入 checkpointer，让 State 能按 `thread_id` 跨调用延续，并学习历史快照、时间旅行与人工修改状态。
