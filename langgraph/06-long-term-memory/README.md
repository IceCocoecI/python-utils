# 06 · Long-term Memory：Store、Namespace 与跨 Thread 记忆

> 目标环境：Python 3.11、LangGraph 1.0.6。  
> 本章完全离线，使用 `InMemoryStore`，不需要向量模型、数据库或 API Key。

## 学完本章能做什么

你将能够：

1. 区分 thread checkpoint 与跨 thread Store。
2. 使用 `InMemoryStore` 编译带长期记忆能力的图。
3. 在节点中通过 `runtime.store` 显式读写记忆。
4. 用 namespace 隔离租户、用户和记忆类别。
5. 使用 `put/get/search/delete/list_namespaces` 管理记忆生命周期。
6. 让同一用户从不同 thread 读取共享偏好，同时隔离其他用户。
7. 解释 Store 为什么不会自动成为模型上下文，也不会随 checkpoint 时间旅行回滚。

## 1. 问题：thread 记住了会话，但用户有多个 thread

第 05 章的 checkpointer 可以让同一 thread 延续 State：

```text
thread A: turn 1 -> turn 2 -> turn 3
```

但一个用户通常有多条会话：

```text
alice
├── thread A: 旅行计划
├── thread B: 编程问题
└── thread C: 客服工单
```

如果 Alice 在 thread A 中说“我偏好深色主题”，thread B 是否应该知道？

- 把偏好只放在 thread A 的 State：thread B 看不到。
- 把 Alice 所有会话强行合并成同一 thread：不同任务互相污染。
- 每次调用都从外部手工拼接完整用户资料：调用层越来越复杂。

LangGraph 的 **Store** 用于显式保存这种不属于单一执行线程、但需要在应用范围内复用的资料。

## 2. “长期”描述作用域，不保证磁盘耐久

本章使用 `InMemoryStore`。它可以在同一进程、同一 Store 实例中跨 thread 共享数据，因此相对于 thread State 属于 long-term memory；但进程退出后数据仍会消失。

```text
Long-term scope: 跨 thread / 跨图调用
Durable storage: 跨进程重启 / 多实例共享
```

这两个维度不要混淆。`InMemoryStore` 提供前者，不提供后者。生产环境需要选择持久 Store 后端并设计数据治理。

## 3. 边界：五种数据放在哪里

| 数据 | 推荐位置 | 原因 |
|---|---|---|
| 当前问题、草稿、执行步骤 | State | 会被节点逐步更新 |
| 用户 ID、租户、依赖 | Runtime Context | 每次调用的可信只读环境 |
| 某条会话的消息和执行位置 | Checkpoint | 按 thread 保存 State 历史 |
| 跨会话偏好、事实、用户资料 | Store | 业务显式管理，可跨 thread 读取 |
| 本次送给模型的消息和检索片段 | 模型上下文窗口 | 必须控制相关性和 token 数量 |

最关键的对比：

| 维度 | Checkpointer | Store |
|---|---|---|
| 默认作用域 | `thread_id` | 由 namespace 自己设计 |
| 写入方式 | 图在执行边界自动保存 | 节点或应用显式 `put/delete` |
| 读取方式 | `get_state/get_state_history` | `get/search` |
| 主要内容 | 整体 State、执行位置、任务元数据 | 业务选择的独立记忆条目 |
| 时间旅行 | 可从历史 State 分支 | 不会自动回滚 |
| 典型用途 | 恢复、HITL、多轮线程 | 用户偏好、资料、跨 thread 知识 |

## 4. 心智模型：thread 纵向延续，Store 横向共享

```text
                         Store
             namespace=(users, alice, preferences)
                         theme=dark
                            ▲
                            │ explicit get/put
             ┌──────────────┼──────────────┐
             │              │              │
        thread A       thread B       thread C
       checkpoint      checkpoint      checkpoint
       State A1-A3     State B1-B2     State C1

             Bob 使用另一个 namespace，默认看不到 Alice 的条目
```

Checkpoint 管理每条纵向时间线，Store 提供应用设计的横向共享空间。

## 5. 核心 API

### 5.1 创建 Store 并在 compile 时注入

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
graph = builder.compile(store=store)
```

Store 与 checkpointer 相互独立，可以只用其中一个，也可以同时使用：

```python
graph = builder.compile(
    checkpointer=InMemorySaver(),
    store=InMemoryStore(),
)
```

### 5.2 节点通过 Runtime 访问 Store

```python
def memory_node(state, runtime: Runtime[UserContext]):
    store = runtime.store
    if store is None:
        raise RuntimeError("this graph requires a Store")
    ...
```

只有在编译时传了 Store，`runtime.store` 才可用。显式检查比稍后出现模糊的 `NoneType` 错误更容易诊断。

### 5.3 Namespace 是字符串元组

```python
namespace = ("users", user_id, "preferences")
```

更完整的多租户结构可以是：

```python
namespace = (
    "tenants", tenant_id,
    "users", user_id,
    "notes",
)
```

Namespace 既是组织结构，也是最重要的逻辑隔离边界。它不是路径字符串，不要手工用 `"/".join(...)` 替代结构化元组。

### 5.4 put：按 key 保存字典 value

```python
store.put(
    namespace,
    "theme",
    {"value": "dark"},
)
```

同一 namespace 下相同 key 再次 `put` 会更新该条目。因此 key 应稳定表达记忆身份：

- 偏好可以用 `theme`、`language`。
- 笔记可以用业务 note ID。
- 事件型记忆可以使用稳定事件 ID，避免重试时重复创建。

当前 API 的 value 应是 `dict[str, Any]`。生产中优先保存可序列化、结构稳定的数据，不要把客户端、打开的文件或协程对象放进去。

### 5.5 get：精确读取一条记忆

```python
item = store.get(namespace, "theme")
if item is not None:
    theme = item.value["value"]
```

返回的是 Store Item，不是 value 本身。常用字段包括 `key`、`namespace`、`value` 和时间元数据。

### 5.6 search：列出或过滤条目

```python
items = store.search(
    namespace,
    filter={"tag": "python"},
    limit=100,
)
```

本章使用结构化 filter，完全离线。语义搜索需要为 Store 配置索引和 embedding，不是简单传入 `query` 就自动获得高质量向量检索；这应作为独立主题学习和测试。

### 5.7 delete 与 namespace 检查

```python
store.delete(namespace, key)

namespaces = store.list_namespaces(
    prefix=("tenants", tenant_id),
)
```

长期记忆必须有删除能力。用户纠正、隐私请求、保留期和业务注销都可能要求清理旧数据。

## 6. 示例一：跨 Thread 共享，跨用户隔离

文件：[examples/cross_thread_memory.py](./examples/cross_thread_memory.py)

运行：

```bash
conda run -n langgraph python langgraph/06-long-term-memory/examples/cross_thread_memory.py
conda run -n langgraph python langgraph/06-long-term-memory/examples/cross_thread_memory.py --self-test
```

示例同时启用：

```python
builder.compile(
    checkpointer=InMemorySaver(),
    store=InMemoryStore(),
)
```

### 6.1 Context 提供可信用户身份

```python
@dataclass(frozen=True)
class UserContext:
    user_id: str
```

节点从 `runtime.context.user_id` 构造 namespace：

```python
namespace = (
    "users",
    runtime.context.user_id,
    "preferences",
)
```

不要从模型输出或普通 prompt 中提取一个未经验证的 `user_id` 后直接访问 Store。真实应用应由认证层构造 `UserContext`。

### 6.2 Alice 在 thread A 保存

```python
graph.invoke(
    {
        "request": {
            "operation": "remember",
            "key": "theme",
            "value": "dark",
        }
    },
    {"configurable": {"thread_id": "alice-thread-a"}},
    context=UserContext(user_id="alice"),
)
```

这里特意把一次入口请求放进整体覆盖的 `request` channel，而没有把
`operation`、`key`、`value` 平铺为三个持久 state channel。同一个 thread 的新输入会与
checkpoint State 按 channel 合并；如果第二次输入省略平铺的 `value`，旧值仍可能留在 State
中。嵌套字典没有 reducer，新的 `request` 会整体替换旧请求，因此缺失字段会被节点校验拒绝，
不会误用上一轮的值。

```python
# 同一个 thread：第一次保存 theme=dark。
graph.invoke(
    {
        "request": {
            "operation": "remember",
            "key": "theme",
            "value": "dark",
        }
    },
    config,
    context=alice,
)

# 第二次请求没有 value：抛出 ValueError，而不是保存 language=dark。
graph.invoke(
    {"request": {"operation": "remember", "key": "language"}},
    config,
    context=alice,
)
```

整体覆盖 request envelope 适合表达“每轮都必须完整提交”的临时输入；真正需要跨轮累积的
对话、进度或聚合结果则应放在独立的持久 channel，并明确 reducer 语义。

节点显式执行：

```python
store.put(
    ("users", "alice", "preferences"),
    "theme",
    {"value": "dark"},
)
```

### 6.3 Alice 在 thread B 读取

新 thread 的 checkpoint State 不包含 thread A 的状态，但它使用同一个 Store 和同一个用户 namespace：

```text
thread A State != thread B State
Alice Store namespace == Alice Store namespace
```

因此 thread B 能得到 `theme=dark`。

### 6.4 Bob 默认读不到 Alice 的记忆

Bob 的 namespace 是：

```python
("users", "bob", "preferences")
```

即使 key 同样叫 `theme`，也与 Alice 的条目不同。

注意：这种隔离由应用选择 namespace 实现，`InMemoryStore` 不会理解“alice”和“bob”代表权限主体。若代码构造了错误 namespace，Store 不会替你阻止数据泄漏。

## 7. 示例二：分层 Namespace、过滤与删除

文件：[examples/namespaces_and_search.py](./examples/namespaces_and_search.py)

运行：

```bash
conda run -n langgraph python langgraph/06-long-term-memory/examples/namespaces_and_search.py
conda run -n langgraph python langgraph/06-long-term-memory/examples/namespaces_and_search.py --self-test
```

示例使用：

```python
(
    "tenants", tenant_id,
    "users", user_id,
    "notes",
)
```

### 7.1 Namespace 设计从隔离需求出发

逐段含义：

```text
tenants / acme / users / alice / notes
          │              │       │
          租户           用户    记忆类别
```

这种结构允许：

- Alice 和 Bob 使用相同 note ID 而不冲突。
- 不同租户中同名用户互不影响。
- 同一用户的 preferences、notes、facts 分开管理。
- 运维工具按租户前缀列出 namespace。

不要在所有数据前都省略 tenant。多租户系统中，只有 user ID 通常不足以建立全局隔离。

### 7.2 Value 保存可演进结构

笔记 value：

```python
{
    "text": "Runtime carries Context",
    "tag": "python",
}
```

结构化字典比单个字符串更容易：

- 添加来源、置信度和版本。
- 使用 filter 查找。
- 迁移 schema。
- 做审计和数据治理。

### 7.3 Filter 不是语义搜索

```python
store.search(namespace, filter={"tag": "python"})
```

这表示精确的元数据过滤。它不会理解“编程”和“Python”在语义上相关。语义记忆通常还需要：

1. embedding 模型。
2. Store 索引配置。
3. query 构造和召回数量。
4. 相关性、权限和 token 预算过滤。
5. 可重复的离线评估集。

本章不为了展示一个 API 而偷偷调用网络 embedding。

### 7.4 删除是完整记忆系统的一部分

示例删除 Alice 的 `n1` 后再次搜索，只剩 `n3`。Bob 使用相同 key 的 `n1` 不受影响，因为 namespace 不同。

## 8. Store 不会自动成为模型记忆

给图传入 Store，只代表节点可以访问它。LangGraph 不会自动：

- 猜测应该保存哪些内容。
- 把每句话写入长期记忆。
- 检索所有相关用户资料。
- 把 Store 内容塞入模型 prompt。
- 解决模型上下文窗口上限。

一个完整的模型记忆读取流程通常是：

```text
trusted user context
        │
        ▼
derive namespace
        │
        ▼
retrieve candidate memories
        │
        ▼
filter by permission / relevance / freshness
        │
        ▼
select a bounded subset
        │
        ▼
render into model context
```

写入流程也应有明确策略：

```text
conversation/event
   -> decide whether it is durable
   -> normalize and validate
   -> choose namespace and stable key
   -> put/update/delete
```

“把所有消息永久保存”通常不是好的记忆策略，会带来噪声、隐私、成本和错误事实累积。

## 9. Store 与时间旅行不是同一事务

第 05 章可以从旧 checkpoint 分支，但 Store 是独立系统：

```text
checkpoint time travel  --X--> automatic Store rollback
```

如果一个历史节点曾执行 `store.put()`，从旧 checkpoint 重跑它可能再次写 Store。生产设计需要考虑：

- 稳定 key 和幂等 upsert。
- 写入事件 ID。
- 记忆版本或来源 checkpoint ID。
- 是否允许旧分支覆盖新记忆。
- checkpoint 与 Store 写入并非天然原子事务。

本章示例使用稳定偏好 key 和笔记 ID，使重复 `put` 表现为更新，而不是无限追加重复条目。

## 10. 记忆 Schema 与治理

长期记忆应比临时 State 更谨慎。可考虑为 value 保存：

```python
{
    "value": "dark",
    "source": "explicit_user_preference",
    "confidence": 1.0,
    "schema_version": 1,
    "updated_at": "...",
}
```

设计问题包括：

- 这条信息是用户明确表达，还是模型推断？
- 用户后来说了相反内容，更新还是新增？
- 哪些节点允许写入？
- 多久过期？
- 用户如何查看、纠正和删除？
- 是否包含敏感或受监管数据？
- namespace 是否包含完整租户隔离维度？
- 检索结果是否在进入模型前再次做权限检查？

记忆能力越强，数据治理要求越高。

## 11. InMemoryStore 的边界

`InMemoryStore` 适合：

- 离线教程。
- 单元测试。
- 本地原型。
- 验证 namespace 和记忆策略。

它不提供：

- 进程重启后的数据保留。
- 多进程或多机器共享。
- 生产备份和恢复。
- 自动访问控制。
- 自动加密和合规策略。
- 自动高质量语义索引。

生产后端选择应单独验证同步/异步 API、并发、索引、TTL、迁移和删除语义。

## 12. 常见坑

### 12.1 以为 Store 会自动记住一切

不会。节点必须显式 `put/get/search/delete`，应用必须定义写入和召回策略。

### 12.2 每次请求创建新的 InMemoryStore

新实例是空 Store。跨调用共享要求复用同一实例或使用真正共享的持久后端。

### 12.3 从不可信 State 构造用户 namespace

攻击者可能修改 `user_id` 读取他人记忆。身份和租户应来自认证后的 Runtime Context，并在服务边界校验。

### 12.4 Namespace 缺少租户或记忆类别

所有数据挤在同一空间会导致 key 冲突、权限边界模糊和难以清理。

### 12.5 把 thread_id 当作长期用户 namespace

这会让用户每开一个新会话就失去旧偏好。thread 标识执行实例，user namespace 标识长期主体。

### 12.6 把大型对象或客户端作为 value

Store value 应是稳定、可序列化的结构化数据。运行依赖放 Runtime Context。

### 12.7 把所有 Store 结果放进模型 prompt

会引入无关信息、token 浪费和隐私风险。先过滤，再选择有限的相关记忆。

### 12.8 认为 checkpoint 时间旅行会回滚 Store

不会。两者生命周期独立，重放节点还可能再次产生 Store 写入。

### 12.9 只实现保存，不实现纠正和删除

长期记忆一定会过期或出错。更新、删除、审计和保留期不是可选附加项。

### 12.10 把 InMemoryStore 当生产数据库

进程结束后数据消失。这里的“长期”是跨 thread 作用域，而非磁盘耐久承诺。

## 13. 练习

1. 给偏好示例增加 `forget` 操作，使用 `store.delete()` 删除指定偏好。
2. 让 Alice 在三个不同 thread 中读取同一个 `language` 偏好，验证 checkpoint 各自独立。
3. 给笔记 value 增加 `source` 和 `schema_version`，再按 `source` 过滤。
4. 新增 tenant `globex`，使用相同 user ID 和 note ID，验证租户隔离。
5. 用同一 key 连续 `put` 两次，观察它是更新还是新增两条记录。
6. 从第 05 章的历史 checkpoint 重跑一个写 Store 的节点，设计 stable operation ID 防止重复写。
7. 设计一个“只保存用户明确说我喜欢……”的记忆策略，并列出不应保存的敏感信息。

## 14. 自检

先不看答案：

1. Checkpoint 和 Store 的默认隔离维度有什么不同？
2. `compile(store=...)` 后，用户偏好会自动写入吗？
3. 为什么 user ID 应来自 Runtime Context？
4. Alice 在 thread A 保存的 Store 条目，thread B 能否读取？
5. `InMemoryStore` 能否跨进程重启保留数据？
6. 时间旅行到旧 checkpoint 会自动恢复当时的 Store 吗？
7. `filter={"tag": "python"}` 是语义搜索吗？
8. Store 中有 100 条记忆时，是否应全部加入模型上下文？

答案：

1. Checkpoint 默认按 thread；Store 按应用设计的 namespace。
2. 不会；业务节点必须显式调用 Store API。
3. 它属于可信身份边界，不能让 prompt 或模型输出任意选择他人 namespace。
4. 可以，前提是两次调用复用同一 Store，并使用同一用户 namespace。
5. 不能；它只存在于当前进程内存。
6. 不会；checkpoint 与 Store 是两个独立系统。
7. 不是；它是结构化元数据过滤。
8. 不应；应按权限、相关性、新鲜度和 token 预算选择有限子集。

运行机器自检：

```bash
conda run -n langgraph python langgraph/06-long-term-memory/examples/cross_thread_memory.py --self-test
conda run -n langgraph python langgraph/06-long-term-memory/examples/namespaces_and_search.py --self-test
```

## 15. 本章总结

- Checkpoint 延续一条 thread 的 State，Store 保存跨 thread 的业务记忆。
- Store 由节点显式读写，不会自动决定记什么、取什么或放进 prompt。
- Runtime Context 提供可信用户和租户，namespace 据此建立隔离边界。
- `put/get/search/delete/list_namespaces` 构成最小记忆生命周期。
- 相同 key 在不同 namespace 中是不同条目。
- 模型上下文窗口只是 Store 召回结果中的有限、经过筛选的一部分。
- Checkpoint 时间旅行不会自动回滚 Store，重放写节点必须考虑幂等。
- `InMemoryStore` 能跨 thread，但不能跨进程重启；“长期”首先描述作用域。

到这里，State、Runtime Context、checkpoint、Store 和模型上下文窗口已经有了清晰边界。后续工具调用、人机交互、多 Agent 和综合项目都应建立在这套边界之上。
