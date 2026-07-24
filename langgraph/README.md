# LangGraph 系统学习仓库

这不是一组彼此孤立的 API 示例，而是一套围绕 **状态、控制流、持久化和可恢复执行** 组织的 LangGraph 课程。所有核心示例默认离线运行，不需要模型服务、网络或 API Key；先把编排机制看清，再把确定性的模拟节点替换成真实模型和工具。

课程基于当前本机环境验证：

- Python 3.11.14
- LangGraph 1.0.6
- langchain-core 1.2.7
- langgraph-checkpoint 4.0.0

代码只使用 LangGraph 1.x 的公共接口，并在 `VERSION_MATRIX.md` 中说明版本边界。

## 1. 你最终应该掌握什么

学完后，不只是能画出一张图，而是能回答并实现这些问题：

1. 哪些数据属于业务状态，哪些属于运行时上下文，哪些应该放到长期 Store？
2. 同一字段被多个节点更新时，为什么需要 Reducer？并行写入发生冲突时如何处理？
3. 如何表达分支、循环、动态并行、子图和跨图跳转？
4. `thread_id`、checkpoint、短期记忆和长期记忆是什么关系？
5. 如何暂停工作流，让人审批或修改状态，然后从原位置恢复？
6. 如何流式观察节点更新、自定义事件和子图事件？
7. 如何让工具调用、重试、幂等性和错误处理形成可靠闭环？
8. 什么时候应该用 LangGraph，什么时候普通函数或 LangChain 高层 Agent 已经足够？

## 2. 核心心智模型

LangGraph 可以先理解成一个按“超级步”执行的有状态运行时：

```text
输入
  |
  v
+-------------------- Compiled StateGraph --------------------+
|                                                             |
|  当前 State --读取--> Node --返回部分更新--> Reducer 合并     |
|      ^                                         |             |
|      |                                         v             |
|      +----------- Edge / Router / Command -----+             |
|                                                             |
+-----------------------+-------------------------------------+
                        |
             checkpoint | stream events
                        v
                 持久化 / 观察 / 恢复

运行时旁路：Context 提供本次运行的依赖和身份
跨线程旁路：Store 保存可跨会话复用的长期信息
```

最重要的三条边界：

| 容器 | 生命周期 | 典型内容 | 不应放什么 |
|---|---|---|---|
| `State` | 一次线程的执行过程，可被 checkpoint | 消息、步骤、路由结果、中间产物 | 数据库连接、静态服务对象 |
| `Context` | 一次 `invoke/stream` 调用 | `user_id`、租户、权限、依赖、请求配置 | 需要在节点间逐步变化的业务数据 |
| `Store` | 跨线程、跨对话 | 用户偏好、长期事实、共享知识 | 每轮临时消息、未确认的中间状态 |

进一步阅读：[概念地图](./CONCEPT_MAP.md)。

## 3. 课程目录

每一章都遵循同一结构：问题背景 -> 心智模型 -> API -> 代码拆解 -> 常见坑 -> 练习 -> 自检。`examples/` 中的脚本都有 `--self-test`，适合先运行再阅读。

| 阶段 | 模块 | 重点 | 学完后的交付物 |
|---|---|---|---|
| 入门 | [01 首个图](./01-foundations/README.md) | `StateGraph`、节点、边、编译、输入/输出 schema | 能从零写线性图并解释一次状态更新 |
| 基础 | [02 状态与 Reducer](./02-state-and-reducers/README.md) | `TypedDict`、`Annotated`、自定义 Reducer、`add_messages` | 能设计无并行冲突的状态结构 |
| 基础 | [03 控制流](./03-control-flow/README.md) | 条件边、循环、`Command`、终止条件 | 能把业务规则翻译成可测试路由 |
| 单模块 | [04 运行时上下文](./04-runtime-context/README.md) | `context_schema`、`Runtime`、依赖注入、配置边界 | 能隔离不同用户与运行环境 |
| 单模块 | [05 检查点与短期记忆](./05-persistence/README.md) | `InMemorySaver`、`thread_id`、历史、时间旅行 | 能续接会话并检查历史状态 |
| 单模块 | [06 长期记忆](./06-long-term-memory/README.md) | `InMemoryStore`、namespace、读写时机、记忆治理 | 能跨线程读取用户偏好且避免串数据 |
| Agent | [07 工具与 Agent 循环](./07-tools-and-agents/README.md) | ToolNode、`tools_condition`、模型-工具循环、安全边界 | 能解释和测试一次完整工具调用 |
| Agent | [08 Human-in-the-loop](./08-human-in-the-loop/README.md) | `interrupt`、`Command(resume=...)`、状态修改 | 能暂停审批并从断点恢复，而非从头重跑 |
| 工程 | [09 流式与可靠性](./09-streaming-and-resilience/README.md) | stream modes、writer、retry、幂等、Functional API | 能观察执行过程并控制失败副作用 |
| 进阶 | [10 子图与多智能体](./10-subgraphs-and-multi-agent/README.md) | direct/wrapper 子图、`Command.PARENT` handoff、`Send`、supervisor | 能设计父子状态协议、跨图交接和动态协作 |

综合学习不是把更多节点堆在一张图里。请在单模块之后进入 [综合 Lab：可恢复客服工作流](./labs/01-support-agent/README.md)，它把以下能力连成一个可测试系统：

```text
Runtime Context -> 长期偏好读取 -> 意图路由
                                  |-> FAQ 子流程 -------|
                                  |-> 账单并行调查 -----|-> 汇总
                                  |-> 高风险退款 -> 人工审批

全程：checkpoint + thread_id + stream + trace + Store
```

## 4. 快速开始

所有命令都从仓库根目录 `python-utils/` 执行。

```bash
conda activate langgraph
python --version
python -c "import importlib.metadata as m; print(m.version('langgraph'))"
```

先运行最小示例：

```bash
python langgraph/01-foundations/examples/hello_state_graph.py
python langgraph/01-foundations/examples/hello_state_graph.py --self-test
```

运行全部离线示例的 smoke test：

```bash
python langgraph/scripts/smoke_test.py
```

运行标准库测试：

```bash
python -m unittest discover -s langgraph/tests -v
```

检查 Markdown 本地链接：

```bash
python langgraph/scripts/check_links.py
```

不激活环境时，等价写法是：

```bash
conda run -n langgraph python langgraph/scripts/smoke_test.py
conda run -n langgraph python -m unittest discover -s langgraph/tests -v
```

环境细节见 [ENVIRONMENT.md](./ENVIRONMENT.md)，常见错误见 [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)。

## 5. 推荐学习方式

不要只“看懂”。每章按下面的闭环执行：

```text
读 README 的问题与心智模型
        |
        v
运行 example，观察最终状态和事件
        |
        v
改一个状态字段、路由条件或故障条件
        |
        v
运行 --self-test 和单元测试
        |
        v
回答自检题，记录失败原因
```

评价是否掌握一个主题时，用四级标准：

| 层级 | 判断标准 |
|---|---|
| 能运行 | 能在当前环境复现输出 |
| 能解释 | 能说明状态、节点、边和副作用的关系 |
| 能修改 | 改需求后能独立调整 schema、路由和测试 |
| 能取舍 | 能说明替代方案、失败模式和生产边界 |

可直接选择一条路线：

- [完整路线](./routes/full-track.md)：适合第一次系统学习，约 4 周。
- [实践速通路线](./routes/practice-track.md)：已有 LangChain/Python 基础，约 5 天。
- [生产工程路线](./routes/production-track.md)：重点关注可靠性、持久化、观测与测试。

阶段验收见 [assessments/README.md](./assessments/README.md)，学习记录模板见 [progress/README.md](./progress/README.md)。

## 6. 为什么示例默认不用真实 LLM

真实模型会引入网络、密钥、费用、随机输出、限流和模型协议差异。它们会遮住本课程真正要观察的东西：

- 节点收到的 state 是什么；
- 节点返回了哪一部分更新；
- reducer 怎样合并并发更新；
- router 为什么选择某条边；
- checkpoint 在哪个时刻保存；
- resume 是从断点继续还是错误地重新提交输入；
- Store 的 namespace 是否隔离用户。

因此示例使用确定性的规则节点和 `AIMessage(tool_calls=...)` 来模拟模型决策。掌握机制后，替换点通常只是一个节点：

```python
def call_model(state: AgentState, runtime: Runtime[AppContext]) -> dict:
    response = runtime.context.model.invoke(state["messages"])
    return {"messages": [response]}
```

模型客户端属于依赖，应通过 context 或闭包注入；不要把不可序列化客户端写进 state。课程不默认安装 `langchain-openai`，以与你当前的 `langgraph` Conda 环境保持一致。

## 7. Graph API 与 Functional API 怎么选

| 场景 | 优先选择 |
|---|---|
| 需要显式可视化的分支、循环、多角色流程 | Graph API |
| 已有普通 Python 控制流，主要需要持久化和恢复 | Functional API |
| 团队需要审查状态 schema 与路由拓扑 | Graph API |
| 短小、线性、任务式工作流 | Functional API |
| 只是两三个无状态函数顺序调用 | 普通 Python，未必需要 LangGraph |

本课程以 Graph API 为主，第 09 章单独对比 Functional API。两者共享 checkpoint、Store、interrupt 等运行时能力，不需要把它们当作两个完全不同的框架。

## 8. 生产化边界

教学中的内存组件会在进程结束后丢失：

- `InMemorySaver` 只适合教学、测试和单进程开发；
- `InMemoryStore` 只适合教学、测试和单进程开发；
- 生产中应选择官方支持的持久化 checkpointer/store，并设计迁移、加密、TTL、权限与删除策略；
- 有副作用的节点必须考虑重放和幂等；checkpoint 不会自动撤销已发送邮件或已扣款事务；
- 高风险工具必须做参数校验、权限检查、审批和审计，不应只相信模型产生的 tool call；
- 图的行为应以状态断言、执行路径和恢复测试验证，不能只看最终自然语言。

第 09 章和 [生产路线](./routes/production-track.md) 会系统展开这些问题。

## 9. 仓库约定

- 顶层目录已经叫 `langgraph/`，故意不创建 `langgraph/__init__.py`。否则它会遮蔽 Conda 环境中安装的官方 `langgraph` 包。
- 单模块脚本可以直接运行，综合项目使用内部包名 `support_agent`。
- 核心示例不访问网络、不读环境变量、不调用付费服务。
- 示例中的内存持久化不是生产数据库，文档会明确指出替换边界。
- Python 代码使用 ASCII；中文解释放在 README 和命令行输出中需要时才使用。
- 所有新增本地链接必须通过 `scripts/check_links.py`。

## 10. 官方资料

课程用于建立结构化理解，不替代官方文档。遇到版本细节时，以对应版本 API reference 和 release notes 为准：

- LangGraph overview: <https://docs.langchain.com/oss/python/langgraph/overview>
- LangGraph API reference: <https://reference.langchain.com/python/langgraph/>
- LangGraph releases: <https://github.com/langchain-ai/langgraph/releases>

建议先完成 01-03，再按你的目标选择 04-10；不要一开始就复制一个大型 Agent 模板，因为那会把状态设计、控制流和持久化问题混在一起。
