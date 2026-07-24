# 10 · 子图与多 Agent：划分边界、动态分工与并行汇总

> 目标环境：Python 3.11、LangGraph 1.0.6。  
> 本章完全离线，用确定性函数讲清子图映射、跨图 handoff、`Send`、reducer 和 supervisor/worker 协作语义。

## 学完本章能做什么

你将能够：

1. 判断什么时候应该拆子图，而不是继续把所有节点堆进一张大图。
2. 把编译后的 `StateGraph` 直接注册为父图节点，并通过共享 state key 传递数据。
3. 根据父子 state 是否同构，在直接注册 compiled graph 和 wrapper 显式映射之间选择。
4. 验证 LangGraph 1.0.6 中两种嵌套方式都能产生子图 namespace 事件。
5. 使用 `Command(graph=Command.PARENT, goto=...)` 从子图 handoff 到父图节点。
6. 解释跨图 update 在哪一层合并，以及共享 `messages` reducer 为什么重要。
7. 使用 `Send` 按运行时计划动态创建多个 worker 调用。
8. 为并行写入的 state channel 配置 reducer。
9. 解释 LangGraph 的 superstep 屏障为什么会让汇总节点等待全部 worker。
10. 将 supervisor 的规划职责与 worker 的执行职责分开。
11. 编写不依赖并行完成顺序的稳定汇总和测试。

## 1. 从“大图”走向有边界的工作流

前面的章节已经覆盖工具循环、人工审批、流式和重试。把这些能力放进真实系统后，一张图很快会出现几十个节点：

```text
输入规范化 -> 研究 -> 搜索 -> 证据检查 -> 撰写 -> 审批 -> 发布
                  |                    |
                  +-- 多个工具循环     +-- 多个修改分支
```

如果所有细节都暴露在同一层，会产生三个问题：

- 阅读者无法一眼看出业务阶段，只能看到大量实现节点。
- 独立模块难以单测、复用和分配给不同团队维护。
- 流式事件、checkpoint 和错误定位缺少清晰的层级。

子图提供静态模块边界。父图只表达“先研究、再生成报告”，研究子图内部再表达“找来源、提取笔记”：

```text
父图
START -> [research 子图] -> render -> END
              |
              +-- find_sources -> extract_notes
```

但子图并不解决所有规模问题。如果 worker 数量只能在运行时根据请求决定，就需要动态 fan-out：

```text
                          +-> analyst -+
request -> supervisor ----+-> engineer +-> synthesize
                          +-> writer --+
```

本章的三个核心概念分工如下：

| 概念 | 解决的问题 | 结构何时确定 |
|---|---|---|
| 原生子图 | 封装一个内部有固定拓扑的模块 | 构图/编译时 |
| `Command.PARENT` handoff | 从子图把状态增量和控制权交给父图目的节点 | 子图运行时 |
| `Send` | 为本次运行动态创建若干节点调用 | 运行时 |

## 2. 先建立三个层次的心智模型

### 2.1 图是模块，不只是函数列表

一个编译后的子图包含：

- 自己的 state schema；
- 自己的节点与边；
- 自己的执行层级和流式事件；
- 在合适配置下可参与 checkpoint、interrupt 和调试。

因此，子图最重要的价值不是少写一个函数，而是保留一个可被运行时识别的工作流边界。

### 2.2 多 Agent 首先是职责划分

“多个 Agent”不等于必须创建多个聊天机器人。更可靠的定义是：

```text
supervisor：决定需要哪些角色、分配什么输入、如何验收
worker：只完成一个边界明确的任务，返回结构化结果
synthesizer：等结果齐全后排序、校验和汇总
```

本章不调用 LLM，但控制流与真实多 Agent 系统相同。以后可以把确定性函数替换成绑定不同提示词、工具和权限的模型节点。

### 2.3 并行更新是“合并”，不是共享可变对象

多个 worker 在同一个 superstep 中各自读取输入快照并返回 update。运行时随后合并这些 update：

```text
同一份上游状态
   |        |        |
worker A worker B worker C
   |        |        |
 update A  update B  update C
        \    |    /
          reducer
             |
       下一 superstep
```

节点不应通过修改同一个全局 list 来通信。它们返回独立增量，reducer 明确定义怎样合并。

## 3. 直接注册与包装调用：差异在状态映射

父图调用子图有两种常用方式。父子 schema 共享字段时，可以直接注册编译图：

```python
research_graph = build_research_subgraph()

parent.add_node("research", research_graph)
```

运行时按父子 schema 的同名 channel 传递输入，并把子图输出合并回父 state。若字段名或形状不同，可以使用 wrapper 做显式协议转换：

```python
research_graph = build_research_subgraph()

def run_research(state):
    child_input = {
        "topic": state["report_topic"],
        "sources": [],
        "notes": "",
    }
    child_output = research_graph.invoke(child_input)
    return {"research_notes": child_output["notes"]}

parent.add_node("research", run_research)
```

这里必须修正一个常见但不准确的说法：**在 LangGraph 1.0.6 中，wrapper 内同步调用 compiled graph 的 `invoke()`，并不会自动隐藏子图 stream namespace。** 当该调用发生在正在执行的 LangGraph 节点内时，本章的可执行对照表明，父图使用 `stream(..., subgraphs=True)` 后，两种写法都会产生非空 namespace 的内部节点事件。

因此不能用“是否看得到 namespace”作为选型依据。真正差异是状态协议由谁负责：

| 方式 | 输入/输出映射 | 更适合 | 主要风险 |
|---|---|---|---|
| 直接注册 compiled graph | LangGraph 按父子同名 channel 自动映射 | schema 有明确共享字段 | 同名字段的类型或 reducer 不一致 |
| wrapper 内 `subgraph.invoke()` | wrapper 明确构造 child input、筛选 child output | schema 名称/形状不同，需要校验或转换 | 返回完整 child state，误覆盖父字段或重复合并累积 channel |

这条结论限定于“子图在当前父节点执行上下文中被嵌套调用”。在另一个进程中独立调用子图，是另一条顶层运行，不应期待它自动出现在原父图的 namespace 中。checkpoint、interrupt 和异步并发等更复杂路径也要写端到端测试，不能只从一次 stream 输出推断。

## 4. 示例一：把编译后的子图直接放进父图

文件：[examples/native_subgraph.py](./examples/native_subgraph.py)

运行：

```bash
conda run -n langgraph python langgraph/10-subgraphs-and-multi-agent/examples/native_subgraph.py
conda run -n langgraph python langgraph/10-subgraphs-and-multi-agent/examples/native_subgraph.py --show-events
conda run -n langgraph python langgraph/10-subgraphs-and-multi-agent/examples/native_subgraph.py --compare-wrapper
conda run -n langgraph python langgraph/10-subgraphs-and-multi-agent/examples/native_subgraph.py --self-test
```

### 4.1 两层 state schema

研究子图只声明自己需要的字段：

```python
class ResearchState(TypedDict):
    topic: str
    sources: list[str]
    notes: str
```

父图再增加最终报告：

```python
class ReportState(TypedDict):
    topic: str
    sources: list[str]
    notes: str
    report: str
```

共享字段形成父子图的数据接口：

| 字段 | 父图 | 子图 | 方向 |
|---|---|---|---|
| `topic` | 有 | 有 | 父图传给子图 |
| `sources` | 有 | 有 | 子图写回父图 |
| `notes` | 有 | 有 | 子图写回父图 |
| `report` | 有 | 无 | 只属于父图 |

原生子图不会凭空推断两个名字不同的字段应该互相映射。例如父图叫 `research_notes`、子图叫 `notes` 时，需要重新设计 schema 或显式做输入/输出转换。

### 4.2 构建研究子图

```python
def build_research_subgraph():
    builder = StateGraph(ResearchState)
    builder.add_node("find_sources", find_sources)
    builder.add_node("extract_notes", extract_notes)
    builder.add_edge(START, "find_sources")
    builder.add_edge("find_sources", "extract_notes")
    builder.add_edge("extract_notes", END)
    return builder.compile()
```

逐段理解：

1. `StateGraph(ResearchState)` 声明模块的状态协议。
2. `find_sources` 读取 `topic`，只返回 `sources` 增量。
3. `extract_notes` 在下一步读取已经合并的 `sources`，返回 `notes`。
4. `compile()` 得到可独立 `invoke()`、测试，也可嵌入父图的 runnable。

两个节点都返回部分状态：

```python
return {"sources": [...]}
return {"notes": "..."}
```

不必复制未变化的 `topic`。LangGraph 会按 channel 合并 update；返回完整 state 反而容易意外覆盖其他字段。

### 4.3 原生嵌入父图

```python
def build_graph():
    research = build_research_subgraph()

    builder = StateGraph(ReportState)
    builder.add_node("research", research)
    builder.add_node("render", render_report)
    builder.add_edge(START, "research")
    builder.add_edge("research", "render")
    builder.add_edge("render", END)
    return builder.compile()
```

执行过程可以分成五步：

1. 父图从 `ReportState` 中把共享字段交给 `research` 子图。
2. 子图的 `find_sources` 生成来源。
3. 子图的 `extract_notes` 读取来源并生成笔记。
4. 子图结束，共享输出字段合并回父图 state。
5. 父图 `render` 读取 `topic` 和 `notes`，写入 `report`。

这条边表达的是模块级依赖：`render` 不关心研究模块内部有几个节点，只关心子图完成后 `notes` 已可用。

### 4.4 为什么初始状态仍给空字段

示例使用：

```python
{
    "topic": topic,
    "sources": [],
    "notes": "",
    "report": "",
}
```

这让 `TypedDict` 的必填字段与运行时输入一致，也让示例的状态形状直观。生产代码也可以使用更精确的输入 schema、可选字段或独立 input/output schema，避免要求调用者提供本应由图生成的占位值。

不要把教学示例中的空字符串误解成 LangGraph 的普遍要求；真正要求来自你定义的 schema 和节点读取行为。

### 4.5 用同一输入验证 direct 与 wrapper

`--compare-wrapper` 会分别执行：

```text
direct:  parent.add_node("research", compiled_research_graph)
wrapper: parent.add_node("research", run_research)
         run_research -> compiled_research_graph.invoke(mapped_input)
```

两组事件都包含 `find_sources`、`extract_notes` 的非空 namespace。父层 `research` update 则体现映射差异：直接注册返回子图 output schema 中的 `topic/sources/notes`，wrapper 只返回它明确选择的 `sources/notes`。

`self_test()` 同时断言这两件事。也就是说，本例不是用最终报告猜测执行方式，而是直接验证嵌套事件和父层 update 契约。

## 5. 观察子图：subgraphs=True 与 namespace

普通父图更新流：

```python
graph.stream(state, stream_mode="updates")
```

主要站在父图层级看 `research` 和 `render`。无论子图是直接注册，还是由当前父节点中的 wrapper 调用，要观察嵌套执行都使用：

```python
graph.stream(
    state,
    stream_mode="updates",
    subgraphs=True,
)
```

此时每个事件形如：

```python
(namespace, update)
```

其中：

- `namespace == ()` 表示父图层级。
- 非空 tuple 表示某个嵌套运行层级。
- `update` 仍是以节点名为 key 的状态增量。

示例输出的逻辑形状类似：

```text
(('research:<运行实例标识>',), {'find_sources': {'sources': [...]}})
(('research:<运行实例标识>',), {'extract_notes': {'notes': '...'}})
((), {'research': {'topic': ..., 'sources': ..., 'notes': ...}})
((), {'render': {'report': '...'}})
```

namespace 中的实例标识是运行时标识，不应硬编码进业务逻辑或测试。稳定测试应断言：

- 至少存在一个非空 namespace；
- 非空 namespace 的更新包含预期内部节点名；
- 空 namespace 的更新包含预期父节点名。

这正是 `self_test()` 的做法。它验证层级语义，而不是依赖一次运行产生的具体 ID。

如果继续嵌套更深层子图，tuple 可以表达多级路径。消费端应把 namespace 当结构化路径处理，不要拼接字符串后再用脆弱的 `split()` 解析。

## 6. 子图的设计边界

### 6.1 什么时候值得拆子图

适合拆分：

- 内部有多个节点，但父流程只关心模块输入与输出。
- 模块需要独立测试、复用或由独立团队维护。
- 模块内部有自己的循环、工具或审批逻辑。
- 希望在 trace 和 stream 中保留清晰层级。

不一定值得拆分：

- 只有一个简单纯函数节点，没有独立状态协议。
- 父子图需要共享大量内部字段，边界比原图还复杂。
- 只是为了缩短一个源文件，而没有业务模块边界。

### 6.2 共享 schema 是公开协议

把共享 key 当作模块 API 管理：

- 名称与类型要稳定。
- 明确谁读取、谁写入。
- list 等并发 channel 要定义 reducer。
- 不把子图内部调试字段全部泄露到父图。
- 修改 schema 时同时更新父图、子图和契约测试。

### 6.3 checkpoint 不是任意嵌套调用的魔法

原生嵌入让运行时看见子图边界，但持久化行为仍取决于编译方式、checkpointer 和调用 config。涉及子图内 interrupt、跨进程恢复或每个子图独立记忆时，应为这些路径编写端到端恢复测试，不要仅凭最终输出推断 checkpoint 语义。

## 7. 跨图 handoff：从子图跳到父图目的节点

子图正常到达 `END` 后由父图的静态边继续，是“完成一个模块”。handoff 表达的是另一种意图：当前 specialist 主动把控制权和必要状态交给父图中的指定节点。

```text
父图 classify_request
       |
       v
  [billing 子图]
       |
       +-- inspect_billing_request
       |
       +-- handoff_to_parent
               |
               | Command(graph=Command.PARENT,
               |         goto="compose_billing_reply",
               |         update=...)
               v
父图 compose_billing_reply -> END
```

路由与 handoff 的区别不在于有没有条件，而在于谁做决定、目标在哪一层：

| 机制 | 决策位置 | 目标位置 | 典型用途 |
|---|---|---|---|
| 父图 conditional edge | 父图 router | 父图已知节点 | 根据输入选择 specialist |
| 子图普通 edge/`Command(goto=...)` | 子图节点 | 当前子图节点 | specialist 内部流程控制 |
| `Command.PARENT` | 子图节点 | 父图节点 | specialist 完成后交给父级协调或另一个模块 |

### 7.1 运行跨图示例

文件：[examples/handoff_parent_command.py](./examples/handoff_parent_command.py)

```bash
conda run -n langgraph python langgraph/10-subgraphs-and-multi-agent/examples/handoff_parent_command.py
conda run -n langgraph python langgraph/10-subgraphs-and-multi-agent/examples/handoff_parent_command.py --show-events
conda run -n langgraph python langgraph/10-subgraphs-and-multi-agent/examples/handoff_parent_command.py --self-test
```

它包含 billing 和 general 两条路径。general 直接由父图结束；billing 进入子图分析，再由子图 handoff 到父图 `compose_billing_reply`。

### 7.2 父子状态怎样映射

父图状态包含完整协调信息：

```python
class SupportState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: str
    specialist_note: str
    handled_by: str
```

billing 子图只声明它需要的共享字段：

```python
class BillingState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    specialist_note: str
```

| channel | 父图 | 子图 | 作用 |
|---|---|---|---|
| `messages` | `add_messages` | `add_messages` | 父输入传入子图，handoff 消息合并回父图 |
| `specialist_note` | 覆盖 | 覆盖 | 子图产出结构化分析，父目的节点读取 |
| `route` | 有 | 无 | 只属于父级协调 |
| `handled_by` | 有 | 无 | 父目的节点写最终处理者 |

共享字段不是“全局变量”。直接注册 compiled graph 后，LangGraph 依据 schema 的同名 channel 形成接口；没有出现在 `BillingState` 中的父字段不会成为 specialist 的内部协议。

`messages` 在两层都使用 `add_messages`，让新消息按 message id 追加或替换。父子若对同名 channel 定义不同类型或 reducer，handoff update 的含义就会变得不可靠。

### 7.3 `Command.PARENT` 的三个参数

handoff 节点返回：

```python
return Command(
    graph=Command.PARENT,
    goto="compose_billing_reply",
    update={
        "specialist_note": state["specialist_note"],
        "messages": [
            AIMessage(
                content="Billing specialist finished analysis.",
                id="billing-specialist",
            )
        ],
    },
)
```

三个参数必须一起理解：

1. `graph=Command.PARENT` 把目标解析层级切换到父图；省略它时，运行时会在当前 billing 子图中寻找目标。
2. `goto="compose_billing_reply"` 指向父图中的实际节点名。它是跨图协议的一部分，重命名目的节点时必须更新 handoff 和测试。
3. `update=...` 是交给父图的状态增量。父图先按自己的 channel reducer 合并，再让目的节点读取合并后的 state。

父图没有添加 `billing_specialist -> compose_billing_reply` 静态边。该路径就是由 `Command.PARENT` 创建的；否则阅读拓扑时会无法判断究竟是静态边还是 handoff 决定下一步。

### 7.4 为什么 update 只放新增消息

handoff 已经读取了完整 `state["messages"]`，但它不把整段历史再次返回，而只返回一条新 `AIMessage`：

```text
父 messages: [HumanMessage]
handoff delta: [AIMessage(id="billing-specialist")]
父 add_messages 合并后: [HumanMessage, AIMessage]
```

这种“返回 delta”写法让 reducer 的职责清晰，也避免把子图收到的历史误当作新输出。`add_messages` 能按稳定 ID 处理更新，但其他简单 append reducer 可能直接复制整段历史，因此不要依赖 reducer 替错误的输出契约兜底。

随后父图的 `compose_billing_reply` 再追加 `id="parent-final"` 的消息。自测断言最终消息 ID 顺序恰好是：

```text
user-request -> billing-specialist -> parent-final
```

### 7.5 从 stream 识别控制权已经回父图

`--show-events` 的稳定逻辑形状是：

```text
((), {'classify_request': ...})
(('billing_specialist:<实例标识>',), {'inspect_billing_request': ...})
((), {'billing_specialist': {'specialist_note': ..., 'messages': [...]}})
((), {'compose_billing_reply': ...})
```

`inspect_billing_request` 位于非空 namespace，说明它在子图内执行。handoff 的 update 以父层 `billing_specialist` update 出现，随后目的节点 `compose_billing_reply` 也位于空 namespace。这比只检查最终文本更直接地验证了跨图路径。

### 7.6 目标节点的静态类型与运行时验证

同一张图内常用 `Command[Literal["next_node"]]` 帮助绘图和静态校验。但这里的 `compose_billing_reply` 只存在于父图；在目标版本 1.0.6 中，把这个父节点写进子图节点的泛型 `Literal` 会让子图编译器把它当作未知的本地节点。示例因此返回非参数化的 `Command`，并用端到端 self-test 验证父目标确实被执行。

这意味着父节点名的拼写错误会推迟到运行时暴露。生产代码应把 handoff 目标集中定义、为每条跨图路径写契约测试，并确保目标 update 满足父 state schema。

## 8. 动态 fan-out：为什么需要 Send

普通条件边通常从一组已知目标中选择下一节点：

```text
route -> analyst 或 engineer
```

但下面的问题无法只靠静态边表达：

> supervisor 读取本次请求后，可能选择 1、3 或 20 个 worker；每个 worker 还需要不同输入。

`Send(node_name, input_state)` 表示“为指定节点创建一次带独立输入的调用”。路由函数可以返回一个 `Send` 列表：

```python
return [
    Send("worker", {"request": request, "worker": role})
    for role in selected_workers
]
```

worker 节点在图定义中只有一个，运行时调用实例数却由 state 决定。这与提前创建 `analyst_node`、`engineer_node`、`writer_node` 三个固定节点不同。

## 9. 示例三：supervisor 动态派发 worker

文件：[examples/send_supervisor.py](./examples/send_supervisor.py)

运行：

```bash
conda run -n langgraph python langgraph/10-subgraphs-and-multi-agent/examples/send_supervisor.py
conda run -n langgraph python langgraph/10-subgraphs-and-multi-agent/examples/send_supervisor.py --request "Plan a small internal change"
conda run -n langgraph python langgraph/10-subgraphs-and-multi-agent/examples/send_supervisor.py --self-test
```

图的静态拓扑是：

```text
START -> supervisor_plan -> [Send worker N 次] -> supervisor_synthesize -> END
```

针对默认输入，本次运行的动态形状是：

```text
                              +-> worker(analyst) --+
supervisor_plan -> dispatch --+-> worker(engineer) -+-> supervisor_synthesize
                              +-> worker(writer) ---+
```

### 9.1 父图状态与 worker 输入分开

完整团队状态：

```python
class TeamState(TypedDict):
    request: str
    selected_workers: list[str]
    worker_results: Annotated[list[dict[str, str]], add]
    report: str
```

单个 worker 只需要：

```python
class WorkerInput(TypedDict):
    request: str
    worker: str
```

这样做的好处是最小权限和低耦合：worker 看不到其他 worker 的结果，也不能依赖尚未完成的汇总状态。真实系统中，不同 worker 还可以绑定不同工具和凭证。

### 9.2 supervisor 只规划，不偷偷执行 worker 工作

```python
def supervisor_plan(state):
    ...
    return {"selected_workers": selected}
```

示例按关键词选择角色；真实系统可以换成规则引擎或结构化输出模型。无论如何，规划节点的输出应是可审计的结构化计划，而不是一边规划一边直接调用所有工具。

这种分离让测试可以分别回答：

- supervisor 是否选对角色？
- dispatch 是否为每个角色创建一次调用？
- 每个 worker 是否只完成自己的职责？
- 汇总是否等待并包含所有结果？

### 9.3 条件边返回 Send 列表

```python
def dispatch_workers(state):
    return [
        Send(
            "worker",
            {"request": state["request"], "worker": worker},
        )
        for worker in state["selected_workers"]
    ]
```

注册方式：

```python
builder.add_conditional_edges(
    "supervisor_plan",
    dispatch_workers,
    ["worker"],
)
```

这里的 `["worker"]` 声明可能的目标，便于构图与校验；真正创建多少次调用由返回的 `Send` 数量决定。

每个 `Send` 的第二个参数是该次 worker 调用的输入，不必是完整 `TeamState`。这使 map 阶段能够只携带当前任务。

### 9.4 worker 返回一个可合并增量

```python
def worker(state: WorkerInput):
    ...
    return {
        "worker_results": [
            {"worker": role, "result": templates[role]}
        ]
    }
```

每个 worker 只返回一个单元素 list。它没有读取和 append 父 state 中的旧 list，因为并行节点不应通过共享可变对象协调。

## 10. 并发写入为什么必须有 reducer

三个 worker 在同一个 superstep 都写 `worker_results`。如果字段只是：

```python
worker_results: list[dict[str, str]]
```

运行时面对三个候选新值，不知道应该保留哪个。典型结果是并发更新冲突，而不是自动猜测你想拼接 list。

示例声明：

```python
worker_results: Annotated[list[dict[str, str]], add]
```

`Annotated[..., add]` 告诉 state channel 使用 `operator.add` 合并更新：

```text
[analyst result] + [engineer result] + [writer result]
```

reducer 是数据语义，不只是消除异常的语法。选择 reducer 时要问：

- 是 append、集合并集、按 ID 覆盖，还是数值求和？
- 重试或恢复时重复 update 是否可接受？
- reducer 是否会原地修改输入，造成难以追踪的共享状态？
- 合并是否满足结合律；不同分组方式会不会改变结果？
- 顺序是否属于业务语义？

对于生产 worker 结果，常见做法是携带稳定 `worker_id`/`task_id`，汇总时按 ID 校验缺失与重复。简单的 list `add` 适合教学，但它本身不去重。

## 11. superstep 屏障：为什么汇总会等所有 worker

LangGraph 的执行可理解为 bulk-synchronous parallel 模型：

```text
superstep 1: supervisor_plan
                  |
             产生 N 个 Send

superstep 2: worker[0], worker[1], ... worker[N-1]
                  |
          所有本步更新完成并合并

superstep 3: supervisor_synthesize
```

代码中每个动态 worker 都有一条逻辑边：

```python
builder.add_edge("worker", "supervisor_synthesize")
```

汇总节点在下一 superstep 执行，读取已经合并的 `worker_results`。这就是 fan-out/fan-in 的同步屏障。

它不意味着 worker 按某个顺序完成，也不意味着慢 worker 会被自动忽略。只要本次 fan-out 中有 worker 仍未完成、重试或失败，汇总就不能假装已经拥有完整结果。生产系统需要另外设计超时、失败策略、部分结果规则和可观测性。

## 12. 不要依赖并行结果顺序

即使某一次本地运行得到：

```text
analyst, engineer, writer
```

也不要把并行完成顺序当作 API 契约。真实 worker 的模型延迟、网络工具、重试和调度都会变化。

本例在汇总时显式排序：

```python
ordered = sorted(
    state["worker_results"],
    key=lambda item: item["worker"],
)
```

因此最终报告稳定，快照测试不会因为调度变化而抖动。更一般的策略包括：

- 按 supervisor 计划中的位置排序；
- 按稳定 `task_id` 排序；
- 使用 dict/map reducer，再按 key 读取；
- 如果顺序确实无关，测试集合而不是 list 位置。

示例的自测先把结果转换成 `by_worker` 字典检查角色集合，再检查最终报告经过排序。它没有断言原始 `worker_results[0]` 必须来自哪个并行 worker。

## 13. 从确定性示例迁移到真实多 Agent

将 `worker()` 替换成模型节点时，建议保留当前控制面：

```text
请求
  |
  v
supervisor 输出结构化任务计划
  |
  v
Send 每个最小任务输入
  |
  v
不同 worker 使用受限工具集执行
  |
  v
结构化结果 + task_id + evidence + status
  |
  v
synthesizer 校验完整性后生成报告
```

不要让 supervisor 只输出一段无法解析的自然语言，再靠字符串切割派发任务。一个更可靠的计划 schema 可以包含：

```python
{
    "task_id": "research-api-risk",
    "role": "analyst",
    "instruction": "Assess API reliability risks",
    "allowed_tools": ["metrics_reader"],
    "required_output": "risk_list",
}
```

同时要补上：

- 每个 worker 的超时和重试边界；
- 工具授权与参数校验；
- 幂等的外部副作用；
- 最大 fan-out 数量和预算；
- 结果 schema 校验；
- 缺失、失败和重复 task 的处理策略；
- stream/trace 中的 `task_id` 关联。

“多 Agent”增加了调度和协作成本。若任务不能真正拆成独立、可验证的子问题，一个能力更强但边界清晰的单工作流通常更简单。

## 14. 子图、handoff 与 Send 怎样组合

原生子图、handoff 与 `Send` 可以组合，而不是三选一。例如每个 worker 本身是一张包含工具循环和人工审批的子图：

```text
                         +-> analyst subgraph -----+
supervisor -> Send ------+-> engineer subgraph ----+-> synthesize
                         +-> compliance subgraph --+

worker subgraph:
START -> model -> tools -> model -> approval -> END

specialist handoff:
billing subgraph --Command.PARENT--> parent compliance_review
```

设计时分三步：

1. 用子图表达每类 worker 内部稳定的工作流边界。
2. 用 `Send` 表达本次运行需要创建哪些 worker 任务。
3. 用 `Command.PARENT` 表达 specialist 主动把控制权交给哪个父级协调节点。

状态也应分层：

| 层级 | 典型字段 |
|---|---|
| supervisor | request、plan、task IDs、最终报告 |
| 单 worker 输入 | task ID、角色、最小任务描述、权限范围 |
| worker 子图内部 | messages、工具结果、草稿、审批状态 |
| worker 输出 | task ID、结构化结论、证据、状态 |

不要让所有 worker 直接共享一份巨大消息历史。那会扩大上下文、混淆权限，并让 reducer 语义变得难以管理。

## 15. 常见坑

### 15.1 把 stream namespace 当作 direct/wrapper 的选型标准

在 1.0.6 的当前节点执行上下文中，wrapper 内的 compiled graph `invoke()` 也能产生嵌套 namespace。应根据是否需要显式 state 映射选择 wrapper，而不是根据一条错误的“wrapper 不可观察”规则选择。用 `--compare-wrapper` 验证实际事件，不凭印象判断。

### 15.2 父子字段名称或类型不一致

共享 key 是接口。父图的 `notes: list[str]` 和子图的 `notes: str` 即使同名也不是可靠契约；应统一类型或显式转换。

### 15.3 把 namespace 实例 ID 写死在测试中

运行时标识可能每次变化。测试层级为空/非空以及内部节点名，不测试具体实例字符串。

### 15.4 多个 Send 写同一普通字段

没有 reducer 时会产生并发更新冲突。先定义业务上的合并规则，再用 `Annotated` 声明。

### 15.5 reducer 只会追加，不会去重

节点重试、恢复设计或上游重复任务可能带来重复业务结果。需要稳定 task ID 和显式去重策略时，不要把 `operator.add` 当成幂等保证。

### 15.6 假设 worker 按 Send 列表顺序完成

并发完成顺序不应成为业务协议。输出要按稳定 key 排序或使用无序断言。

### 15.7 supervisor 同时承担所有执行工作

这会让规划不可审计，worker 边界名存实亡。supervisor 应输出任务计划，worker 接收最小输入并独立执行。

### 15.8 无限制动态 fan-out

一次模型计划出几千个任务会耗尽并发、token 和外部 API 配额。必须校验角色白名单、任务数量、递归深度和预算。

### 15.9 把 Send 当成后台任务队列

`Send` 是图运行中的动态调度语义，不自动提供跨系统队列的所有能力。超长外部任务仍需考虑持久化、取消、租约、超时和基础设施边界。

### 15.10 认为下游节点会收到部分并行结果并立即汇总

标准 fan-in 在下一 superstep 看到本步合并后的结果。若产品需要“哪个 worker 先完成就先展示哪个”，应使用 streaming 观察 worker update，而不是让最终 synthesizer 基于不完整状态提前运行。

### 15.11 handoff 省略 `Command.PARENT` 或漏传父图 update

省略 `graph=Command.PARENT` 会让 `goto` 在当前子图解析；只写 `goto` 不写必要 update，则父目的节点可能拿不到 specialist 产物。目标层级、目标节点和跨层状态增量是同一份 handoff 协议，必须一起测试。

## 16. 练习

1. 给 `native_subgraph.py` 增加 `validate_sources` 内部节点，并在自测中通过非空 namespace 验证它被执行。
2. 把父图字段 `notes` 改名为 `research_notes`，设计一个明确的状态转换方案，并比较它与共享 key 方案的复杂度。
3. 修改 wrapper，只映射一个 child output；比较 direct 与 wrapper 的父层 update，同时确认两者仍有嵌套 namespace。
4. 再嵌套一层子图，打印 namespace tuple，观察层级路径怎样增长。
5. 给 handoff 增加 `compliance_review` 父节点，并按金额选择两个不同父目标。
6. 故意省略 `Command.PARENT` 或拼错父目标，记录错误发生在编译期还是运行期。
7. 给 `send_supervisor.py` 增加 `reviewer` 角色，并只在请求包含 `security` 时派发。
8. 故意移除 `worker_results` 的 reducer，运行三个 worker，记录并解释并发更新错误。
9. 把 list reducer 改成按 `task_id` 合并的自定义 reducer，并处理重复 ID。
10. 给某个 worker 增加不同延迟，证明最终报告顺序仍由显式排序控制。
11. 让 worker 返回 `status`，设计“全部成功才汇总”和“允许部分成功”两种策略。
12. 把 worker 改造成原生子图，并让其中一条路径 handoff 到父图 review 节点。

## 17. 自检

- [ ] 我能解释子图带来的模块边界，而不只说“少写代码”。
- [ ] 我知道 direct 与 wrapper 在 1.0.6 中都能产生嵌套 namespace，真正差异是状态映射责任。
- [ ] 我能用共享 state key 设计父子图接口。
- [ ] 我知道 `subgraphs=True` 时事件为什么带 namespace tuple。
- [ ] 我不会在测试中硬编码 namespace 的运行实例 ID。
- [ ] 我能解释 `Command.PARENT` 的目标在哪一层解析。
- [ ] 我知道 handoff update 由父图 channel 合并，并只返回新增 messages。
- [ ] 我能从 stream 证明目的父节点确实执行，而不只看最终文本。
- [ ] 我能解释 `Send` 为什么适合运行时动态 fan-out。
- [ ] 我知道每个 `Send` 可以携带不同的最小输入。
- [ ] 我能为并发写入选择业务上正确的 reducer。
- [ ] 我知道 list `add` 不提供去重或幂等保证。
- [ ] 我能解释 superstep 屏障为什么让 synthesizer 等待全部 worker。
- [ ] 我不会依赖并行 worker 的完成顺序。
- [ ] 我能区分 supervisor 规划、worker 执行和 synthesizer 汇总。
- [ ] 我知道生产多 Agent 还需要预算、权限、超时和失败策略。

## 18. 本章结论

子图、handoff 和 `Send` 解决三个不同维度的问题：子图封装固定内部流程，`Command.PARENT` 把控制权和状态增量交回指定父节点，`Send` 根据本次 state 动态展开并行任务。三者都依赖明确的数据协议；共享 key 定义父子接口，handoff update 定义跨层交付，reducer 定义并发更新怎样合并，superstep 定义何时可以进入下一阶段。

可靠的多 Agent 系统不是让更多模型自由交谈，而是让规划、执行、汇总、权限和失败边界都可见、可测试。先用本章的确定性代码掌握调度语义，再替换成真实模型，问题会容易定位得多。
