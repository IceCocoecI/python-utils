# 03 · 控制流：条件路由、循环与 Command

> 目标环境：Python 3.11、LangGraph 1.0.6。  
> 本章完全离线，不需要模型、API Key 或网络。

## 学完本章能做什么

你将能够：

1. 用 `add_conditional_edges()` 根据状态选择唯一分支。
2. 把“分类节点”和“路由函数”的职责分开。
3. 让多个分支汇聚到同一个后处理节点。
4. 用条件边构造可终止循环。
5. 用 `Command(update=..., goto=...)` 在一个节点中同时更新状态和选择下一跳。
6. 比较条件边和 `Command` 的适用场景。
7. 使用计数器、停止条件和 `recursion_limit` 防止失控循环。

## 1. 从固定流程到运行时决策

前两章的边是固定的：

```text
START -> A -> B -> C -> END
```

真实工作流经常要根据状态选择路径：

```text
                    ┌-> math handler ─┐
START -> classify --+-> text handler ─+-> finalize -> END
                    └-> fallback ─────┘
```

或者在结果不满足条件时回到之前的节点：

```text
START -> generate -> evaluate --accepted--> END
                       │
                       └----retry---------> generate
```

控制流的核心不是“让图看起来复杂”，而是把运行时决策显式化：

- 决策依赖哪些 state 字段？
- 可能返回哪些路径？
- 每条路径最终是否可结束？
- 循环最多执行多少次？

## 2. 条件边的心智模型

固定边表达无条件下一步：

```python
builder.add_edge("a", "b")
```

条件边表达“先运行路由函数，再把结果映射到节点”：

```python
builder.add_conditional_edges(
    "classify",
    choose_branch,
    {
        "math": "sum",
        "text": "uppercase",
        "fallback": "fallback",
    },
)
```

执行过程：

```text
classify 节点完成
      │
      ▼
choose_branch(current_state)
      │
      ├─ "math"     -> sum 节点
      ├─ "text"     -> uppercase 节点
      └─ "fallback" -> fallback 节点
```

路由函数读取已经合并过节点更新的当前状态。它通常只做决策，不返回 state update。

## 3. 示例一：确定性条件路由

文件：[examples/conditional_routing.py](./examples/conditional_routing.py)

运行：

```bash
conda run -n langgraph python langgraph/03-control-flow/examples/conditional_routing.py "sum: 3 5 8"
conda run -n langgraph python langgraph/03-control-flow/examples/conditional_routing.py "upper: state is data"
conda run -n langgraph python langgraph/03-control-flow/examples/conditional_routing.py --self-test
```

为了只学习图机制，示例用字符串前缀模拟分类，不调用模型：

| 输入 | 分类 | 目标节点 | 输出 |
|---|---|---|---|
| `sum: 3 5 8` | `math` | `sum` | `sum=16` |
| `upper: state is data` | `text` | `uppercase` | `STATE IS DATA` |
| `ping` | `fallback` | `fallback` | 不支持提示 |

### 3.1 分类节点负责写状态

```python
def classify(state: RouteState) -> dict[str, object]:
    ...
    return {"category": category, "path": ["classify"]}
```

分类节点完成业务计算，将结果写入 `category`。在真实应用中，这里可以是规则、结构化模型输出或分类器，但它们都应产出一个后续可可靠判断的值。

### 3.2 路由函数负责选择路径

```python
def choose_branch(state: RouteState) -> Literal["math", "text", "fallback"]:
    ...
```

这里使用 `Literal` 明确所有可能返回值。好处包括：

- IDE 和类型检查器能发现拼写错误。
- 读者能直接看到分支全集。
- LangGraph 更容易推断并显示可能的图结构。

路由键与节点名不必相同，因为 `path_map` 负责映射：

```text
路由结果 "math" -> 图节点 "sum"
```

这允许业务决策词汇和实现节点名分别演进。

### 3.3 分支汇聚

```python
builder.add_edge("sum", "finalize")
builder.add_edge("uppercase", "finalize")
builder.add_edge("fallback", "finalize")
builder.add_edge("finalize", END)
```

三个分支最终都进入 `finalize`。适合放在汇聚节点中的逻辑包括：

- 统一输出格式。
- 审计记录。
- 最终安全检查。
- 指标与耗时记录。

不要在每个分支复制相同尾处理代码。

### 3.4 用 reducer 观察真实路径

示例状态包含：

```python
path: Annotated[list[str], add]
```

每个执行节点只追加自己的名字，因此测试可以断言：

```python
["classify", "sum", "finalize"]
```

这比仅断言最终答案更有价值：答案相同不代表图走了预期路径。

## 4. 循环需要三个不变量

一个可靠循环至少要明确：

1. **进度变量**：每轮什么会变化，例如 `retry_count` 或 `value`。
2. **停止条件**：什么状态会退出，例如质量达标或达到上限。
3. **硬上限**：停止条件失效时，系统如何避免无限执行。

示例使用：

```text
进度变量：value 每轮 +1
停止条件：value >= target
硬上限：recursion_limit
```

业务循环还应常见地加入 `max_retries`、deadline 或预算字段。不要只依赖框架全局限制来表达业务规则。

## 5. 示例二：两种循环写法

文件：[examples/loop_and_command.py](./examples/loop_and_command.py)

运行：

```bash
conda run -n langgraph python langgraph/03-control-flow/examples/loop_and_command.py --target 4
conda run -n langgraph python langgraph/03-control-flow/examples/loop_and_command.py --kind conditional --target 4
conda run -n langgraph python langgraph/03-control-flow/examples/loop_and_command.py --kind command --target 4
conda run -n langgraph python langgraph/03-control-flow/examples/loop_and_command.py --self-test
```

两张图完成同一个任务，最终都得到：

```text
value = 4
trace = [0, 1, 2, 3, 4]
status = "done at 4"
```

### 5.1 写法一：节点更新，条件边决策

节点只做递增：

```python
def increment(state: CounterState) -> dict[str, object]:
    next_value = state["value"] + 1
    return {"value": next_value, "trace": [next_value]}
```

路由函数单独判断：

```python
def route_after_increment(state: CounterState) -> Literal["again", "done"]:
    if state["value"] < state["target"]:
        return "again"
    return "done"
```

条件边把 `again` 映射回 `increment`，形成环：

```text
initialize -> increment --again--┐
                  ▲              │
                  └──────────────┘
                  │
                 done -> finish -> END
```

这种写法的优点是计算与路由职责分明。多个节点都可能复用同一个路由判断时，也更容易测试。

### 5.2 写法二：Command 同时更新和跳转

有时下一跳正好由本节点算出的新值决定。`Command` 可以原子地表达两件事：

```python
def command_increment(
    state: CounterState,
) -> Command[Literal["command_increment", "finish"]]:
    next_value = state["value"] + 1
    destination = (
        "command_increment" if next_value < state["target"] else "finish"
    )
    return Command(
        update={"value": next_value, "trace": [next_value]},
        goto=destination,
    )
```

`Command` 的两个关键参数：

| 参数 | 含义 |
|---|---|
| `update` | 本次节点产生的 state 更新，仍遵守 reducer 规则 |
| `goto` | 更新合并后要前往的节点 |

返回类型中的 `Literal` 列出动态目标：

```python
Command[Literal["command_increment", "finish"]]
```

它不仅帮助类型检查，也让图结构工具知道该节点可能去哪里。

### 5.3 条件边与 Command 如何选择

| 场景 | 更自然的选择 |
|---|---|
| 节点完成计算，独立规则读取状态决定下一步 | 条件边 |
| 同一决策同时决定更新内容与下一跳 | `Command` |
| 希望路由逻辑可独立复用和测试 | 条件边 |
| 希望节点封装一次局部状态机转换 | `Command` |
| 简单固定下一步 | 普通 `add_edge` |

`Command` 不是“更高级所以总该使用”的 API。控制流越显式、越容易从图上看懂越好。

## 6. recursion_limit 是护栏，不是业务条件

示例调用时设置：

```python
config = {"recursion_limit": max(25, target + 5)}
graph.invoke(input_state, config=config)
```

LangGraph 会限制单次图运行允许的步数，从而阻止意外无限循环。但业务上仍应在 state 中设计可解释的退出条件，例如：

```text
retry_count >= max_retries
token_budget <= 0
deadline exceeded
quality_score >= threshold
human chose cancel
```

如果只是一味调高 `recursion_limit`，通常是在隐藏终止条件设计问题。

## 7. 常见坑

### 7.1 路由函数返回了映射中不存在的键

如果 `path_map` 只有 `math` 和 `text`，路由函数却返回 `unknown`，运行时无法找到目标。使用 `Literal` 并为未知情况提供显式 fallback。

### 7.2 路由函数顺便修改 state

路由函数应尽量纯粹：读取状态并返回路径。状态变化放在节点 update 或 `Command.update` 中，否则执行语义难以追踪。

### 7.3 分类结果使用自由文本

模型可能输出 `Math`、`math.` 或一段解释，导致精确路由失败。真实模型路由应使用结构化输出、枚举或严格归一化。

### 7.4 循环没有单调进度

如果每轮没有计数增长、预算减少或质量变化，就无法证明会结束。为循环写测试时，不只测成功路径，还要测上限路径。

### 7.5 先判断再递增导致 off-by-one

明确条件检查的是“本轮前”还是“本轮后”的值。示例的两个实现都先计算 `next_value`，再判断是否继续，因而 trace 精确包含 `0..target`。

### 7.6 Command 节点又配置了无条件静态边

`Command.goto` 用于动态跳转。如果同一节点还存在普通静态边，静态边不会自动被“替代”，可能出现额外执行路径。除非确实需要并行语义，否则不要混用。

### 7.7 动态 goto 使用未经约束的外部字符串

目标节点名应来自代码控制的有限集合，而不是直接把用户输入作为 `goto`。用 `Literal`、枚举和显式映射限制目标。

## 8. 练习

1. 给条件路由增加 `lower:` 分支，并让它也汇聚到 `finalize`。
2. 给 `sum:` 的空参数和非法整数增加显式错误分支，而不是抛出未处理异常。
3. 将计数循环改成每次加 2，并定义 target 不是偶数时的停止语义。
4. 给循环 state 增加 `max_steps`，让业务上限小于框架 `recursion_limit`。
5. 新增 `cancelled` 终态，让某个输入直接跳过循环。
6. 使用 `stream_mode="updates"` 收集每轮事件，断言 `increment` 或 `command_increment` 恰好执行 `target` 次。
7. 故意让停止条件永远返回 `again`，设置较小 `recursion_limit`，观察错误并解释护栏作用。

## 9. 自检

- [ ] 我能说明固定边与条件边的区别。
- [ ] 我知道路由函数看到的是源节点更新合并后的状态。
- [ ] 我能解释 `path_map` 的键和值分别是什么。
- [ ] 我会用 `Literal` 列出路由结果或 `Command` 目标。
- [ ] 我能为循环指出进度变量、停止条件和硬上限。
- [ ] 我能比较条件边循环与 `Command` 循环的取舍。
- [ ] 我知道 `Command.update` 仍然遵守 state reducer。
- [ ] 我知道 `recursion_limit` 是框架护栏，不应替代业务重试上限。
- [ ] 我不会把未经约束的用户输入直接作为动态节点名。

## 10. 三章串起来看

到这里，最小 LangGraph 心智模型已经完整：

```text
State 定义数据协议
  │
  ├─ reducer 定义字段如何合并
  │
Node 定义局部计算和增量更新
  │
  ├─ Edge 定义固定执行关系
  ├─ Conditional Edge 定义状态驱动分支
  └─ Command 同时表达更新与动态跳转
```

后续的记忆、持久化、人工审批、工具调用和多智能体，都建立在这套基础语义之上。
