# LangGraph 核心概念与最佳实践

> 本教程基于 LangGraph 官方文档，帮助你深入理解这个强大的 AI Agent 编排框架。

## 目录

1. [什么是 LangGraph](#什么是-langgraph)
2. [核心概念](#核心概念)
3. [设计理念](#设计理念)
4. [最佳实践](#最佳实践)
5. [代码示例](#代码示例)

---

## 什么是 LangGraph

LangGraph 是由 LangChain 团队开发的**低层级编排框架**，专门用于构建、管理和部署**长时间运行的、有状态的 AI Agent**。

### 核心定位

```
┌─────────────────────────────────────────────────────────────────┐
│                        应用层                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Chatbot    │  │  RAG Agent  │  │  多Agent    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│                     LangGraph (编排层)                           │
│  • 状态管理 (State)                                              │
│  • 流程控制 (Graph: Nodes + Edges)                               │
│  • 持久化 (Checkpointing)                                        │
│  • 人机交互 (Human-in-the-Loop)                                  │
├─────────────────────────────────────────────────────────────────┤
│                     LangChain (组件层)                           │
│  • Models (LLM 调用)                                             │
│  • Tools (工具集成)                                              │
│  • Retrievers (检索)                                             │
└─────────────────────────────────────────────────────────────────┘
```

### 为什么需要 LangGraph?

| 传统方法 | LangGraph 方法 |
|---------|---------------|
| 简单的 prompt chain | 复杂的状态图 |
| 无状态请求 | 有状态的长期运行 |
| 线性执行流程 | 条件分支、循环 |
| 无法中断 | 人机交互支持 |
| 失败重试困难 | 持久化检查点 |

---

## 核心概念

### 1. 图 (Graph) - 执行的骨架

LangGraph 将 AI 应用建模为**有向图**：

```
         START
           │
           ▼
      ┌─────────┐
      │ Node A  │  ← 节点：执行单元
      └────┬────┘
           │       ← 边：连接节点
      ┌────▼────┐
      │ Node B  │
      └────┬────┘
           │
      ┌────▼────┐
      │ Router  │  ← 条件节点
      └────┬────┘
          / \
         /   \
    ┌───▼─┐ ┌─▼───┐
    │ C1  │ │ C2  │  ← 分支
    └───┬─┘ └─┬───┘
         \   /
          \ /
       ┌───▼───┐
       │  END  │
       └───────┘
```

**核心组件：**

```python
from langgraph.graph import StateGraph, START, END

# 创建图
graph = StateGraph(State)

# 添加节点 (处理逻辑)
graph.add_node("node_name", node_function)

# 添加边 (连接节点)
graph.add_edge("node_a", "node_b")

# 添加条件边 (路由逻辑)
graph.add_conditional_edges("router", routing_function)

# 编译图
app = graph.compile()
```

### 2. 状态 (State) - 数据的载体

State 是整个图执行过程中的**共享数据容器**：

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    # 消息列表 - 使用 add_messages 累积
    messages: Annotated[list, add_messages]
    
    # 简单字段 - 直接覆盖
    current_step: str
    
    # 累积计数
    retry_count: int
```

**状态更新原则：**

```python
def node_function(state: AgentState) -> dict:
    # 返回部分状态更新，不是完整状态
    return {
        "messages": [new_message],  # 追加到列表
        "current_step": "step_2"    # 覆盖值
    }
```

### 3. 节点 (Node) - 执行的单元

节点是图中的**处理单元**，接收状态、执行逻辑、返回状态更新：

```python
def my_node(state: AgentState) -> dict:
    """
    节点函数签名：
    - 输入: 当前完整状态
    - 输出: 状态更新字典 (部分更新)
    """
    # 1. 读取状态
    messages = state["messages"]
    
    # 2. 执行逻辑 (可以是任何代码)
    result = some_processing(messages)
    
    # 3. 返回状态更新
    return {"messages": [result]}
```

### 4. 边 (Edge) - 流程的控制

```python
# 普通边：无条件转移
graph.add_edge("node_a", "node_b")

# 条件边：根据状态决定下一步
def router(state: AgentState) -> str:
    if state["needs_tool"]:
        return "tool_node"
    return "response_node"

graph.add_conditional_edges(
    "decision_node",
    router,
    {
        "tool_node": "tool_node",
        "response_node": "response_node"
    }
)
```

### 5. 检查点 (Checkpointer) - 持久化

检查点允许图在任意位置**暂停和恢复**：

```python
from langgraph.checkpoint.memory import MemorySaver

# 内存检查点 (开发用)
checkpointer = MemorySaver()

# 编译时添加检查点
app = graph.compile(checkpointer=checkpointer)

# 使用 thread_id 标识会话
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke(input_state, config)
```

---

## 设计理念

### 1. 低抽象 (Low Abstraction)

LangGraph **不抽象** prompts 或架构，你完全控制：

- 如何构造 prompt
- 使用什么模型
- 如何处理工具调用
- 状态如何流转

```python
# 你决定所有细节
def chat_node(state):
    # 你控制 prompt
    prompt = f"User: {state['input']}\nAssistant:"
    
    # 你控制模型调用
    response = my_model.generate(prompt)
    
    # 你控制响应处理
    return {"output": response}
```

### 2. 状态即一切 (State is Everything)

所有信息都通过状态传递，这带来：

- **可追踪**: 任意时刻可以检查状态
- **可恢复**: 从任意检查点恢复
- **可测试**: 输入状态 → 输出状态

### 3. 图即工作流 (Graph as Workflow)

将复杂流程建模为图的好处：

```python
# 声明式定义流程
graph.add_edge(START, "parse_input")
graph.add_edge("parse_input", "validate")
graph.add_conditional_edges("validate", route_by_validity)
graph.add_edge("process", "format_output")
graph.add_edge("format_output", END)

# 而不是命令式代码
# if parse_input():
#     if validate():
#         process()
#         format_output()
```

### 4. 持久优先 (Durable by Default)

LangGraph 设计用于**长时间运行**的任务：

- 检查点自动保存每一步状态
- 失败后从上次状态恢复
- 支持跨会话的持久记忆

---

## 最佳实践

### 1. 状态设计

```python
# ✅ 好的做法：保持状态最小化和类型化
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 必要的消息历史
    current_task: str                         # 当前任务
    context: dict                             # 必要的上下文

# ❌ 避免：在状态中存储临时值
class BadState(TypedDict):
    messages: list
    temp_calculation: float  # 应该在函数内处理
    debug_info: str          # 不应该在状态中
```

### 2. 节点设计

```python
# ✅ 好的做法：节点像纯函数
def process_node(state: AgentState) -> dict:
    # 输入只依赖 state
    # 不修改 state，只返回更新
    result = process(state["input"])
    return {"output": result}

# ❌ 避免：节点有副作用或修改 state
def bad_node(state: AgentState) -> dict:
    state["count"] += 1  # 不要修改输入
    global_var.update()  # 避免全局状态
    return state         # 返回更新，不是完整状态
```

### 3. 错误处理

```python
# ✅ 好的做法：在节点内处理错误
def robust_node(state: AgentState) -> dict:
    try:
        result = risky_operation(state["input"])
        return {"output": result, "error": None}
    except Exception as e:
        return {"output": None, "error": str(e)}

# 然后在路由中处理错误
def error_router(state: AgentState) -> str:
    if state.get("error"):
        return "error_handler"
    return "next_step"
```

### 4. 循环控制

```python
# ✅ 好的做法：添加循环保护
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration_count: int
    max_iterations: int

def should_continue(state: AgentState) -> str:
    # 硬性停止条件
    if state["iteration_count"] >= state["max_iterations"]:
        return "force_end"
    
    # 业务逻辑判断
    if task_completed(state):
        return "end"
    
    return "continue"
```

### 5. 边界验证

```python
# ✅ 好的做法：在节点边界验证
def validated_node(state: AgentState) -> dict:
    # 入口验证
    if not state.get("required_field"):
        raise ValueError("required_field is missing")
    
    # 处理逻辑
    result = process(state["required_field"])
    
    # 出口验证
    if not is_valid_result(result):
        return {"error": "Invalid result", "should_retry": True}
    
    return {"output": result}
```

---

## 代码示例

本目录包含以下学习案例，建议按顺序学习：

### 基础概念

| 文件 | 内容 | 核心知识点 |
|------|------|-----------|
| `01_basic_graph.py` | 基础图构建 | StateGraph, Node, Edge, START, END |
| `02_chatbot_with_tools.py` | 工具调用 | 工具定义, 工具执行, 循环模式 |

### 进阶特性

| 文件 | 内容 | 核心知识点 |
|------|------|-----------|
| `03_human_in_the_loop.py` | 人机交互 | interrupt, 中断点, 状态修改 |
| `04_persistence_memory.py` | 持久化与记忆 | Checkpointer, thread_id, 恢复执行 |

### 高级模式

| 文件 | 内容 | 核心知识点 |
|------|------|-----------|
| `05_conditional_routing.py` | 条件路由 | 条件边, 路由函数, 分支合并 |
| `06_multi_agent.py` | 多Agent协作 | Supervisor, 子图, Agent通信 |

---

## 运行环境

```bash
# 安装依赖
pip install -r requirements.txt

# 设置环境变量 (复制 .env.example 并填写)
cp ../.env.example .env

# 运行示例
python 01_basic_graph.py
```

---

## 参考资源

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangGraph 教程](https://langchain-ai.github.io/langgraph/tutorials/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [LangChain Academy](https://academy.langchain.com/courses/intro-to-langgraph)

---

## 概念速查表

```
┌────────────────────────────────────────────────────────────────┐
│                    LangGraph 核心概念                           │
├────────────────────────────────────────────────────────────────┤
│ Graph      │ 图        │ 定义执行流程的容器                      │
│ State      │ 状态      │ 节点间共享的数据结构                    │
│ Node       │ 节点      │ 执行具体逻辑的函数                      │
│ Edge       │ 边        │ 连接节点，控制流转                      │
│ Reducer    │ 规约器    │ 定义状态如何更新 (覆盖/追加)            │
│ Checkpoint │ 检查点    │ 持久化状态，支持恢复                    │
│ Thread     │ 线程      │ 一个独立的执行上下文                    │
│ Interrupt  │ 中断      │ 暂停执行等待人工输入                    │
├────────────────────────────────────────────────────────────────┤
│                    常用 API                                     │
├────────────────────────────────────────────────────────────────┤
│ StateGraph(State)              │ 创建状态图                     │
│ graph.add_node(name, func)     │ 添加节点                       │
│ graph.add_edge(from, to)       │ 添加普通边                     │
│ graph.add_conditional_edges()  │ 添加条件边                     │
│ graph.compile()                │ 编译图                         │
│ app.invoke(input, config)      │ 同步执行                       │
│ app.stream(input, config)      │ 流式执行                       │
└────────────────────────────────────────────────────────────────┘
```
