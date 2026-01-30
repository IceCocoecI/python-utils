"""
LangGraph QuickStart 教程 - 构建一个简单的数学运算Agent

这个示例展示了如何使用 LangGraph 构建一个能够执行数学运算的智能体。
智能体可以识别用户的数学问题，调用相应的工具进行计算，并返回结果。

LangGraph 是一个用于构建有状态、多步工作流的框架，特别适合构建复杂的AI应用。
"""

# Step 1: 定义工具和模型
from langchain.tools import tool
from langchain.chat_models import init_chat_model

# 初始化聊天模型
# 这里使用 Claude Sonnet 模型，你可以根据需要更换为其他模型
model = init_chat_model(
    "claude-sonnet-4-5-20250929",  # 模型名称
    temperature=0  # 温度参数，0表示完全确定性输出
)

# 定义工具函数
# LangGraph 通过工具来扩展LLM的能力，让LLM可以执行具体的操作

@tool
def multiply(a: int, b: int) -> int:
    """乘法工具：计算 a 和 b 的乘积。

    Args:
        a: 第一个整数
        b: 第二个整数
    
    Returns:
        a 和 b 的乘积
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """加法工具：计算 a 和 b 的和。

    Args:
        a: 第一个整数
        b: 第二个整数
    
    Returns:
        a 和 b 的和
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """除法工具：计算 a 除以 b 的结果。

    Args:
        a: 被除数
        b: 除数
    
    Returns:
        a 除以 b 的结果（浮点数）
    """
    if b == 0:
        return "错误：除数不能为零"
    return a / b

# 将所有工具放入列表
tools = [add, multiply, divide]

# 创建工具名称到工具对象的映射，方便后续查找
tools_by_name = {tool.name: tool for tool in tools}

# 将工具绑定到模型，让模型知道它可以调用这些工具
model_with_tools = model.bind_tools(tools)

# Step 2: 定义状态
# LangGraph 是有状态的，需要定义工作流的状态结构
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class MessagesState(TypedDict):
    """定义工作流的状态结构
    
    Attributes:
        messages: 消息列表，保存对话历史
        llm_calls: LLM调用次数统计
    """
    messages: Annotated[list[AnyMessage], operator.add]  # 使用operator.add来合并消息列表
    llm_calls: int  # 跟踪LLM调用次数


# Step 3: 定义LLM节点
# 这个节点负责调用LLM并让其决定是否需要调用工具
from langchain.messages import SystemMessage


def llm_call(state: MessagesState):
    """LLM节点：调用语言模型并让其决定下一步操作
    
    这个节点会：
    1. 将系统消息和历史消息发送给LLM
    2. LLM决定是直接回复用户还是调用工具
    3. 返回LLM的响应
    
    Args:
        state: 当前工作流状态
    
    Returns:
        包含LLM响应和更新调用次数的状态
    """
    # 构建消息列表：系统消息 + 历史对话消息
    messages = [
        SystemMessage(
            content="你是一个乐于助人的助手，负责对一组输入执行算术运算。"
        )
    ] + state["messages"]
    
    # 调用绑定了工具的LLM
    response = model_with_tools.invoke(messages)
    
    return {
        "messages": [response],  # 将LLM响应添加到消息列表
        "llm_calls": state.get('llm_calls', 0) + 1  # 增加LLM调用计数
    }


# Step 4: 定义工具节点
from langchain.messages import ToolMessage


def tool_node(state: MessagesState):
    """工具节点：执行工具调用
    
    当LLM决定调用工具时，这个节点会：
    1. 解析LLM的工具调用请求
    2. 执行相应的工具函数
    3. 将工具执行结果封装为ToolMessage
    
    Args:
        state: 当前工作流状态
    
    Returns:
        包含工具执行结果的状态
    """
    result = []
    # 获取最后一个消息（应该是LLM的响应）
    last_message = state["messages"][-1]
    
    # 遍历LLM请求的所有工具调用
    for tool_call in last_message.tool_calls:
        # 根据工具名称查找对应的工具函数
        tool = tools_by_name[tool_call["name"]]
        # 使用工具参数调用工具
        observation = tool.invoke(tool_call["args"])
        # 将工具执行结果封装为ToolMessage
        result.append(
            ToolMessage(
                content=str(observation),  # 工具执行结果
                tool_call_id=tool_call["id"]  # 工具调用ID，用于匹配请求和响应
            )
        )
    
    return {"messages": result}


# Step 5: 定义路由逻辑
# 决定工作流的下一个步骤：继续调用工具还是结束
from typing import Literal


def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """路由函数：决定工作流的下一个步骤
    
    这个函数检查LLM的最后一个响应：
    1. 如果LLM调用了工具，则路由到工具节点
    2. 如果LLM没有调用工具（直接回复了用户），则结束工作流
    
    Args:
        state: 当前工作流状态
    
    Returns:
        下一个节点的名称或结束标记
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # 检查LLM是否调用了工具
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_node"  # 有工具调用，转到工具节点
    else:
        return "__end__"  # 没有工具调用，结束工作流


# Step 6: 构建Agent工作流
from langgraph.graph import StateGraph, START, END


# 创建状态图构建器
agent_builder = StateGraph(MessagesState)

# 添加节点到图中
agent_builder.add_node("llm_call", llm_call)  # LLM调用节点
agent_builder.add_node("tool_node", tool_node)  # 工具执行节点

# 添加边（连接节点）
agent_builder.add_edge(START, "llm_call")  # 从开始到LLM节点

# 添加条件边：根据路由函数决定下一步
agent_builder.add_conditional_edges(
    "llm_call",  # 源节点
    should_continue,  # 路由决策函数
    {  # 路由目标映射
        "tool_node": "tool_node",  # 如果返回"tool_node"，则转到工具节点
        "__end__": END  # 如果返回"__end__"，则结束
    }
)

# 工具执行后，总是返回到LLM节点进行下一步处理
agent_builder.add_edge("tool_node", "llm_call")

# 编译工作流，生成可执行的Agent
agent = agent_builder.compile()

# Step 7: 可视化工作流（可选）
# 注意：这段代码需要graphviz和IPython环境才能运行
try:
    from IPython.display import Image, display
    # 生成并显示工作流图
    display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
    print("工作流图已显示（需要IPython环境）")
except ImportError:
    print("无法导入IPython，跳过图形显示")
except Exception as e:
    print(f"图形显示失败：{e}")


# Step 8: 使用Agent
def run_example():
    """运行示例：测试Agent的功能"""
    print("=" * 60)
    print("LangGraph QuickStart 示例")
    print("=" * 60)
    
    # 示例1: 简单的加法
    print("\n示例1: 计算 3 + 4")
    messages = [{"role": "user", "content": "Add 3 and 4."}]
    result = agent.invoke({"messages": messages, "llm_calls": 0})
    
    print("对话历史:")
    for i, m in enumerate(result["messages"]):
        print(f"  [{i}] {m.__class__.__name__}: {m.content}")
    
    print(f"LLM调用次数: {result['llm_calls']}")
    
    # 示例2: 复杂的计算
    print("\n示例2: 计算 (5 * 3) + 2")
    messages = [{"role": "user", "content": "Multiply 5 and 3, then add 2 to the result."}]
    result = agent.invoke({"messages": messages, "llm_calls": 0})
    
    print("对话历史:")
    for i, m in enumerate(result["messages"]):
        print(f"  [{i}] {m.__class__.__name__}: {m.content}")
    
    print(f"LLM调用次数: {result['llm_calls']}")
    
    # 示例3: 除法
    print("\n示例3: 计算 10 ÷ 2")
    messages = [{"role": "user", "content": "Divide 10 by 2."}]
    result = agent.invoke({"messages": messages, "llm_calls": 0})
    
    print("对话历史:")
    for i, m in enumerate(result["messages"]):
        print(f"  [{i}] {m.__class__.__name__}: {m.content}")
    
    print(f"LLM调用次数: {result['llm_calls']}")
    
    print("\n" + "=" * 60)
    print("示例运行完成！")


if __name__ == "__main__":
    # 运行示例
    run_example()
    
    # 你也可以交互式地使用Agent
    print("\n提示：你可以通过以下方式交互式使用Agent:")
    print("  1. 直接调用 agent.invoke()")
    print("  2. 查看工作流图: agent.get_graph().draw_mermaid_png()")
    print("  3. 跟踪状态变化: 检查返回的 messages 和 llm_calls")
    
    print("\n工作流节点说明:")
    print("  - llm_call: 调用语言模型，决定是否使用工具")
    print("  - tool_node: 执行工具调用")
    print("  - 条件路由: 根据LLM输出决定下一步")
    print("  - 循环: 工具执行后返回llm_call继续处理")


# 重要概念总结:
"""
1. 状态 (State): LangGraph工作流是有状态的，我们定义了MessagesState来跟踪消息和调用次数
2. 节点 (Nodes): 工作流的基本构建块，如llm_call和tool_node
3. 边 (Edges): 连接节点的路径，决定工作流的流向
4. 条件边 (Conditional Edges): 根据条件动态决定下一步
5. 工具绑定 (Tool Binding): 让LLM知道它可以调用哪些工具
6. 消息传递: 使用ToolMessage来传递工具调用结果

工作流执行流程:
  开始 → llm_call → (检查工具调用) → 如果有工具调用: tool_node → llm_call
                              → 如果没有工具调用: 结束
"""