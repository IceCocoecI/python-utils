"""
LangGraph 基础教程 02: 带工具调用的聊天机器人
================================================

本示例演示如何构建一个能使用工具的 AI Agent：
1. 定义工具（Tools）
2. 集成 LLM
3. 实现工具调用循环
4. ReAct 模式

学习目标：
- 理解工具调用的工作流程
- 掌握 Agent 循环模式
- 学会处理 LLM 响应和工具执行

工作流程：
    START → agent → should_continue? → tools → agent → ... → END
                         ↓
                        END
"""

import re
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 注意：实际使用需要安装 langchain-openai 并配置 API Key
# 这里提供模拟版本用于学习

# =============================================================================
# 第一部分：状态定义
# =============================================================================

class AgentState(TypedDict):
    """
    聊天机器人的状态
    
    使用 add_messages 作为 reducer，消息会自动累积
    """
    # 消息历史：使用 add_messages reducer 自动追加
    messages: Annotated[list, add_messages]


# =============================================================================
# 第二部分：工具定义
# =============================================================================

def get_weather(city: str) -> str:
    """
    获取城市天气的工具
    
    这是一个模拟工具，实际应用中会调用真实 API
    """
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，温度 25°C，空气质量良好",
        "上海": "多云，温度 28°C，有轻微雾霾",
        "广州": "雷阵雨，温度 32°C，湿度较高",
        "深圳": "晴转多云，温度 30°C",
    }
    return weather_data.get(city, f"抱歉，暂无 {city} 的天气数据")


def search_info(query: str) -> str:
    """
    搜索信息的工具
    
    模拟搜索引擎
    """
    # 模拟搜索结果
    search_results = {
        "langgraph": "LangGraph 是一个用于构建有状态 AI Agent 的框架，支持复杂的工作流编排",
        "python": "Python 是一种广泛使用的高级编程语言，以简洁易读著称",
        "ai agent": "AI Agent 是能够自主执行任务、做出决策的智能系统",
    }
    
    # 简单关键词匹配
    for keyword, result in search_results.items():
        if keyword in query.lower():
            return result
    
    return f"未找到关于 '{query}' 的相关信息"


def calculate(expression: str) -> str:
    """
    计算器工具
    
    执行简单的数学计算
    """
    try:
        # 安全地执行数学表达式
        # 注意：实际应用中需要更严格的安全检查
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "不支持的字符，只允许数字和基本运算符"
        
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


# 工具注册表
TOOLS = {
    "get_weather": {
        "function": get_weather,
        "description": "获取指定城市的天气信息",
        "parameters": {
            "city": "城市名称，如：北京、上海、广州"
        }
    },
    "search_info": {
        "function": search_info,
        "description": "搜索相关信息",
        "parameters": {
            "query": "搜索关键词"
        }
    },
    "calculate": {
        "function": calculate,
        "description": "执行数学计算",
        "parameters": {
            "expression": "数学表达式，如：1+2*3"
        }
    }
}


# =============================================================================
# 第三部分：模拟 LLM（用于演示，不需要 API Key）
# =============================================================================

class MockLLM:
    """
    模拟 LLM 响应
    
    在实际应用中，替换为真实的 LLM（如 OpenAI、Claude 等）
    """
    
    def __init__(self, tools: dict):
        self.tools = tools
    
    def invoke(self, messages: list) -> dict:
        """
        模拟 LLM 的响应
        
        返回格式：
        - content: 文本响应
        - tool_calls: 需要调用的工具列表（如果有）
        """
        # 检查最后一条消息是否是工具响应
        # 如果是，则生成基于工具结果的最终响应
        last_msg = messages[-1] if messages else None
        if last_msg:
            if isinstance(last_msg, dict):
                last_role = last_msg.get("role", "")
                last_content = last_msg.get("content", "")
            else:
                last_role = getattr(last_msg, "type", "")
                last_content = getattr(last_msg, "content", "")
            
            # 如果上一条是工具响应，直接返回最终答案
            if last_role == "tool":
                return {
                    "role": "assistant",
                    "content": f"根据查询结果：{last_content}"
                }
        
        # 获取最后一条用户消息
        user_message = ""
        for msg in reversed(messages):
            # 兼容字典和 LangGraph 消息对象
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            else:
                # LangGraph 消息对象 (HumanMessage, AIMessage 等)
                if getattr(msg, "type", "") == "human":
                    user_message = getattr(msg, "content", "")
                    break
        
        user_message = user_message.lower()
        
        # 简单的意图识别和工具调用决策
        if "天气" in user_message:
            # 提取城市
            cities = ["北京", "上海", "广州", "深圳"]
            for city in cities:
                if city in user_message:
                    return {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_1",
                            "name": "get_weather",
                            "args": {"city": city}
                        }]
                    }
            return {
                "role": "assistant",
                "content": "请告诉我您想查询哪个城市的天气？支持：北京、上海、广州、深圳"
            }
        
        elif "搜索" in user_message or "什么是" in user_message:
            query = user_message.replace("搜索", "").replace("什么是", "").strip()
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_2",
                    "name": "search_info",
                    "args": {"query": query}
                }]
            }
        
        elif "计算" in user_message or any(c in user_message for c in "+-*/"):
            # 尝试提取数学表达式
            expr_match = re.search(r'[\d\s\+\-\*\/\(\)\.]+', user_message)
            if expr_match:
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_3",
                        "name": "calculate",
                        "args": {"expression": expr_match.group().strip()}
                    }]
                }
        
        # 默认：直接回复
        return {
            "role": "assistant",
            "content": f"你好！我是一个 AI 助手，可以帮你：\n1. 查询天气（如：北京天气怎么样）\n2. 搜索信息（如：什么是 LangGraph）\n3. 数学计算（如：计算 123 * 456）"
        }


# =============================================================================
# 第四部分：节点函数
# =============================================================================

# 创建模拟 LLM
mock_llm = MockLLM(TOOLS)


def agent_node(state: AgentState) -> dict:
    """
    Agent 节点：调用 LLM 生成响应
    
    职责：
    1. 将当前状态（消息历史）发送给 LLM
    2. 获取 LLM 的响应（可能包含工具调用）
    3. 将响应添加到消息历史
    """
    messages = state["messages"]
    
    # 调用 LLM
    response = mock_llm.invoke(messages)
    
    # 返回状态更新
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """
    工具执行节点：执行 LLM 请求的工具
    
    职责：
    1. 从最后一条消息中提取工具调用
    2. 执行对应的工具
    3. 将结果添加到消息历史
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # 获取工具调用（兼容字典和消息对象）
    if isinstance(last_message, dict):
        tool_calls = last_message.get("tool_calls", [])
    else:
        tool_calls = getattr(last_message, "tool_calls", []) or []
    
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # 执行工具
        if tool_name in TOOLS:
            tool_func = TOOLS[tool_name]["function"]
            result = tool_func(**tool_args)
        else:
            result = f"未知工具: {tool_name}"
        
        # 构造工具响应消息
        results.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "name": tool_name,
            "content": result
        })
    
    return {"messages": results}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    路由函数：决定下一步走向
    
    判断逻辑：
    - 如果 LLM 响应包含工具调用 → 转到 tools 节点
    - 否则 → 结束
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # 检查是否有工具调用
    # 兼容字典和 LangGraph 消息对象两种格式
    if isinstance(last_message, dict):
        tool_calls = last_message.get("tool_calls", [])
    else:
        # LangGraph 消息对象
        tool_calls = getattr(last_message, "tool_calls", []) or []
    
    if tool_calls:
        return "tools"
    return "end"


# =============================================================================
# 第五部分：构建图
# =============================================================================

def build_agent_graph():
    """
    构建 Agent 图
    
    结构：
                    ┌──────────────────────┐
                    ↓                      │
        START → agent → should_continue? ──┼→ tools
                    ↓                      │
                   END ←───────────────────┘
    """
    graph = StateGraph(AgentState)
    
    # 添加节点
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    
    # 设置入口
    graph.add_edge(START, "agent")
    
    # 添加条件边
    graph.add_conditional_edges(
        "agent",  # 从 agent 节点出发
        should_continue,  # 使用这个函数决定去向
        {
            "tools": "tools",  # 如果返回 "tools"，转到 tools 节点
            "end": END,        # 如果返回 "end"，结束
        }
    )
    
    # 工具执行后，回到 agent
    graph.add_edge("tools", "agent")
    
    # 编译
    app = graph.compile()
    
    return app


# =============================================================================
# 第六部分：演示
# =============================================================================

def get_message_content(msg) -> str:
    """获取消息内容，兼容字典和消息对象"""
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", str(msg))


def demo_simple_chat():
    """演示简单对话"""
    print("=" * 60)
    print("演示 1: 简单对话（无工具调用）")
    print("=" * 60)
    
    app = build_agent_graph()
    
    result = app.invoke({
        "messages": [{"role": "user", "content": "你好"}]
    })
    
    print("\n用户: 你好")
    print(f"助手: {get_message_content(result['messages'][-1])}")


def demo_tool_call():
    """演示工具调用"""
    print("\n" + "=" * 60)
    print("演示 2: 工具调用（查询天气）")
    print("=" * 60)
    
    app = build_agent_graph()
    
    # 流式执行，观察每一步
    print("\n用户: 北京天气怎么样？")
    print("\n执行过程：")
    
    for step_num, state_update in enumerate(app.stream({
        "messages": [{"role": "user", "content": "北京天气怎么样？"}]
    })):
        print(f"\n步骤 {step_num + 1}:")
        for node_name, node_output in state_update.items():
            print(f"  节点: {node_name}")
            messages = node_output.get("messages", []) if isinstance(node_output, dict) else []
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    tool_calls = msg.get("tool_calls", [])
                else:
                    role = getattr(msg, "type", "unknown")
                    content = getattr(msg, "content", "")
                    tool_calls = getattr(msg, "tool_calls", []) or []
                
                if tool_calls:
                    tc = tool_calls[0]
                    if isinstance(tc, dict):
                        tc_name = tc.get("name", "")
                        tc_args = tc.get("args", {})
                    else:
                        tc_name = getattr(tc, "name", "")
                        tc_args = getattr(tc, "args", {})
                    print(f"    [{role}] 调用工具: {tc_name}({tc_args})")
                elif content:
                    print(f"    [{role}] {content}")


def demo_calculation():
    """演示计算器工具"""
    print("\n" + "=" * 60)
    print("演示 3: 工具调用（计算器）")
    print("=" * 60)
    
    app = build_agent_graph()
    
    result = app.invoke({
        "messages": [{"role": "user", "content": "帮我计算 123 * 456 + 789"}]
    })
    
    print("\n用户: 帮我计算 123 * 456 + 789")
    
    # 找到工具返回的结果
    for msg in result["messages"]:
        if isinstance(msg, dict):
            if msg.get("role") == "tool":
                print(f"计算结果: {msg['content']}")
        else:
            # LangGraph 消息对象
            if getattr(msg, "type", "") == "tool":
                print(f"计算结果: {getattr(msg, 'content', '')}")


def demo_multi_turn():
    """演示多轮对话"""
    print("\n" + "=" * 60)
    print("演示 4: 多轮对话")
    print("=" * 60)
    
    app = build_agent_graph()
    
    # 模拟多轮对话
    conversations = [
        "你好，介绍一下你自己",
        "上海天气怎么样？",
        "什么是 AI Agent？"
    ]
    
    messages = []
    
    for user_input in conversations:
        print(f"\n用户: {user_input}")
        messages.append({"role": "user", "content": user_input})
        
        result = app.invoke({"messages": messages})
        
        # 获取新增的消息
        new_messages = result["messages"][len(messages):]
        messages = result["messages"]
        
        # 打印助手响应
        for msg in new_messages:
            if isinstance(msg, dict):
                if msg.get("role") == "assistant" and msg.get("content"):
                    print(f"助手: {msg['content']}")
                elif msg.get("role") == "tool":
                    print(f"[工具 {msg['name']}]: {msg['content']}")
            else:
                # LangGraph 消息对象
                msg_type = getattr(msg, "type", "")
                content = getattr(msg, "content", "")
                if msg_type == "ai" and content:
                    print(f"助手: {content}")
                elif msg_type == "tool":
                    print(f"[工具 {getattr(msg, 'name', '')}]: {content}")


# =============================================================================
# 使用真实 LLM 的版本（需要配置 API Key）
# =============================================================================

def build_real_agent():
    """
    使用真实 LLM 的 Agent（需要 OpenAI API Key）
    
    取消注释以使用：
    """
    pass
    # from langchain_openai import ChatOpenAI
    # from langchain_core.tools import tool
    # 
    # # 定义工具
    # @tool
    # def get_weather(city: str) -> str:
    #     """获取城市天气"""
    #     return f"{city}的天气：晴天，25°C"
    # 
    # # 创建 LLM
    # llm = ChatOpenAI(model="gpt-4o-mini")
    # llm_with_tools = llm.bind_tools([get_weather])
    # 
    # def agent(state: AgentState):
    #     return {"messages": [llm_with_tools.invoke(state["messages"])]}
    # 
    # # ... 其余构建逻辑相同


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    demo_simple_chat()
    demo_tool_call()
    demo_calculation()
    demo_multi_turn()
    
    print("\n" + "=" * 60)
    print("工具调用教程完成！")
    print("=" * 60)
    print("""
关键要点：
1. 工具是普通的 Python 函数，可以执行任意操作
2. Agent 循环模式：LLM → 判断是否需要工具 → 执行工具 → 返回 LLM
3. 条件边 (conditional_edges) 用于实现分支逻辑
4. 消息历史使用 add_messages reducer 自动累积
5. 工具响应需要包含 tool_call_id 以关联请求和响应

ReAct 模式核心思想：
- Reason (推理): LLM 分析问题，决定是否需要工具
- Act (行动): 执行工具获取信息
- 循环直到能够给出最终答案

下一步：学习 03_human_in_the_loop.py，了解人机交互
""")
