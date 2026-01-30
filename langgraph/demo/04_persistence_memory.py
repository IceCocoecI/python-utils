"""
LangGraph 基础教程 04: 持久化与记忆 (Persistence & Memory)
================================================

本示例演示 LangGraph 的持久化和记忆能力：
1. Checkpointer - 检查点，保存执行状态
2. Thread - 线程，独立的执行上下文
3. Time Travel - 时间旅行，回滚到历史状态
4. Memory - 记忆，跨会话的长期记忆

学习目标：
- 理解检查点的工作原理
- 掌握状态持久化和恢复
- 学会实现跨会话记忆

核心价值：
- 失败恢复：从崩溃点继续执行
- 对话历史：保持上下文连贯性
- 时间旅行：调试和回滚
- 长期记忆：用户偏好和知识积累
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import json
from datetime import datetime


# =============================================================================
# 第一部分：检查点基础
# =============================================================================

class ChatState(TypedDict):
    """聊天状态"""
    messages: Annotated[list, add_messages]
    user_name: str


def chat_node(state: ChatState) -> dict:
    """简单聊天节点"""
    messages = state["messages"]
    user_name = state.get("user_name", "用户")
    
    # 获取最后一条消息
    last_msg = messages[-1] if messages else None
    
    if last_msg:
        # 兼容字典和 LangGraph 消息对象
        if isinstance(last_msg, dict):
            content = last_msg.get("content", "")
        else:
            content = getattr(last_msg, "content", str(last_msg))
        response = f"{user_name}，你说的是：'{content}'。我记住了！"
    else:
        response = f"你好 {user_name}！有什么可以帮你的吗？"
    
    return {"messages": [{"role": "assistant", "content": response}]}


def build_chat_graph_with_memory():
    """构建带持久化的聊天图"""
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    
    # 使用内存检查点（开发用）
    # 生产环境应使用 PostgresSaver, SqliteSaver 等
    checkpointer = MemorySaver()
    
    return graph.compile(checkpointer=checkpointer)


def demo_basic_persistence():
    """演示基本持久化"""
    print("=" * 60)
    print("演示 1: 基本持久化（对话历史）")
    print("=" * 60)
    
    app = build_chat_graph_with_memory()
    
    # thread_id 标识一个对话会话
    config = {"configurable": {"thread_id": "user_alice_conversation"}}
    
    def get_msg_content(msg):
        """获取消息内容"""
        if isinstance(msg, dict):
            return msg.get("content", "")
        return getattr(msg, "content", str(msg))
    
    # 第一轮对话
    print("\n第一轮对话：")
    print("-" * 40)
    result1 = app.invoke({
        "messages": [{"role": "user", "content": "我叫小明"}],
        "user_name": "小明"
    }, config)
    print(f"用户: 我叫小明")
    print(f"助手: {get_msg_content(result1['messages'][-1])}")
    
    # 第二轮对话（会保留历史）
    print("\n第二轮对话（保留历史）：")
    print("-" * 40)
    result2 = app.invoke({
        "messages": [{"role": "user", "content": "今天天气真好"}]
    }, config)
    print(f"用户: 今天天气真好")
    print(f"助手: {get_msg_content(result2['messages'][-1])}")
    
    # 查看完整对话历史
    print("\n完整对话历史：")
    print("-" * 40)
    state = app.get_state(config)
    for i, msg in enumerate(state.values["messages"]):
        # 兼容字典和消息对象
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", str(msg))
        print(f"  {i+1}. [{role}] {content}")


# =============================================================================
# 第二部分：多线程（多会话）
# =============================================================================

def demo_multiple_threads():
    """演示多线程（独立会话）"""
    print("\n" + "=" * 60)
    print("演示 2: 多线程（独立会话）")
    print("=" * 60)
    
    app = build_chat_graph_with_memory()
    
    # 两个不同的用户（不同的 thread_id）
    config_alice = {"configurable": {"thread_id": "alice_session"}}
    config_bob = {"configurable": {"thread_id": "bob_session"}}
    
    # Alice 的对话
    print("\nAlice 的对话：")
    print("-" * 40)
    app.invoke({
        "messages": [{"role": "user", "content": "我喜欢Python"}],
        "user_name": "Alice"
    }, config_alice)
    
    # Bob 的对话
    print("\nBob 的对话：")
    print("-" * 40)
    app.invoke({
        "messages": [{"role": "user", "content": "我喜欢JavaScript"}],
        "user_name": "Bob"
    }, config_bob)
    
    # 继续 Alice 的对话
    print("\n继续 Alice 的对话：")
    print("-" * 40)
    result_alice = app.invoke({
        "messages": [{"role": "user", "content": "我最近在学习LangGraph"}]
    }, config_alice)
    
    # 查看两个用户的状态
    alice_state = app.get_state(config_alice)
    bob_state = app.get_state(config_bob)
    
    print(f"\nAlice 消息数: {len(alice_state.values['messages'])}")
    print(f"Bob 消息数: {len(bob_state.values['messages'])}")
    print("\n结论：不同 thread_id 的会话完全独立")


# =============================================================================
# 第三部分：时间旅行
# =============================================================================

class CounterState(TypedDict):
    """计数器状态"""
    count: int
    history: list[str]


def increment(state: CounterState) -> dict:
    """增加计数"""
    new_count = state["count"] + 1
    timestamp = datetime.now().strftime("%H:%M:%S")
    return {
        "count": new_count,
        "history": state["history"] + [f"[{timestamp}] 计数: {new_count}"]
    }


def build_counter_graph():
    """构建计数器图"""
    graph = StateGraph(CounterState)
    graph.add_node("increment", increment)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)
    
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def demo_time_travel():
    """演示时间旅行"""
    print("\n" + "=" * 60)
    print("演示 3: 时间旅行")
    print("=" * 60)
    
    app = build_counter_graph()
    config = {"configurable": {"thread_id": "counter_demo"}}
    
    # 执行多次，创建历史
    print("\n执行 5 次递增：")
    print("-" * 40)
    
    for i in range(5):
        result = app.invoke({"count": i, "history": []}, config)
        print(f"  第 {i+1} 次: count = {result['count']}")
    
    # 获取状态历史
    print("\n状态历史（检查点）：")
    print("-" * 40)
    
    history = list(app.get_state_history(config))
    print(f"共有 {len(history)} 个检查点")
    
    # 显示部分历史
    for i, state_snapshot in enumerate(history[:3]):  # 只显示前3个
        print(f"\n  检查点 {i+1}:")
        print(f"    config: {state_snapshot.config}")
        print(f"    count: {state_snapshot.values.get('count', 'N/A')}")
    
    # 时间旅行：回滚到之前的状态
    if len(history) > 2:
        print("\n时间旅行：回滚到第 2 个检查点")
        print("-" * 40)
        
        # 获取要回滚到的检查点配置
        rollback_config = history[2].config
        print(f"回滚到 config: {rollback_config}")
        
        # 从该检查点继续执行
        rollback_result = app.invoke(None, rollback_config)
        print(f"回滚后执行结果: count = {rollback_result['count']}")


# =============================================================================
# 第四部分：长期记忆（跨会话）
# =============================================================================

class MemoryState(TypedDict):
    """带长期记忆的状态"""
    messages: Annotated[list, add_messages]
    user_preferences: dict


# 模拟持久化存储（实际应用中使用数据库）
PERSISTENT_MEMORY = {}


def load_user_memory(user_id: str) -> dict:
    """加载用户记忆"""
    return PERSISTENT_MEMORY.get(user_id, {
        "preferences": {},
        "facts": [],
        "last_seen": None
    })


def save_user_memory(user_id: str, memory: dict):
    """保存用户记忆"""
    PERSISTENT_MEMORY[user_id] = memory


def memory_aware_chat(state: MemoryState) -> dict:
    """记忆感知的聊天节点"""
    messages = state["messages"]
    preferences = state.get("user_preferences", {})
    
    last_msg = messages[-1] if messages else {}
    # 兼容字典和消息对象
    if isinstance(last_msg, dict):
        content = last_msg.get("content", "")
    else:
        content = getattr(last_msg, "content", "")
    
    # 提取并记住偏好
    response_parts = []
    
    # 检查是否是询问偏好的问题（不是表达偏好）
    is_question = any(q in content for q in ["吗", "？", "?", "什么", "哪些"])
    
    if "喜欢" in content and not is_question and "不喜欢" not in content:
        # 提取喜好（排除疑问句和否定句）
        thing = content.split("喜欢")[-1].strip().rstrip("。！?？")
        if thing and len(thing) > 0:
            preferences["likes"] = preferences.get("likes", [])
            if thing not in preferences["likes"]:
                preferences["likes"].append(thing)
            response_parts.append(f"我记住了，你喜欢{thing}")
    
    if ("讨厌" in content or "不喜欢" in content) and not is_question:
        # 提取不喜欢的事物
        if "不喜欢" in content:
            thing = content.split("不喜欢")[-1].strip().rstrip("。！?？")
        else:
            thing = content.split("讨厌")[-1].strip().rstrip("。！?？")
        if thing and len(thing) > 0:
            preferences["dislikes"] = preferences.get("dislikes", [])
            if thing not in preferences["dislikes"]:
                preferences["dislikes"].append(thing)
            response_parts.append(f"我记住了，你不喜欢{thing}")
    
    # 使用已知偏好
    if preferences.get("likes"):
        response_parts.append(f"我知道你喜欢：{', '.join(preferences['likes'])}")
    
    if not response_parts:
        response_parts.append("告诉我你的喜好，我会记住的！")
    
    response = " ".join(response_parts)
    
    return {
        "messages": [{"role": "assistant", "content": response}],
        "user_preferences": preferences
    }


def build_memory_graph():
    """构建带记忆的图"""
    graph = StateGraph(MemoryState)
    graph.add_node("chat", memory_aware_chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def demo_long_term_memory():
    """演示长期记忆"""
    print("\n" + "=" * 60)
    print("演示 4: 长期记忆（偏好学习）")
    print("=" * 60)
    
    app = build_memory_graph()
    user_id = "user_123"
    
    # 加载用户历史记忆
    user_memory = load_user_memory(user_id)
    config = {"configurable": {"thread_id": f"memory_{user_id}"}}
    
    conversations = [
        "我喜欢Python",
        "我还喜欢咖啡",
        "我讨厌加班",
        "你还记得我喜欢什么吗？"
    ]
    
    current_preferences = user_memory.get("preferences", {})
    
    for msg in conversations:
        print(f"\n用户: {msg}")
        result = app.invoke({
            "messages": [{"role": "user", "content": msg}],
            "user_preferences": current_preferences
        }, config)
        
        response = result["messages"][-1]
        if isinstance(response, dict):
            content = response.get("content", "")
        else:
            content = getattr(response, "content", str(response))
        print(f"助手: {content}")
        
        # 更新偏好
        current_preferences = result.get("user_preferences", current_preferences)
    
    # 保存用户记忆
    user_memory["preferences"] = current_preferences
    user_memory["last_seen"] = datetime.now().isoformat()
    save_user_memory(user_id, user_memory)
    
    print("\n" + "-" * 40)
    print(f"已保存用户记忆: {json.dumps(user_memory, ensure_ascii=False, indent=2)}")


# =============================================================================
# 第五部分：不同的检查点存储
# =============================================================================

def explain_checkpointers():
    """解释不同的检查点存储"""
    print("\n" + "=" * 60)
    print("补充说明: 检查点存储选项")
    print("=" * 60)
    
    print("""
LangGraph 支持多种检查点存储：

1. MemorySaver（内存）
   - 适用于：开发和测试
   - 特点：快速，但重启后数据丢失
   - 代码：
     from langgraph.checkpoint.memory import MemorySaver
     checkpointer = MemorySaver()

2. SqliteSaver（SQLite）
   - 适用于：单机部署，小规模应用
   - 特点：持久化到文件，简单可靠
   - 代码：
     from langgraph.checkpoint.sqlite import SqliteSaver
     checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

3. PostgresSaver（PostgreSQL）
   - 适用于：生产环境，多实例部署
   - 特点：支持并发，可扩展
   - 代码：
     from langgraph.checkpoint.postgres import PostgresSaver
     checkpointer = PostgresSaver.from_conn_string(
         "postgresql://user:pass@localhost/db"
     )

4. 自定义 Checkpointer
   - 继承 BaseCheckpointSaver 实现自己的存储
   - 可以使用 Redis, MongoDB, S3 等

选择建议：
- 开发测试：MemorySaver
- 小型应用：SqliteSaver
- 生产环境：PostgresSaver
""")


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    demo_basic_persistence()
    demo_multiple_threads()
    demo_time_travel()
    demo_long_term_memory()
    explain_checkpointers()
    
    print("\n" + "=" * 60)
    print("持久化与记忆教程完成！")
    print("=" * 60)
    print("""
关键要点：
1. Checkpointer 是持久化的核心，保存每一步状态
2. thread_id 标识独立的执行上下文（会话）
3. 不同 thread 的状态完全隔离
4. get_state_history() 获取历史检查点
5. 可以回滚到任意历史检查点继续执行
6. 长期记忆需要额外的持久化存储

持久化带来的能力：
- 失败恢复：从崩溃点继续
- 对话连贯：保持上下文
- 时间旅行：调试和回滚
- 人机交互：中断和恢复

下一步：学习 05_conditional_routing.py，了解条件路由
""")
