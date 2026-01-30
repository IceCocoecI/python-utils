"""
LangGraph 基础教程 05: 条件路由 (Conditional Routing)
================================================

本示例演示 LangGraph 的条件路由能力：
1. 条件边 (Conditional Edges) - 根据状态决定下一步
2. 路由函数 (Router Function) - 分支逻辑
3. 分支合并 - 多路径汇聚
4. 复杂流程控制 - 循环、重试、分发

学习目标：
- 理解条件边的工作原理
- 掌握路由函数设计
- 学会实现复杂的业务流程

核心模式：
- 意图分类 + 路由
- 质量检查 + 重试
- 并行分发 + 汇聚
"""

import random
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


# =============================================================================
# 第一部分：基本条件路由
# =============================================================================

class SimpleRouteState(TypedDict):
    """简单路由状态"""
    input_type: str  # "question", "command", "chat"
    input_text: str
    response: str


def classify_input(state: SimpleRouteState) -> dict:
    """分类输入类型"""
    text = state["input_text"].lower()
    
    if any(q in text for q in ["什么", "怎么", "为什么", "如何", "？", "?"]):
        return {"input_type": "question"}
    elif any(c in text for c in ["执行", "运行", "打开", "关闭", "创建"]):
        return {"input_type": "command"}
    else:
        return {"input_type": "chat"}


def route_by_type(state: SimpleRouteState) -> str:
    """
    路由函数：根据输入类型决定下一个节点
    
    返回值必须是 add_conditional_edges 中定义的键之一
    """
    input_type = state["input_type"]
    
    if input_type == "question":
        return "answer"
    elif input_type == "command":
        return "execute"
    else:
        return "chat"


def answer_question(state: SimpleRouteState) -> dict:
    """回答问题"""
    return {"response": f"[问答模式] 关于 '{state['input_text']}' 的回答..."}


def execute_command(state: SimpleRouteState) -> dict:
    """执行命令"""
    return {"response": f"[命令模式] 正在执行: {state['input_text']}"}


def casual_chat(state: SimpleRouteState) -> dict:
    """闲聊"""
    return {"response": f"[闲聊模式] {state['input_text']}? 有趣的话题！"}


def build_simple_router():
    """构建简单路由图"""
    graph = StateGraph(SimpleRouteState)
    
    # 添加节点
    graph.add_node("classify", classify_input)
    graph.add_node("answer", answer_question)
    graph.add_node("execute", execute_command)
    graph.add_node("chat", casual_chat)
    
    # 入口边
    graph.add_edge(START, "classify")
    
    # 条件边：从 classify 根据路由函数分发
    graph.add_conditional_edges(
        "classify",       # 源节点
        route_by_type,    # 路由函数
        {                 # 路由映射
            "answer": "answer",
            "execute": "execute",
            "chat": "chat"
        }
    )
    
    # 所有分支汇聚到 END
    graph.add_edge("answer", END)
    graph.add_edge("execute", END)
    graph.add_edge("chat", END)
    
    return graph.compile()


def demo_simple_routing():
    """演示简单路由"""
    print("=" * 60)
    print("演示 1: 简单条件路由")
    print("=" * 60)
    
    app = build_simple_router()
    
    test_inputs = [
        "Python是什么？",
        "执行数据备份",
        "今天真是个好天气"
    ]
    
    for text in test_inputs:
        result = app.invoke({
            "input_type": "",
            "input_text": text,
            "response": ""
        })
        print(f"\n输入: {text}")
        print(f"类型: {result['input_type']}")
        print(f"响应: {result['response']}")


# =============================================================================
# 第二部分：质量检查与重试
# =============================================================================

class QualityCheckState(TypedDict):
    """质量检查状态"""
    task: str
    result: str
    quality_score: float
    retry_count: int
    max_retries: int


def generate_result(state: QualityCheckState) -> dict:
    """生成结果"""
    retry_count = state["retry_count"]
    
    # 模拟：前几次生成低质量结果，最后一次生成高质量结果
    if retry_count < 2:
        quality = 0.3 + retry_count * 0.2
        result = f"低质量结果 (尝试 {retry_count + 1})"
    else:
        quality = 0.9
        result = f"高质量结果 (尝试 {retry_count + 1})"
    
    return {
        "result": result,
        "quality_score": quality,
        "retry_count": retry_count + 1
    }


def check_quality(state: QualityCheckState) -> Literal["accept", "retry", "fail"]:
    """
    质量检查路由函数
    
    返回值决定下一步：
    - "accept": 质量合格，接受结果
    - "retry": 质量不合格但可重试
    - "fail": 超过重试次数，失败
    """
    if state["quality_score"] >= 0.8:
        return "accept"
    elif state["retry_count"] < state["max_retries"]:
        return "retry"
    else:
        return "fail"


def accept_result(state: QualityCheckState) -> dict:
    """接受结果"""
    return {"result": f"✓ 已接受: {state['result']}"}


def handle_failure(state: QualityCheckState) -> dict:
    """处理失败"""
    return {"result": f"✗ 失败: 在 {state['retry_count']} 次尝试后仍无法生成高质量结果"}


def build_quality_check_graph():
    """构建质量检查图"""
    graph = StateGraph(QualityCheckState)
    
    graph.add_node("generate", generate_result)
    graph.add_node("accept", accept_result)
    graph.add_node("fail", handle_failure)
    
    graph.add_edge(START, "generate")
    
    # 条件边：检查质量决定下一步
    graph.add_conditional_edges(
        "generate",
        check_quality,
        {
            "accept": "accept",  # 质量合格
            "retry": "generate",  # 重试（循环回 generate）
            "fail": "fail"       # 失败
        }
    )
    
    graph.add_edge("accept", END)
    graph.add_edge("fail", END)
    
    return graph.compile()


def demo_quality_check():
    """演示质量检查与重试"""
    print("\n" + "=" * 60)
    print("演示 2: 质量检查与重试")
    print("=" * 60)
    
    app = build_quality_check_graph()
    
    print("\n流式执行（观察重试过程）：")
    print("-" * 40)
    
    for step_num, event in enumerate(app.stream({
        "task": "生成报告",
        "result": "",
        "quality_score": 0.0,
        "retry_count": 0,
        "max_retries": 5
    })):
        for node, output in event.items():
            print(f"\n步骤 {step_num + 1} - 节点: {node}")
            if output.get("quality_score"):
                print(f"  质量分数: {output['quality_score']}")
            if output.get("result"):
                print(f"  结果: {output['result']}")


# =============================================================================
# 第三部分：多条件分支
# =============================================================================

class PriorityState(TypedDict):
    """优先级路由状态"""
    task: str
    priority: str  # "low", "medium", "high", "critical"
    processed_by: str
    result: str


def analyze_priority(state: PriorityState) -> dict:
    """分析任务优先级"""
    task = state["task"].lower()
    
    if any(word in task for word in ["紧急", "立即", "马上", "critical"]):
        priority = "critical"
    elif any(word in task for word in ["重要", "尽快", "high"]):
        priority = "high"
    elif any(word in task for word in ["普通", "一般", "medium"]):
        priority = "medium"
    else:
        priority = "low"
    
    return {"priority": priority}


def route_by_priority(state: PriorityState) -> str:
    """根据优先级路由"""
    priority_map = {
        "critical": "critical_handler",
        "high": "high_handler",
        "medium": "medium_handler",
        "low": "low_handler"
    }
    return priority_map.get(state["priority"], "low_handler")


def critical_handler(state: PriorityState) -> dict:
    return {
        "processed_by": "紧急处理团队",
        "result": f"[紧急] 已立即处理: {state['task']}"
    }


def high_handler(state: PriorityState) -> dict:
    return {
        "processed_by": "高级工程师",
        "result": f"[高优先级] 已优先处理: {state['task']}"
    }


def medium_handler(state: PriorityState) -> dict:
    return {
        "processed_by": "普通工程师",
        "result": f"[中等优先级] 已排队处理: {state['task']}"
    }


def low_handler(state: PriorityState) -> dict:
    return {
        "processed_by": "自动化系统",
        "result": f"[低优先级] 已自动处理: {state['task']}"
    }


def build_priority_router():
    """构建优先级路由图"""
    graph = StateGraph(PriorityState)
    
    graph.add_node("analyze", analyze_priority)
    graph.add_node("critical_handler", critical_handler)
    graph.add_node("high_handler", high_handler)
    graph.add_node("medium_handler", medium_handler)
    graph.add_node("low_handler", low_handler)
    
    graph.add_edge(START, "analyze")
    
    graph.add_conditional_edges(
        "analyze",
        route_by_priority,
        {
            "critical_handler": "critical_handler",
            "high_handler": "high_handler",
            "medium_handler": "medium_handler",
            "low_handler": "low_handler"
        }
    )
    
    # 所有处理器最后都到 END
    for handler in ["critical_handler", "high_handler", "medium_handler", "low_handler"]:
        graph.add_edge(handler, END)
    
    return graph.compile()


def demo_priority_routing():
    """演示优先级路由"""
    print("\n" + "=" * 60)
    print("演示 3: 多条件分支（优先级路由）")
    print("=" * 60)
    
    app = build_priority_router()
    
    tasks = [
        "紧急！服务器宕机",
        "重要：更新用户权限",
        "普通任务：生成周报",
        "清理临时文件"
    ]
    
    for task in tasks:
        result = app.invoke({
            "task": task,
            "priority": "",
            "processed_by": "",
            "result": ""
        })
        print(f"\n任务: {task}")
        print(f"  优先级: {result['priority']}")
        print(f"  处理者: {result['processed_by']}")


# =============================================================================
# 第四部分：分支后合并
# =============================================================================

class ParallelState(TypedDict):
    """并行处理状态"""
    data: str
    branch_a_result: str
    branch_b_result: str
    final_result: str


def branch_a(state: ParallelState) -> dict:
    """分支 A 处理"""
    return {"branch_a_result": f"分支A处理: {state['data'].upper()}"}


def branch_b(state: ParallelState) -> dict:
    """分支 B 处理"""
    return {"branch_b_result": f"分支B处理: {state['data'][::-1]}"}


def merge_results(state: ParallelState) -> dict:
    """合并分支结果"""
    a_result = state.get("branch_a_result", "")
    b_result = state.get("branch_b_result", "")
    return {"final_result": f"合并结果: [{a_result}] + [{b_result}]"}


def build_parallel_graph():
    """构建并行处理图
    
    注意：真正的并行需要使用 Send API
    这里演示的是条件路由 + 合并的模式
    """
    graph = StateGraph(ParallelState)
    
    graph.add_node("branch_a", branch_a)
    graph.add_node("branch_b", branch_b)
    graph.add_node("merge", merge_results)
    
    # 简化示例：顺序执行两个分支然后合并
    graph.add_edge(START, "branch_a")
    graph.add_edge("branch_a", "branch_b")
    graph.add_edge("branch_b", "merge")
    graph.add_edge("merge", END)
    
    return graph.compile()


def demo_branch_merge():
    """演示分支合并"""
    print("\n" + "=" * 60)
    print("演示 4: 分支处理与合并")
    print("=" * 60)
    
    app = build_parallel_graph()
    
    result = app.invoke({
        "data": "Hello",
        "branch_a_result": "",
        "branch_b_result": "",
        "final_result": ""
    })
    
    print(f"\n原始数据: Hello")
    print(f"分支A结果: {result['branch_a_result']}")
    print(f"分支B结果: {result['branch_b_result']}")
    print(f"最终结果: {result['final_result']}")


# =============================================================================
# 第五部分：状态机模式
# =============================================================================

class OrderState(TypedDict):
    """订单状态机"""
    order_id: str
    status: str  # "created", "paid", "shipped", "delivered", "cancelled"
    history: list[str]
    message: str


def route_by_status(state: OrderState) -> str:
    """根据订单状态路由"""
    status = state["status"]
    return f"handle_{status}"


def handle_created(state: OrderState) -> dict:
    """处理新创建的订单"""
    # 模拟：70% 概率支付，30% 概率取消
    if random.random() > 0.3:
        new_status = "paid"
        msg = "订单已支付"
    else:
        new_status = "cancelled"
        msg = "订单已取消"
    
    return {
        "status": new_status,
        "message": msg,
        "history": state["history"] + [f"created -> {new_status}"]
    }


def handle_paid(state: OrderState) -> dict:
    """处理已支付的订单"""
    return {
        "status": "shipped",
        "message": "订单已发货",
        "history": state["history"] + ["paid -> shipped"]
    }


def handle_shipped(state: OrderState) -> dict:
    """处理已发货的订单"""
    return {
        "status": "delivered",
        "message": "订单已送达",
        "history": state["history"] + ["shipped -> delivered"]
    }


def handle_delivered(state: OrderState) -> dict:
    """处理已送达的订单"""
    return {"message": "订单完成！"}


def handle_cancelled(state: OrderState) -> dict:
    """处理已取消的订单"""
    return {"message": "订单已取消，流程结束"}


def should_continue_order(state: OrderState) -> str:
    """决定订单流程是否继续"""
    terminal_states = ["delivered", "cancelled"]
    if state["status"] in terminal_states:
        return "end"
    return "continue"


def route_node(state: OrderState) -> dict:
    """路由节点：不做任何处理，只是作为路由入口"""
    return {}


def build_order_state_machine():
    """构建订单状态机"""
    graph = StateGraph(OrderState)
    
    # 添加路由节点（作为循环入口点）
    graph.add_node("router", route_node)
    
    # 添加各状态处理节点
    graph.add_node("handle_created", handle_created)
    graph.add_node("handle_paid", handle_paid)
    graph.add_node("handle_shipped", handle_shipped)
    graph.add_node("handle_delivered", handle_delivered)
    graph.add_node("handle_cancelled", handle_cancelled)
    
    # 入口到路由节点
    graph.add_edge(START, "router")
    
    # 从路由节点根据状态分发
    graph.add_conditional_edges(
        "router",
        route_by_status,
        {
            "handle_created": "handle_created",
            "handle_paid": "handle_paid",
            "handle_shipped": "handle_shipped",
            "handle_delivered": "handle_delivered",
            "handle_cancelled": "handle_cancelled"
        }
    )
    
    # 每个处理节点后，检查是否继续
    for handler in ["handle_created", "handle_paid", "handle_shipped"]:
        graph.add_conditional_edges(
            handler,
            should_continue_order,
            {
                "continue": "router",  # 继续循环回到路由节点
                "end": END
            }
        )
    
    # 终态直接结束
    graph.add_edge("handle_delivered", END)
    graph.add_edge("handle_cancelled", END)
    
    return graph.compile()


def demo_state_machine():
    """演示状态机"""
    print("\n" + "=" * 60)
    print("演示 5: 状态机模式（订单处理）")
    print("=" * 60)
    
    # 设置随机种子以获得可重复的演示结果
    random.seed(42)
    
    app = build_order_state_machine()
    
    print("\n订单处理流程：")
    print("-" * 40)
    
    for step_num, event in enumerate(app.stream({
        "order_id": "ORD-001",
        "status": "created",
        "history": [],
        "message": ""
    })):
        for node, output in event.items():
            # 跳过空输出的节点（如路由节点）
            if output is None or output == {}:
                continue
            print(f"\n步骤 {step_num + 1}:")
            print(f"  节点: {node}")
            if isinstance(output, dict):
                if output.get("status"):
                    print(f"  新状态: {output['status']}")
                if output.get("message"):
                    print(f"  消息: {output['message']}")


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    demo_simple_routing()
    demo_quality_check()
    demo_priority_routing()
    demo_branch_merge()
    demo_state_machine()
    
    print("\n" + "=" * 60)
    print("条件路由教程完成！")
    print("=" * 60)
    print("""
关键要点：
1. add_conditional_edges() 实现条件分支
2. 路由函数返回字符串，对应目标节点
3. 条件边可以实现循环（目标是之前的节点）
4. 多个分支可以汇聚到同一个节点
5. 状态机模式：状态 + 路由 = 复杂流程控制

常见模式：
- 意图分类 + 路由分发
- 质量检查 + 重试循环
- 优先级分发
- 分支并行 + 结果合并
- 状态机流程

下一步：学习 06_multi_agent.py，了解多智能体协作
""")
