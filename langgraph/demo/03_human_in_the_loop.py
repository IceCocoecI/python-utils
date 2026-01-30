"""
LangGraph 基础教程 03: 人机交互 (Human-in-the-Loop)
================================================

本示例演示 LangGraph 的人机交互能力：
1. 中断点 (Interrupt) - 暂停执行等待人工输入
2. 状态检查 - 在执行过程中查看状态
3. 状态修改 - 人工修改状态后继续执行
4. 批准/拒绝 - 人工审核 Agent 决策

学习目标：
- 理解 interrupt 机制
- 掌握检查点与线程的使用
- 学会实现人工审批流程

应用场景：
- 敏感操作需要人工批准（如：发送邮件、执行交易）
- 用户需要提供额外输入
- Agent 需要人工指导
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# =============================================================================
# 第一部分：状态定义
# =============================================================================

class HumanInLoopState(TypedDict):
    """人机交互示例的状态"""
    messages: Annotated[list, add_messages]
    
    # 待执行的操作
    pending_action: dict | None
    
    # 人工审批结果
    human_approved: bool | None
    
    # 最终结果
    result: str


# =============================================================================
# 第二部分：示例1 - 基本中断
# =============================================================================

class BasicInterruptState(TypedDict):
    """基本中断示例的状态"""
    input_value: str
    human_input: str
    output: str


def step_before_interrupt(state: BasicInterruptState) -> dict:
    """中断前的处理步骤"""
    print("  [步骤1] 执行中断前的处理...")
    processed = f"预处理: {state['input_value'].upper()}"
    return {"input_value": processed}


def human_input_node(state: BasicInterruptState) -> dict:
    """
    需要人工输入的节点
    
    使用 interrupt() 暂停执行，等待人工提供输入
    """
    print("  [步骤2] 等待人工输入...")
    
    # interrupt() 会暂停图的执行
    # 参数是传递给人工的信息
    human_response = interrupt({
        "message": "请提供额外信息",
        "current_value": state["input_value"]
    })
    
    # 当图恢复执行时，interrupt() 返回人工提供的值
    return {"human_input": human_response}


def step_after_interrupt(state: BasicInterruptState) -> dict:
    """中断后的处理步骤"""
    print("  [步骤3] 处理人工输入，生成最终结果...")
    result = f"结果: {state['input_value']} + 人工输入: {state['human_input']}"
    return {"output": result}


def build_basic_interrupt_graph():
    """构建基本中断示例图"""
    graph = StateGraph(BasicInterruptState)
    
    graph.add_node("step1", step_before_interrupt)
    graph.add_node("human", human_input_node)
    graph.add_node("step3", step_after_interrupt)
    
    graph.add_edge(START, "step1")
    graph.add_edge("step1", "human")
    graph.add_edge("human", "step3")
    graph.add_edge("step3", END)
    
    # 必须使用 checkpointer 才能支持中断
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    
    return app


def demo_basic_interrupt():
    """演示基本中断机制"""
    print("=" * 60)
    print("演示 1: 基本中断机制")
    print("=" * 60)
    
    app = build_basic_interrupt_graph()
    
    # thread_id 标识一个执行上下文
    config = {"configurable": {"thread_id": "demo_thread_1"}}
    
    initial_state = {
        "input_value": "hello",
        "human_input": "",
        "output": ""
    }
    
    print("\n第一阶段：启动执行（会在中断点暂停）")
    print("-" * 40)
    
    # 首次调用会执行到中断点
    for event in app.stream(initial_state, config):
        print(f"  事件: {event}")
    
    # 检查当前状态
    current_state = app.get_state(config)
    print(f"\n当前状态: {current_state.values}")
    print(f"下一步节点: {current_state.next}")
    
    # 获取中断信息
    if current_state.tasks:
        for task in current_state.tasks:
            if hasattr(task, 'interrupts') and task.interrupts:
                print(f"中断信息: {task.interrupts[0].value}")
    
    print("\n第二阶段：提供人工输入，恢复执行")
    print("-" * 40)
    
    # 使用 Command 恢复执行，提供人工输入
    for event in app.stream(
        Command(resume="用户提供的信息"),  # 恢复执行并传入人工输入
        config
    ):
        print(f"  事件: {event}")
    
    # 获取最终结果
    final_state = app.get_state(config)
    print(f"\n最终结果: {final_state.values['output']}")


# =============================================================================
# 第三部分：示例2 - 审批流程
# =============================================================================

class ApprovalState(TypedDict):
    """审批流程状态"""
    request: str
    analysis: str
    action_plan: dict | None
    approved: bool | None
    result: str


def analyze_request(state: ApprovalState) -> dict:
    """分析请求"""
    request = state["request"]
    
    # 模拟分析
    if "删除" in request:
        risk_level = "高风险"
        action = {"type": "delete", "target": request.replace("删除", "").strip()}
    elif "发送" in request:
        risk_level = "中风险"
        action = {"type": "send", "content": request}
    else:
        risk_level = "低风险"
        action = {"type": "query", "content": request}
    
    return {
        "analysis": f"风险级别: {risk_level}",
        "action_plan": action
    }


def request_approval(state: ApprovalState) -> dict:
    """请求人工审批"""
    # 显示待审批的操作
    approval_request = {
        "message": "请审批以下操作",
        "request": state["request"],
        "analysis": state["analysis"],
        "action_plan": state["action_plan"]
    }
    
    # 中断等待审批
    response = interrupt(approval_request)
    
    # response 期望是 True/False 或 "approved"/"rejected"
    if isinstance(response, bool):
        approved = response
    else:
        approved = response.lower() in ["yes", "approved", "true", "1"]
    
    return {"approved": approved}


def execute_action(state: ApprovalState) -> dict:
    """执行操作"""
    if state["approved"]:
        action = state["action_plan"]
        result = f"已执行操作: {action['type']} - 成功"
    else:
        result = "操作已被拒绝，未执行任何操作"
    
    return {"result": result}


def build_approval_graph():
    """构建审批流程图"""
    graph = StateGraph(ApprovalState)
    
    graph.add_node("analyze", analyze_request)
    graph.add_node("approve", request_approval)
    graph.add_node("execute", execute_action)
    
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "approve")
    graph.add_edge("approve", "execute")
    graph.add_edge("execute", END)
    
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def demo_approval_flow():
    """演示审批流程"""
    print("\n" + "=" * 60)
    print("演示 2: 审批流程")
    print("=" * 60)
    
    app = build_approval_graph()
    config = {"configurable": {"thread_id": "approval_demo"}}
    
    # 高风险请求
    initial_state = {
        "request": "删除用户数据表",
        "analysis": "",
        "action_plan": None,
        "approved": None,
        "result": ""
    }
    
    print("\n提交请求: '删除用户数据表'")
    print("-" * 40)
    
    # 执行到中断点
    for event in app.stream(initial_state, config):
        for node, output in event.items():
            # 跳过 interrupt 事件
            if node == "__interrupt__":
                continue
            if isinstance(output, dict):
                if output.get("analysis"):
                    print(f"  分析结果: {output['analysis']}")
                if output.get("action_plan"):
                    print(f"  执行计划: {output['action_plan']}")
    
    # 获取待审批信息
    state = app.get_state(config)
    if state.tasks:
        for task in state.tasks:
            if hasattr(task, 'interrupts') and task.interrupts:
                print(f"\n待审批: {task.interrupts[0].value}")
    
    print("\n模拟人工审批: 拒绝")
    print("-" * 40)
    
    # 拒绝操作
    for event in app.stream(Command(resume=False), config):
        for node, output in event.items():
            if node == "__interrupt__":
                continue
            if isinstance(output, dict) and output.get("result"):
                print(f"  执行结果: {output['result']}")


# =============================================================================
# 第四部分：示例3 - 状态修改
# =============================================================================

class EditableState(TypedDict):
    """可编辑状态示例"""
    data: dict
    processed: bool
    output: str


def process_data(state: EditableState) -> dict:
    """处理数据"""
    data = state["data"]
    # 简单处理
    processed_data = {k: v.upper() if isinstance(v, str) else v for k, v in data.items()}
    return {"data": processed_data, "processed": True}


def generate_output(state: EditableState) -> dict:
    """生成输出"""
    return {"output": f"处理结果: {state['data']}"}


def build_editable_graph():
    """构建可编辑状态图"""
    graph = StateGraph(EditableState)
    
    graph.add_node("process", process_data)
    graph.add_node("output", generate_output)
    
    graph.add_edge(START, "process")
    graph.add_edge("process", "output")
    graph.add_edge("output", END)
    
    checkpointer = MemorySaver()
    # interrupt_before 在指定节点前中断
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["output"]  # 在 output 节点前中断
    )


def demo_state_editing():
    """演示状态修改"""
    print("\n" + "=" * 60)
    print("演示 3: 状态修改")
    print("=" * 60)
    
    app = build_editable_graph()
    config = {"configurable": {"thread_id": "edit_demo"}}
    
    initial_state = {
        "data": {"name": "alice", "city": "beijing"},
        "processed": False,
        "output": ""
    }
    
    print("\n初始数据:", initial_state["data"])
    print("-" * 40)
    
    # 执行到中断点
    for event in app.stream(initial_state, config):
        print(f"  执行: {event}")
    
    # 获取当前状态
    state = app.get_state(config)
    print(f"\n处理后状态: {state.values['data']}")
    
    print("\n人工修改状态...")
    print("-" * 40)
    
    # 人工修改状态
    modified_data = state.values["data"].copy()
    modified_data["extra_field"] = "ADDED_BY_HUMAN"
    
    # 更新状态
    app.update_state(
        config,
        {"data": modified_data}
    )
    
    # 检查修改后的状态
    new_state = app.get_state(config)
    print(f"修改后状态: {new_state.values['data']}")
    
    print("\n继续执行...")
    print("-" * 40)
    
    # 继续执行（传 None 表示继续而不提供新输入）
    for event in app.stream(None, config):
        print(f"  执行: {event}")
    
    final_state = app.get_state(config)
    print(f"\n最终输出: {final_state.values['output']}")


# =============================================================================
# 第五部分：示例4 - 条件中断
# =============================================================================

class ConditionalInterruptState(TypedDict):
    """条件中断状态"""
    amount: float
    requires_approval: bool
    approved: bool | None
    result: str


def check_amount(state: ConditionalInterruptState) -> dict:
    """检查金额是否需要审批"""
    amount = state["amount"]
    requires_approval = amount > 1000  # 大于1000需要审批
    return {"requires_approval": requires_approval}


def maybe_approve(state: ConditionalInterruptState) -> dict:
    """可能需要审批"""
    if state["requires_approval"]:
        response = interrupt({
            "message": f"金额 {state['amount']} 超过阈值，需要审批",
            "amount": state["amount"]
        })
        return {"approved": response}
    else:
        # 小额自动批准
        return {"approved": True}


def process_transaction(state: ConditionalInterruptState) -> dict:
    """处理交易"""
    if state["approved"]:
        return {"result": f"交易成功: {state['amount']}元"}
    else:
        return {"result": "交易被拒绝"}


def build_conditional_interrupt_graph():
    """构建条件中断图"""
    graph = StateGraph(ConditionalInterruptState)
    
    graph.add_node("check", check_amount)
    graph.add_node("approve", maybe_approve)
    graph.add_node("process", process_transaction)
    
    graph.add_edge(START, "check")
    graph.add_edge("check", "approve")
    graph.add_edge("approve", "process")
    graph.add_edge("process", END)
    
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def demo_conditional_interrupt():
    """演示条件中断"""
    print("\n" + "=" * 60)
    print("演示 4: 条件中断（基于金额）")
    print("=" * 60)
    
    app = build_conditional_interrupt_graph()
    
    # 测试小额交易（无需审批）
    print("\n测试1: 小额交易 (500元) - 无需审批")
    print("-" * 40)
    
    config1 = {"configurable": {"thread_id": "small_amount"}}
    result = app.invoke({
        "amount": 500,
        "requires_approval": False,
        "approved": None,
        "result": ""
    }, config1)
    print(f"  结果: {result['result']}")
    
    # 测试大额交易（需要审批）
    print("\n测试2: 大额交易 (5000元) - 需要审批")
    print("-" * 40)
    
    config2 = {"configurable": {"thread_id": "large_amount"}}
    
    # 第一阶段：执行到中断
    for event in app.stream({
        "amount": 5000,
        "requires_approval": False,
        "approved": None,
        "result": ""
    }, config2):
        print(f"  执行: {event}")
    
    # 获取中断信息
    state = app.get_state(config2)
    if state.tasks:
        for task in state.tasks:
            if hasattr(task, 'interrupts') and task.interrupts:
                print(f"\n  待审批: {task.interrupts[0].value}")
    
    # 批准交易
    print("\n  人工批准...")
    for event in app.stream(Command(resume=True), config2):
        for node, output in event.items():
            if output.get("result"):
                print(f"  结果: {output['result']}")


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    demo_basic_interrupt()
    demo_approval_flow()
    demo_state_editing()
    demo_conditional_interrupt()
    
    print("\n" + "=" * 60)
    print("人机交互教程完成！")
    print("=" * 60)
    print("""
关键要点：
1. interrupt() 函数暂停图执行，等待人工输入
2. 必须使用 checkpointer 才能支持中断（保存状态）
3. thread_id 标识一个执行上下文，允许恢复
4. Command(resume=value) 恢复执行并传入人工输入
5. interrupt_before/interrupt_after 在指定节点前后中断
6. update_state() 允许人工修改状态

使用场景：
- 敏感操作审批（删除、支付、发送）
- 需要额外输入的场景
- Agent 行为纠正
- 数据验证和修正

下一步：学习 04_persistence_memory.py，了解持久化和记忆
""")
