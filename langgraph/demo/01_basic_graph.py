"""
LangGraph 基础教程 01: 基础图构建
================================================

本示例演示 LangGraph 的核心概念：
1. StateGraph - 状态图
2. State - 状态定义
3. Node - 节点函数
4. Edge - 边（连接）
5. START/END - 特殊节点

学习目标：
- 理解 LangGraph 的图结构
- 掌握状态定义和更新方式
- 学会构建简单的线性流程
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END


# =============================================================================
# 第一部分：状态定义
# =============================================================================

class SimpleState(TypedDict):
    """
    状态是图中所有节点共享的数据结构。
    
    设计原则：
    1. 使用 TypedDict 保证类型安全
    2. 只包含必要的字段
    3. 每个字段都应该有明确的用途
    """
    # 输入数据
    input_text: str
    
    # 处理过程中的数据
    processed_text: str
    
    # 最终输出
    output: str
    
    # 执行追踪（可选，用于调试）
    steps: list[str]


# =============================================================================
# 第二部分：节点函数
# =============================================================================

def preprocess_node(state: SimpleState) -> dict:
    """
    预处理节点：清理和标准化输入
    
    节点函数的规范：
    - 输入：完整的当前状态
    - 输出：状态更新字典（只包含需要更新的字段）
    
    重要：返回的是"增量更新"，不是完整状态
    """
    input_text = state["input_text"]
    
    # 执行预处理：去除空白、转小写
    cleaned = input_text.strip().lower()
    
    # 返回状态更新
    return {
        "processed_text": cleaned,
        "steps": state.get("steps", []) + ["preprocess"]
    }


def transform_node(state: SimpleState) -> dict:
    """
    转换节点：对数据进行核心处理
    """
    text = state["processed_text"]
    
    # 简单转换：反转字符串并添加标记
    transformed = f"[处理结果] {text[::-1]}"
    
    return {
        "processed_text": transformed,
        "steps": state["steps"] + ["transform"]
    }


def output_node(state: SimpleState) -> dict:
    """
    输出节点：生成最终结果
    """
    result = state["processed_text"]
    
    return {
        "output": f"最终输出: {result}",
        "steps": state["steps"] + ["output"]
    }


# =============================================================================
# 第三部分：构建图
# =============================================================================

def build_simple_graph():
    """
    构建一个简单的线性处理图
    
    图结构：
        START → preprocess → transform → output → END
    """
    # 1. 创建状态图，指定状态类型
    graph = StateGraph(SimpleState)
    
    # 2. 添加节点
    # add_node(节点名称, 节点函数)
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("transform", transform_node)
    graph.add_node("output", output_node)
    
    # 3. 添加边，定义执行顺序
    # START 是特殊的入口节点
    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "transform")
    graph.add_edge("transform", "output")
    # END 是特殊的出口节点
    graph.add_edge("output", END)
    
    # 4. 编译图
    # 编译后的图是可执行的应用
    app = graph.compile()
    
    return app


# =============================================================================
# 第四部分：运行和测试
# =============================================================================

def demo_basic_execution():
    """演示基本执行流程"""
    print("=" * 60)
    print("演示 1: 基本执行")
    print("=" * 60)
    
    # 构建图
    app = build_simple_graph()
    
    # 准备输入状态
    initial_state = {
        "input_text": "  Hello LangGraph World!  ",
        "processed_text": "",
        "output": "",
        "steps": []
    }
    
    # 执行图（同步调用）
    result = app.invoke(initial_state)
    
    # 打印结果
    print(f"\n输入: '{initial_state['input_text']}'")
    print(f"执行步骤: {result['steps']}")
    print(f"输出: {result['output']}")
    
    return result


def demo_stream_execution():
    """演示流式执行，观察每一步的输出"""
    print("\n" + "=" * 60)
    print("演示 2: 流式执行（观察每一步）")
    print("=" * 60)
    
    app = build_simple_graph()
    
    initial_state = {
        "input_text": "Stream Demo",
        "processed_text": "",
        "output": "",
        "steps": []
    }
    
    # 使用 stream() 观察每一步
    print("\n执行过程：")
    for step_num, state_update in enumerate(app.stream(initial_state)):
        print(f"\n步骤 {step_num + 1}:")
        for node_name, node_output in state_update.items():
            print(f"  节点: {node_name}")
            print(f"  输出: {node_output}")


def demo_graph_visualization():
    """演示图的可视化（Mermaid 格式）"""
    print("\n" + "=" * 60)
    print("演示 3: 图结构可视化")
    print("=" * 60)
    
    app = build_simple_graph()
    
    # 获取 Mermaid 格式的图描述
    try:
        mermaid = app.get_graph().draw_mermaid()
        print("\nMermaid 图定义（可在 Mermaid 编辑器中渲染）：")
        print(mermaid)
    except Exception as e:
        print(f"可视化需要额外依赖: {e}")


# =============================================================================
# 进阶示例：使用 Reducer 的状态
# =============================================================================

from operator import add

class AccumulatingState(TypedDict):
    """
    使用 Annotated 定义带有 Reducer 的状态
    
    Reducer 决定了状态如何更新：
    - 默认行为：新值覆盖旧值
    - add reducer：新值追加到列表
    - 自定义 reducer：任意更新逻辑
    """
    # 普通字段：新值覆盖旧值
    current_value: str
    
    # 带 reducer 的字段：新值追加到列表
    # Annotated[类型, reducer函数]
    history: Annotated[list[str], add]


def step_a(state: AccumulatingState) -> dict:
    return {
        "current_value": "A",
        "history": ["执行了步骤A"]  # 会追加，不是覆盖
    }


def step_b(state: AccumulatingState) -> dict:
    return {
        "current_value": "B",
        "history": ["执行了步骤B"]  # 会追加到 history
    }


def demo_reducer():
    """演示 Reducer 的效果"""
    print("\n" + "=" * 60)
    print("演示 4: Reducer（状态累积）")
    print("=" * 60)
    
    # 构建图
    graph = StateGraph(AccumulatingState)
    graph.add_node("step_a", step_a)
    graph.add_node("step_b", step_b)
    graph.add_edge(START, "step_a")
    graph.add_edge("step_a", "step_b")
    graph.add_edge("step_b", END)
    
    app = graph.compile()
    
    # 执行
    initial = {"current_value": "", "history": []}
    result = app.invoke(initial)
    
    print(f"\n最终 current_value: {result['current_value']}")  # 只有最后一个值
    print(f"history（累积）: {result['history']}")  # 所有历史记录


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    # 运行所有演示
    demo_basic_execution()
    demo_stream_execution()
    demo_graph_visualization()
    demo_reducer()
    
    print("\n" + "=" * 60)
    print("基础图教程完成！")
    print("=" * 60)
    print("""
关键要点：
1. StateGraph 是核心容器，定义状态类型
2. 节点是普通函数，接收状态返回更新
3. 边定义执行顺序，START/END 是特殊节点
4. compile() 后才能执行
5. invoke() 同步执行，stream() 流式执行
6. Reducer 控制状态更新方式（覆盖 vs 追加）

下一步：学习 02_chatbot_with_tools.py，了解工具调用
""")
