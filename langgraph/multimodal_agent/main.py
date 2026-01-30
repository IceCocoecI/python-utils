"""
多模态内容生成 Agent - 主入口

提供命令行界面和程序化调用接口。
"""

import sys
from pathlib import Path

# 添加父目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Any

from langchain_core.messages import HumanMessage

from multimodal_agent.config import get_config
from multimodal_agent.logging_config import setup_logging, get_logger, LogContext
from multimodal_agent.state import AgentState, create_initial_state, TaskType
from multimodal_agent.graph import create_multimodal_agent, print_graph_structure

# 初始化日志
setup_logging()
logger = get_logger(__name__)


def run_agent(user_input: str) -> AgentState:
    """
    运行多模态内容生成 Agent
    
    Args:
        user_input: 用户输入
        
    Returns:
        AgentState: 执行完成后的状态
    """
    # 绑定请求上下文
    import uuid
    request_id = str(uuid.uuid4())[:8]
    LogContext.bind(request_id=request_id)
    
    logger.info(
        "agent_invocation_started",
        user_input_length=len(user_input),
        user_input_preview=user_input[:100],
    )
    
    try:
        # 创建 Agent
        agent = create_multimodal_agent()
        
        # 创建初始状态
        initial_state = create_initial_state(user_input)
        
        # 添加用户消息到对话历史
        initial_state["messages"] = [HumanMessage(content=user_input)]
        
        # 执行 Agent
        final_state = agent.invoke(initial_state)
        
        logger.info(
            "agent_invocation_completed",
            task_type=final_state.get("task_type"),
            has_error=final_state.get("error_message") is not None,
            is_completed=final_state.get("is_completed"),
        )
        
        return final_state
        
    except Exception as e:
        logger.error(
            "agent_invocation_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        raise
        
    finally:
        LogContext.clear()


def format_result(state: AgentState) -> str:
    """
    格式化输出结果
    
    Args:
        state: Agent 最终状态
        
    Returns:
        str: 格式化的结果字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("执行结果")
    lines.append("=" * 60)
    
    # 任务类型
    task_type = state.get("task_type")
    if task_type:
        task_type_names = {
            TaskType.TEXT: "文本生成",
            TaskType.IMAGE: "图片生成",
            TaskType.VIDEO: "视频生成",
            TaskType.UNKNOWN: "未知",
        }
        lines.append(f"\n任务类型: {task_type_names.get(task_type, str(task_type))}")
    
    # 生成内容
    content = state.get("generated_content")
    if content:
        lines.append(f"\n生成内容:")
        lines.append(f"  - 类型: {content.task_type}")
        lines.append(f"  - 内容/URL: {content.content}")
        if content.metadata:
            lines.append(f"  - 元数据: {content.metadata}")
        lines.append(f"  - 创建时间: {content.created_at}")
    
    # 错误信息
    error = state.get("error_message")
    if error:
        lines.append(f"\n错误信息:")
        lines.append(f"  - 类型: {error.error_type}")
        lines.append(f"  - 消息: {error.message}")
        lines.append(f"  - 节点: {error.node_name}")
    
    # 对话历史
    messages = state.get("messages", [])
    if messages:
        lines.append(f"\n对话历史 ({len(messages)} 条消息):")
        for i, msg in enumerate(messages):
            role = msg.__class__.__name__.replace("Message", "")
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            lines.append(f"  [{i+1}] {role}: {content_preview}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


def interactive_mode():
    """
    交互模式 - 持续接收用户输入
    """
    print("\n" + "=" * 60)
    print("多模态内容生成 Agent - 交互模式")
    print("=" * 60)
    print("\n支持的任务类型:")
    print("  1. 文本生成 - 写文章、故事、代码等")
    print("  2. 图片生成 - 生成图片、绘画等")
    print("  3. 视频生成 - 生成视频、动画等")
    print("\n输入 'quit' 或 'exit' 退出")
    print("输入 'graph' 查看工作流结构")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n请输入您的需求: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ("quit", "exit", "q"):
                print("\n再见！")
                break
                
            if user_input.lower() == "graph":
                print(print_graph_structure(None))
                continue
            
            # 运行 Agent
            final_state = run_agent(user_input)
            
            # 格式化并打印结果
            print(format_result(final_state))
            
        except KeyboardInterrupt:
            print("\n\n已中断，再见！")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            logger.exception("interactive_mode_error")


def run_examples():
    """
    运行示例请求
    """
    examples = [
        "请写一篇关于人工智能在医疗领域应用的文章",
        "画一张夕阳下的海边风景画",
        "生成一个展示产品功能的10秒宣传视频",
        "今天天气怎么样？",  # 这个应该触发文本生成
    ]
    
    print("\n" + "=" * 60)
    print("运行示例请求")
    print("=" * 60)
    
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"示例 {i}: {example}")
        print("=" * 60)
        
        try:
            final_state = run_agent(example)
            print(format_result(final_state))
        except Exception as e:
            print(f"执行失败: {e}")
        
        print("-" * 60)


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="多模态内容生成 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py --interactive          # 交互模式
  python main.py --examples             # 运行示例
  python main.py --input "写一篇文章"    # 单次执行
  python main.py --graph                # 显示工作流图
        """
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="启动交互模式"
    )
    parser.add_argument(
        "-e", "--examples",
        action="store_true",
        help="运行示例请求"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="直接处理指定的输入"
    )
    parser.add_argument(
        "-g", "--graph",
        action="store_true",
        help="显示工作流图结构"
    )
    
    args = parser.parse_args()
    
    if args.graph:
        print(print_graph_structure(None))
        return
    
    if args.examples:
        run_examples()
        return
    
    if args.input:
        final_state = run_agent(args.input)
        print(format_result(final_state))
        return
    
    if args.interactive:
        interactive_mode()
        return
    
    # 默认显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()
