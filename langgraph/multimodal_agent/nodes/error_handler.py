"""
错误处理节点

统一处理 Agent 执行过程中的错误。
"""

from langchain_core.messages import AIMessage

from multimodal_agent.logging_config import get_logger
from multimodal_agent.state import AgentState, TaskType, ErrorInfo

logger = get_logger(__name__)


def error_handler_node(state: AgentState) -> dict:
    """
    错误处理节点
    
    处理执行过程中的错误，生成友好的错误消息。
    
    Args:
        state: 当前 Agent 状态
        
    Returns:
        dict: 状态更新
    """
    error = state.get("error_message")
    task_type = state.get("task_type")
    user_input = state.get("user_input", "")
    
    logger.info(
        "error_handler_processing",
        has_error=error is not None,
        task_type=task_type.value if task_type else None,
    )
    
    # 如果已有错误信息
    if error:
        logger.error(
            "handling_error",
            error_type=error.error_type,
            message=error.message,
            node_name=error.node_name,
        )
        
        ai_message = AIMessage(
            content=f"抱歉，处理您的请求时出现了问题。\n\n"
                    f"错误类型: {error.error_type}\n"
                    f"错误信息: {error.message}\n\n"
                    f"请稍后重试，或尝试换一种方式描述您的需求。"
        )
        
        return {
            "messages": [ai_message],
            "is_completed": True,
        }
    
    # 未知任务类型
    if task_type == TaskType.UNKNOWN or task_type is None:
        logger.warning(
            "unknown_task_type",
            user_input_preview=user_input[:100] if user_input else "",
        )
        
        error_info = ErrorInfo(
            error_type="UnknownTaskType",
            message="无法识别您的请求类型",
            details={
                "user_input_preview": user_input[:200] if user_input else "",
            },
            node_name="error_handler",
        )
        
        ai_message = AIMessage(
            content="抱歉，我无法理解您的请求。\n\n"
                    "我支持以下类型的内容生成：\n"
                    "1. **文本生成**: 写文章、写故事、写代码、总结等\n"
                    "2. **图片生成**: 生成图片、画图、设计等\n"
                    "3. **视频生成**: 制作视频、动画等\n\n"
                    "请尝试更明确地描述您想要的内容类型。\n\n"
                    "示例：\n"
                    "- \"请写一篇关于人工智能的文章\"\n"
                    "- \"画一张日落的风景画\"\n"
                    "- \"生成一个展示产品的短视频\""
        )
        
        return {
            "messages": [ai_message],
            "error_message": error_info,
            "is_completed": True,
        }
    
    # 通用错误处理
    logger.warning(
        "generic_error_handling",
    )
    
    ai_message = AIMessage(
        content="处理过程中出现了未知问题，请稍后重试。"
    )
    
    return {
        "messages": [ai_message],
        "error_message": ErrorInfo(
            error_type="GenericError",
            message="处理过程中出现未知问题",
            node_name="error_handler",
        ),
        "is_completed": True,
    }
