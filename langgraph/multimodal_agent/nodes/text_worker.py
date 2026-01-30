"""
文本生成 Worker 节点

负责调用文本生成服务并处理结果。
"""

from langchain_core.messages import AIMessage

from multimodal_agent.utils import run_async

from multimodal_agent.logging_config import get_logger
from multimodal_agent.services.text_service import TextGenerationService, ServiceException
from multimodal_agent.state import AgentState, TaskType, ContentResult, ErrorInfo

logger = get_logger(__name__)


def text_worker_node(state: AgentState) -> dict:
    """
    文本生成 Worker 节点
    
    调用文本生成服务处理用户请求。
    
    Args:
        state: 当前 Agent 状态
        
    Returns:
        dict: 状态更新
    """
    user_input = state["user_input"]
    
    logger.info(
        "text_worker_started",
        user_input_length=len(user_input),
    )
    
    try:
        # 创建服务实例
        service = TextGenerationService()
        
        # 使用安全的异步运行方法
        response = run_async(_generate_text(service, user_input))
        
        # 构建生成结果
        content_result = ContentResult(
            task_type=TaskType.TEXT,
            content=response.content,
            metadata={
                "request_id": response.request_id,
                "model": response.model,
                "usage": response.usage,
                "finish_reason": response.finish_reason,
            },
        )
        
        # 构建 AI 消息
        ai_message = AIMessage(
            content=f"文本生成完成！\n\n{response.content}"
        )
        
        logger.info(
            "text_worker_completed",
            request_id=response.request_id,
            content_length=len(response.content),
        )
        
        return {
            "messages": [ai_message],
            "generated_content": content_result,
            "is_completed": True,
        }
        
    except ServiceException as e:
        logger.error(
            "text_worker_service_error",
            error=str(e),
            status_code=e.status_code,
        )
        
        error_info = ErrorInfo(
            error_type="TextServiceError",
            message=str(e),
            details={
                "status_code": e.status_code,
                "response_body": e.response_body,
            },
            node_name="text_worker",
        )
        
        ai_message = AIMessage(
            content=f"文本生成失败：{e.message}"
        )
        
        return {
            "messages": [ai_message],
            "error_message": error_info,
            "is_completed": True,
        }
        
    except Exception as e:
        logger.error(
            "text_worker_unexpected_error",
            error=str(e),
            error_type=type(e).__name__,
        )
        
        error_info = ErrorInfo(
            error_type=type(e).__name__,
            message=f"文本生成时发生意外错误: {str(e)}",
            node_name="text_worker",
        )
        
        ai_message = AIMessage(
            content=f"文本生成失败：发生意外错误"
        )
        
        return {
            "messages": [ai_message],
            "error_message": error_info,
            "is_completed": True,
        }


async def _generate_text(service: TextGenerationService, prompt: str):
    """
    异步生成文本
    
    Args:
        service: 文本生成服务
        prompt: 用户提示
        
    Returns:
        TextGenerationResponse: 生成结果
    """
    try:
        return await service.generate(prompt=prompt)
    finally:
        await service.close()
