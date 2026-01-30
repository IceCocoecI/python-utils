"""
视频生成 Worker 节点

负责调用视频生成服务并处理结果。
"""

from langchain_core.messages import AIMessage

from multimodal_agent.utils import run_async

from multimodal_agent.logging_config import get_logger
from multimodal_agent.services.video_service import VideoGenerationService, ServiceException
from multimodal_agent.state import AgentState, TaskType, ContentResult, ErrorInfo

logger = get_logger(__name__)


def video_worker_node(state: AgentState) -> dict:
    """
    视频生成 Worker 节点
    
    调用视频生成服务处理用户请求。
    
    Args:
        state: 当前 Agent 状态
        
    Returns:
        dict: 状态更新
    """
    user_input = state["user_input"]
    
    logger.info(
        "video_worker_started",
        user_input_length=len(user_input),
    )
    
    try:
        # 创建服务实例
        service = VideoGenerationService()
        
        # 使用安全的异步运行方法
        response = run_async(_generate_video(service, user_input))
        
        # 构建生成结果
        content_result = ContentResult(
            task_type=TaskType.VIDEO,
            content=response.video_url,
            metadata={
                "request_id": response.request_id,
                "thumbnail_url": response.thumbnail_url,
                "duration": response.duration,
                "resolution": response.resolution,
                "status": response.status,
            },
        )
        
        # 构建 AI 消息
        ai_message = AIMessage(
            content=f"视频生成完成！\n\n"
                    f"- 视频 URL: {response.video_url}\n"
                    f"- 缩略图: {response.thumbnail_url}\n"
                    f"- 时长: {response.duration} 秒\n"
                    f"- 分辨率: {response.resolution}\n"
                    f"- 状态: {response.status}"
        )
        
        logger.info(
            "video_worker_completed",
            request_id=response.request_id,
            video_url=response.video_url,
            duration=response.duration,
        )
        
        return {
            "messages": [ai_message],
            "generated_content": content_result,
            "is_completed": True,
        }
        
    except ServiceException as e:
        logger.error(
            "video_worker_service_error",
            error=str(e),
            status_code=e.status_code,
        )
        
        error_info = ErrorInfo(
            error_type="VideoServiceError",
            message=str(e),
            details={
                "status_code": e.status_code,
                "response_body": e.response_body,
            },
            node_name="video_worker",
        )
        
        ai_message = AIMessage(
            content=f"视频生成失败：{e.message}"
        )
        
        return {
            "messages": [ai_message],
            "error_message": error_info,
            "is_completed": True,
        }
        
    except Exception as e:
        logger.error(
            "video_worker_unexpected_error",
            error=str(e),
            error_type=type(e).__name__,
        )
        
        error_info = ErrorInfo(
            error_type=type(e).__name__,
            message=f"视频生成时发生意外错误: {str(e)}",
            node_name="video_worker",
        )
        
        ai_message = AIMessage(
            content=f"视频生成失败：发生意外错误"
        )
        
        return {
            "messages": [ai_message],
            "error_message": error_info,
            "is_completed": True,
        }


async def _generate_video(service: VideoGenerationService, prompt: str):
    """
    异步生成视频
    
    Args:
        service: 视频生成服务
        prompt: 用户提示
        
    Returns:
        VideoGenerationResponse: 生成结果
    """
    try:
        return await service.generate(
            prompt=prompt,
            duration=5,
            resolution="1080p",
            fps=30,
            style="cinematic",
        )
    finally:
        await service.close()
