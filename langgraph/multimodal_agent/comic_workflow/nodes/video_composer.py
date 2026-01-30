"""
视频合成节点

将所有场景的图片、音频合成为最终视频。
"""

from langchain_core.messages import AIMessage

from multimodal_agent.utils import run_async

from multimodal_agent.logging_config import get_logger
from multimodal_agent.comic_workflow.state import (
    ComicState,
    WorkflowStatus,
)
from multimodal_agent.services.providers import get_provider
from multimodal_agent.services.providers.base import ProviderType
from multimodal_agent.services.providers.video.base import VideoGenerationConfig

logger = get_logger(__name__)


def video_composer_node(state: ComicState) -> dict:
    """
    视频合成节点
    
    Args:
        state: 当前状态
        
    Returns:
        dict: 状态更新
    """
    scenes = state.get("scenes", [])
    provider_config = state["provider_config"]
    
    # 筛选有效场景（有选中图片的）
    valid_scenes = [s for s in scenes if s.selected_image]
    
    if not valid_scenes:
        logger.error(
            "video_composition_failed_no_valid_scenes",
            workflow_id=state["workflow_id"],
        )
        return {
            "status": WorkflowStatus.FAILED,
            "error": "没有可用于合成的场景",
        }
    
    logger.info(
        "video_composition_started",
        workflow_id=state["workflow_id"],
        scene_count=len(valid_scenes),
    )
    
    try:
        # 计算总时长（每个场景至少 3 秒）
        total_duration = sum(
            max(s.audio.duration, 3.0) if s.audio else 3.0
            for s in valid_scenes
        )
        # 确保至少有 5 秒
        total_duration = max(total_duration, 5.0)
        
        # 这里模拟视频合成
        # 在实际实现中，可能需要：
        # 1. 为每个场景的图片生成动画效果
        # 2. 添加音频轨道
        # 3. 合成最终视频
        
        async def compose_video():
            video_provider = get_provider(
                ProviderType.VIDEO,
                provider_config.video_provider,
            )
            
            # 构建合成提示
            scene_descriptions = "\n".join([
                f"Scene {s.scene_number}: {s.description[:50]}"
                for s in valid_scenes
            ])
            
            config = VideoGenerationConfig(
                model=provider_config.video_model,
                duration=int(min(total_duration, 60)),  # 最长60秒
                resolution="1080p",
            )
            
            result = await video_provider.generate(
                prompt=f"Comic video composition:\n{scene_descriptions}",
                config=config,
            )
            
            return result
        
        result = run_async(compose_video())
        
        logger.info(
            "video_composition_completed",
            workflow_id=state["workflow_id"],
            video_url=result.content,
            duration=total_duration,
        )
        
        ai_message = AIMessage(
            content=f"视频合成完成！\n\n"
                    f"- 视频 URL: {result.content}\n"
                    f"- 包含场景: {len(valid_scenes)} 个\n"
                    f"- 总时长: 约 {total_duration:.1f} 秒"
        )
        
        return {
            "final_video_url": result.content,
            "status": WorkflowStatus.COMPLETED,
            "messages": [ai_message],
        }
        
    except Exception as e:
        logger.error(
            "video_composition_failed",
            workflow_id=state["workflow_id"],
            error=str(e),
        )
        
        return {
            "status": WorkflowStatus.FAILED,
            "error": f"视频合成失败: {str(e)}",
            "messages": [AIMessage(content=f"视频合成失败: {str(e)}")],
        }
