"""
音频生成节点

为每个场景生成配音（对白和旁白）。
"""

import asyncio

from langchain_core.messages import AIMessage

from multimodal_agent.utils import run_async

from multimodal_agent.logging_config import get_logger
from multimodal_agent.comic_workflow.state import (
    ComicState,
    Scene,
    SceneAudio,
    WorkflowStatus,
)
from multimodal_agent.services.providers import get_provider
from multimodal_agent.services.providers.base import ProviderType
from multimodal_agent.services.providers.audio.base import AudioGenerationConfig

logger = get_logger(__name__)


async def generate_scene_audio(
    scene: Scene,
    provider_name: str,
    model: str | None,
) -> SceneAudio | None:
    """
    为单个场景生成音频
    
    Args:
        scene: 场景
        provider_name: Provider 名称
        model: 模型名称
        
    Returns:
        SceneAudio | None: 音频信息
    """
    # 合并对白和旁白作为配音文本
    text_parts = []
    
    if scene.narration:
        text_parts.append(scene.narration)
    
    if scene.dialogue:
        text_parts.append(scene.dialogue)
    
    if not text_parts:
        # 如果没有对白和旁白，跳过
        return None
    
    text = "\n".join(text_parts)
    
    audio_provider = get_provider(ProviderType.AUDIO, provider_name)
    
    config = AudioGenerationConfig(
        model=model,
        voice="narrator",  # 默认使用旁白声音
        language="zh-CN",
        speed=1.0,
    )
    
    result = await audio_provider.generate(text, config)
    
    return SceneAudio(
        audio_id=f"{scene.scene_id}_audio",
        url=result.content,
        text=text,
        voice=result.voice,
        duration=result.duration,
    )


def audio_generator_node(state: ComicState) -> dict:
    """
    音频生成节点
    
    Args:
        state: 当前状态
        
    Returns:
        dict: 状态更新
    """
    scenes = state.get("scenes", [])
    provider_config = state["provider_config"]
    
    # 只处理已选择图片的场景
    scenes_to_process = [s for s in scenes if s.selected_image]
    
    if not scenes_to_process:
        logger.warning(
            "audio_generation_skipped_no_selected_images",
            workflow_id=state["workflow_id"],
        )
        return {
            "messages": [AIMessage(content="没有需要生成配音的场景。")],
        }
    
    logger.info(
        "audio_generation_started",
        workflow_id=state["workflow_id"],
        scene_count=len(scenes_to_process),
    )
    
    try:
        async def generate_all():
            tasks = []
            for scene in scenes_to_process:
                task = generate_scene_audio(
                    scene=scene,
                    provider_name=provider_config.audio_provider,
                    model=provider_config.audio_model,
                )
                tasks.append(task)
            return await asyncio.gather(*tasks)
        
        all_audios = run_async(generate_all())
        
        # 更新场景的音频
        # 创建场景ID到音频的映射
        scene_audio_map = {}
        for scene, audio in zip(scenes_to_process, all_audios):
            if audio:
                scene_audio_map[scene.scene_id] = audio
        
        # 更新所有场景
        updated_scenes = []
        audio_count = 0
        for scene in scenes:
            if scene.scene_id in scene_audio_map:
                scene.audio = scene_audio_map[scene.scene_id]
                audio_count += 1
            updated_scenes.append(scene)
        
        # 按场景序号排序
        updated_scenes.sort(key=lambda s: s.scene_number)
        
        total_duration = sum(s.audio.duration for s in updated_scenes if s.audio)
        
        logger.info(
            "audio_generation_completed",
            workflow_id=state["workflow_id"],
            audio_count=audio_count,
            total_duration=total_duration,
        )
        
        ai_message = AIMessage(
            content=f"配音生成完成！共生成 {audio_count} 个音频，总时长约 {total_duration:.1f} 秒。"
        )
        
        return {
            "scenes": updated_scenes,
            "messages": [ai_message],
        }
        
    except Exception as e:
        logger.error(
            "audio_generation_failed",
            workflow_id=state["workflow_id"],
            error=str(e),
        )
        
        return {
            "status": WorkflowStatus.FAILED,
            "error": f"音频生成失败: {str(e)}",
            "messages": [AIMessage(content=f"配音生成失败: {str(e)}")],
        }
