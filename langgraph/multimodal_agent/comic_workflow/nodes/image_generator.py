"""
图片生成节点

为每个场景生成多张候选图片。
"""

import asyncio

from langchain_core.messages import AIMessage

from multimodal_agent.utils import run_async

from multimodal_agent.logging_config import get_logger
from multimodal_agent.comic_workflow.state import (
    ComicState, 
    Scene, 
    SceneImage,
    WorkflowStatus,
    InteractionPoint,
)
from multimodal_agent.services.providers import get_provider
from multimodal_agent.services.providers.base import ProviderType
from multimodal_agent.services.providers.image.base import ImageGenerationConfig

logger = get_logger(__name__)

# 每个场景生成的候选图片数量
IMAGES_PER_SCENE = 4


def build_image_prompt(scene: Scene, style_preferences: dict) -> str:
    """
    构建图片生成提示词
    
    Args:
        scene: 场景
        style_preferences: 风格偏好
        
    Returns:
        str: 图片提示词
    """
    style = style_preferences.get("art_style", "anime illustration")
    
    prompt_parts = [
        scene.description,
        f"Art style: {style}",
        "high quality",
        "detailed",
    ]
    
    if scene.dialogue:
        # 可以添加对白相关的情感描述
        prompt_parts.append("emotional scene")
    
    return ", ".join(prompt_parts)


async def generate_scene_images(
    scene: Scene,
    provider_name: str,
    model: str | None,
    style_preferences: dict,
    count: int = IMAGES_PER_SCENE,
) -> list[SceneImage]:
    """
    为单个场景生成多张候选图片
    
    Args:
        scene: 场景
        provider_name: Provider 名称
        model: 模型名称
        style_preferences: 风格偏好
        count: 生成数量
        
    Returns:
        list[SceneImage]: 图片列表
    """
    image_provider = get_provider(ProviderType.IMAGE, provider_name)
    
    prompt = build_image_prompt(scene, style_preferences)
    
    config = ImageGenerationConfig(
        model=model,
        n=count,
        size="1024x1024",
        style=style_preferences.get("image_style", "vivid"),
    )
    
    result = await image_provider.generate(prompt, config)
    
    # 处理返回的 URL（可能是单个或列表）
    urls = result.content if isinstance(result.content, list) else [result.content]
    
    images = []
    for i, url in enumerate(urls):
        image = SceneImage(
            image_id=f"{scene.scene_id}_img_{i+1}",
            url=url,
            prompt=prompt,
            is_selected=False,
            metadata={
                "provider": provider_name,
                "model": result.model,
                "revised_prompt": result.revised_prompt,
            },
        )
        images.append(image)
    
    return images


def image_generator_node(state: ComicState) -> dict:
    """
    图片生成节点
    
    为所有场景生成候选图片。
    
    Args:
        state: 当前状态
        
    Returns:
        dict: 状态更新
    """
    scenes = state.get("scenes", [])
    provider_config = state["provider_config"]
    style_preferences = state.get("style_preferences", {})
    interaction_mode = state.get("interaction_mode", "interactive")
    
    if not scenes:
        logger.error(
            "image_generation_failed_no_scenes",
            workflow_id=state["workflow_id"],
        )
        return {
            "status": WorkflowStatus.FAILED,
            "error": "没有可用的场景",
        }
    
    logger.info(
        "image_generation_started",
        workflow_id=state["workflow_id"],
        scene_count=len(scenes),
        images_per_scene=IMAGES_PER_SCENE,
    )
    
    try:
        # 异步生成所有场景的图片
        async def generate_all():
            tasks = []
            for scene in scenes:
                task = generate_scene_images(
                    scene=scene,
                    provider_name=provider_config.image_provider,
                    model=provider_config.image_model,
                    style_preferences=style_preferences,
                )
                tasks.append(task)
            return await asyncio.gather(*tasks)
        
        all_images = run_async(generate_all())
        
        # 更新场景的图片候选
        updated_scenes = []
        for scene, images in zip(scenes, all_images):
            scene.image_candidates = images
            
            # 如果是自动模式，自动选择第一张
            if interaction_mode == "auto":
                scene.selected_image = images[0] if images else None
                if scene.selected_image:
                    scene.selected_image.is_selected = True
            
            updated_scenes.append(scene)
        
        total_images = sum(len(s.image_candidates) for s in updated_scenes)
        
        logger.info(
            "image_generation_completed",
            workflow_id=state["workflow_id"],
            total_images=total_images,
        )
        
        # 构建消息
        message_content = f"图片生成完成！共生成 {total_images} 张图片（{len(scenes)} 个场景，每个 {IMAGES_PER_SCENE} 张）。\n"
        
        if interaction_mode == "interactive":
            message_content += "\n请为每个场景选择一张图片。"
        
        ai_message = AIMessage(content=message_content)
        
        # 根据交互模式决定状态
        if interaction_mode == "interactive":
            # 交互模式：等待用户选择
            return {
                "scenes": updated_scenes,
                "status": WorkflowStatus.WAITING_INPUT,
                "current_interaction": InteractionPoint.IMAGE_SELECTION,
                "pending_selection": {
                    "type": "image_selection",
                    "scenes": [
                        {
                            "scene_id": s.scene_id,
                            "scene_number": s.scene_number,
                            "description": s.description[:100],
                            "candidates": [
                                {"image_id": img.image_id, "url": img.url}
                                for img in s.image_candidates
                            ],
                        }
                        for s in updated_scenes
                    ],
                },
                "messages": [ai_message],
            }
        else:
            # 自动模式：直接继续
            return {
                "scenes": updated_scenes,
                "messages": [ai_message],
            }
        
    except Exception as e:
        logger.error(
            "image_generation_failed",
            workflow_id=state["workflow_id"],
            error=str(e),
        )
        
        return {
            "status": WorkflowStatus.FAILED,
            "error": f"图片生成失败: {str(e)}",
            "messages": [AIMessage(content=f"图片生成失败: {str(e)}")],
        }
