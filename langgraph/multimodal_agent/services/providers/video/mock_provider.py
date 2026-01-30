"""
Mock 视频生成 Provider

用于测试和演示的模拟 Provider。
"""

import asyncio
import uuid
from typing import Any

from multimodal_agent.logging_config import get_logger
from multimodal_agent.services.providers.base import ProviderInfo, ProviderType
from multimodal_agent.services.providers.registry import register_provider
from multimodal_agent.services.providers.video.base import (
    VideoProvider,
    VideoGenerationConfig,
    VideoGenerationResult,
)

logger = get_logger(__name__)


@register_provider(ProviderType.VIDEO, "mock", is_default=True)
class MockVideoProvider(VideoProvider):
    """
    Mock 视频生成 Provider
    
    用于测试，返回模拟的视频 URL。
    """
    
    SUPPORTED_MODELS = ["mock-runway", "mock-sora", "mock-pika"]
    
    def __init__(self, delay: float = 2.0):
        super().__init__()
        self.delay = delay
    
    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="mock",
            provider_type=ProviderType.VIDEO,
            description="Mock 视频生成 Provider（测试用）",
            models=self.SUPPORTED_MODELS,
        )
    
    async def generate(
        self,
        prompt: str,
        config: VideoGenerationConfig | None = None,
        **kwargs: Any,
    ) -> VideoGenerationResult:
        """模拟生成视频"""
        config = config or VideoGenerationConfig()
        model = config.model or "mock-runway"
        
        logger.info(
            "mock_video_generation_started",
            model=model,
            prompt_length=len(prompt),
            duration=config.duration,
        )
        
        # 模拟延迟（视频生成通常较慢）
        await asyncio.sleep(self.delay)
        
        # 生成模拟 URL
        request_id = str(uuid.uuid4())
        
        result = VideoGenerationResult(
            request_id=request_id,
            content=f"https://cdn.mock-videos.com/{request_id}.mp4",
            provider="mock",
            model=model,
            thumbnail_url=f"https://cdn.mock-videos.com/{request_id}_thumb.jpg",
            duration=config.duration,
            resolution=config.resolution,
            status="completed",
            metadata={
                "fps": config.fps,
                "style": config.style,
                "aspect_ratio": config.aspect_ratio,
            },
        )
        
        logger.info(
            "mock_video_generation_completed",
            request_id=result.request_id,
            duration=result.duration,
        )
        
        return result
    
    async def generate_from_image(
        self,
        image_url: str,
        prompt: str | None = None,
        config: VideoGenerationConfig | None = None,
        **kwargs: Any,
    ) -> VideoGenerationResult:
        """从图片生成视频"""
        config = config or VideoGenerationConfig()
        model = config.model or "mock-runway"
        
        logger.info(
            "mock_image_to_video_started",
            model=model,
            image_url=image_url[:50],
        )
        
        await asyncio.sleep(self.delay)
        
        request_id = str(uuid.uuid4())
        
        result = VideoGenerationResult(
            request_id=request_id,
            content=f"https://cdn.mock-videos.com/{request_id}_i2v.mp4",
            provider="mock",
            model=model,
            thumbnail_url=image_url,  # 使用原图作为缩略图
            duration=config.duration,
            resolution=config.resolution,
            status="completed",
            metadata={
                "source_image": image_url,
                "motion_prompt": prompt,
            },
        )
        
        logger.info(
            "mock_image_to_video_completed",
            request_id=result.request_id,
        )
        
        return result
