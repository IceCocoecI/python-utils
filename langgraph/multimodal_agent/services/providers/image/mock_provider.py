"""
Mock 图片生成 Provider

用于测试和演示的模拟 Provider。
"""

import asyncio
import uuid
from typing import Any

from multimodal_agent.logging_config import get_logger
from multimodal_agent.services.providers.base import ProviderInfo, ProviderType
from multimodal_agent.services.providers.registry import register_provider
from multimodal_agent.services.providers.image.base import (
    ImageProvider,
    ImageGenerationConfig,
    ImageGenerationResult,
)

logger = get_logger(__name__)


@register_provider(ProviderType.IMAGE, "mock")
class MockImageProvider(ImageProvider):
    """
    Mock 图片生成 Provider
    
    用于测试，返回模拟的图片 URL。
    """
    
    SUPPORTED_MODELS = ["mock-dalle", "mock-sd", "mock-mj"]
    
    def __init__(self, delay: float = 1.0):
        super().__init__()
        self.delay = delay
    
    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="mock",
            provider_type=ProviderType.IMAGE,
            description="Mock 图片生成 Provider（测试用）",
            models=self.SUPPORTED_MODELS,
        )
    
    async def generate(
        self,
        prompt: str,
        config: ImageGenerationConfig | None = None,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """模拟生成图片"""
        config = config or ImageGenerationConfig()
        model = config.model or "mock-dalle"
        n = config.n
        
        logger.info(
            "mock_image_generation_started",
            model=model,
            prompt_length=len(prompt),
            n=n,
        )
        
        # 模拟延迟
        await asyncio.sleep(self.delay * n)
        
        # 生成模拟 URL
        request_id = str(uuid.uuid4())
        urls = [
            f"https://cdn.mock-images.com/{request_id}/{i+1}.png"
            for i in range(n)
        ]
        
        content = urls[0] if n == 1 else urls
        
        result = ImageGenerationResult(
            request_id=request_id,
            content=content,
            provider="mock",
            model=model,
            revised_prompt=self._revise_prompt(prompt),
            size=config.size,
            style=config.style,
            metadata={
                "quality": config.quality,
                "count": n,
                "negative_prompt": config.negative_prompt,
            },
        )
        
        logger.info(
            "mock_image_generation_completed",
            request_id=result.request_id,
            count=n,
        )
        
        return result
    
    def _revise_prompt(self, prompt: str) -> str:
        """优化提示词"""
        enhancements = [
            "highly detailed",
            "professional quality", 
            "8k resolution",
        ]
        revised = f"{prompt}, {', '.join(enhancements)}"
        return revised if len(revised) <= 1000 else prompt
