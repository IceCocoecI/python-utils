"""
视频生成 Provider 基类
"""

from abc import abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

from multimodal_agent.services.providers.base import (
    BaseProvider,
    ProviderInfo,
    ProviderType,
    GenerationConfig,
    GenerationResult,
)


class VideoGenerationConfig(GenerationConfig):
    """视频生成配置"""
    
    duration: int = Field(default=5, ge=1, le=60, description="视频时长（秒）")
    resolution: Literal["720p", "1080p", "4k"] = Field(
        default="1080p",
        description="视频分辨率"
    )
    fps: int = Field(default=30, ge=15, le=60, description="帧率")
    style: str = Field(default="cinematic", description="视频风格")
    aspect_ratio: str = Field(default="16:9", description="宽高比")


class VideoGenerationResult(GenerationResult):
    """视频生成结果"""
    
    content: str = Field(description="视频 URL")
    thumbnail_url: str | None = Field(default=None, description="缩略图 URL")
    duration: int = Field(description="视频时长（秒）")
    resolution: str = Field(description="视频分辨率")
    status: str = Field(default="completed", description="生成状态")


class VideoProvider(BaseProvider[VideoGenerationResult]):
    """
    视频生成 Provider 抽象基类
    
    所有视频生成 Provider 都必须继承此类。
    """
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.VIDEO
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: VideoGenerationConfig | None = None,
        **kwargs: Any,
    ) -> VideoGenerationResult:
        """生成视频"""
        pass
    
    async def generate_batch(
        self,
        prompts: list[str],
        config: VideoGenerationConfig | None = None,
        **kwargs: Any,
    ) -> list[VideoGenerationResult]:
        """批量生成视频"""
        results = []
        for prompt in prompts:
            result = await self.generate(prompt, config, **kwargs)
            results.append(result)
        return results
    
    async def generate_from_image(
        self,
        image_url: str,
        prompt: str | None = None,
        config: VideoGenerationConfig | None = None,
        **kwargs: Any,
    ) -> VideoGenerationResult:
        """
        从图片生成视频（图生视频）
        
        Args:
            image_url: 输入图片 URL
            prompt: 可选的运动/风格提示
            config: 配置
            **kwargs: 额外参数
            
        Returns:
            VideoGenerationResult: 生成结果
        """
        # 默认实现：将图片信息添加到提示中
        full_prompt = f"Based on image: {image_url}"
        if prompt:
            full_prompt += f". Motion/style: {prompt}"
        return await self.generate(full_prompt, config, **kwargs)
