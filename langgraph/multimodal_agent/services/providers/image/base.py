"""
图片生成 Provider 基类
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


class ImageGenerationConfig(GenerationConfig):
    """图片生成配置"""
    
    size: str = Field(default="1024x1024", description="图片尺寸")
    quality: Literal["standard", "hd"] = Field(default="standard", description="图片质量")
    style: str = Field(
        default="vivid", 
        description="图片风格（如 natural, vivid, anime, realistic 等）"
    )
    n: int = Field(default=1, ge=1, le=10, description="生成数量")
    negative_prompt: str | None = Field(default=None, description="负面提示词")


class ImageGenerationResult(GenerationResult):
    """图片生成结果"""
    
    content: str | list[str] = Field(description="图片 URL（单个或多个）")
    revised_prompt: str | None = Field(default=None, description="优化后的提示")
    size: str = Field(description="图片尺寸")
    style: str = Field(description="图片风格")


class ImageProvider(BaseProvider[ImageGenerationResult]):
    """
    图片生成 Provider 抽象基类
    
    所有图片生成 Provider 都必须继承此类。
    """
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.IMAGE
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: ImageGenerationConfig | None = None,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """生成图片"""
        pass
    
    async def generate_batch(
        self,
        prompts: list[str],
        config: ImageGenerationConfig | None = None,
        **kwargs: Any,
    ) -> list[ImageGenerationResult]:
        """批量生成图片"""
        results = []
        for prompt in prompts:
            result = await self.generate(prompt, config, **kwargs)
            results.append(result)
        return results
    
    async def generate_multiple(
        self,
        prompt: str,
        count: int = 4,
        config: ImageGenerationConfig | None = None,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """
        生成多张图片（同一提示）
        
        Args:
            prompt: 提示词
            count: 生成数量
            config: 配置
            **kwargs: 额外参数
            
        Returns:
            ImageGenerationResult: 包含多个 URL 的结果
        """
        # 创建新的配置对象，避免修改传入的对象
        if config is None:
            new_config = ImageGenerationConfig(n=count)
        else:
            new_config = config.model_copy(update={"n": count})
        return await self.generate(prompt, new_config, **kwargs)
