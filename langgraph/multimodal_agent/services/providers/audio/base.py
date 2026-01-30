"""
音频生成 Provider 基类
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


class AudioGenerationConfig(GenerationConfig):
    """音频生成配置"""
    
    voice: str = Field(default="default", description="声音/角色")
    language: str = Field(default="zh-CN", description="语言")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="语速")
    pitch: float = Field(default=1.0, ge=0.5, le=2.0, description="音调")
    format: Literal["mp3", "wav", "ogg"] = Field(default="mp3", description="输出格式")
    sample_rate: int = Field(default=24000, description="采样率")


class AudioGenerationResult(GenerationResult):
    """音频生成结果"""
    
    content: str = Field(description="音频 URL")
    duration: float = Field(description="音频时长（秒）")
    format: str = Field(description="音频格式")
    voice: str = Field(description="使用的声音")
    text: str = Field(description="原始文本")


class AudioProvider(BaseProvider[AudioGenerationResult]):
    """
    音频生成 Provider 抽象基类
    
    所有音频生成 Provider（TTS、音效等）都必须继承此类。
    """
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.AUDIO
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: AudioGenerationConfig | None = None,
        **kwargs: Any,
    ) -> AudioGenerationResult:
        """
        生成音频（TTS）
        
        Args:
            prompt: 要转换的文本
            config: 配置
            **kwargs: 额外参数
            
        Returns:
            AudioGenerationResult: 生成结果
        """
        pass
    
    async def generate_batch(
        self,
        prompts: list[str],
        config: AudioGenerationConfig | None = None,
        **kwargs: Any,
    ) -> list[AudioGenerationResult]:
        """批量生成音频"""
        results = []
        for prompt in prompts:
            result = await self.generate(prompt, config, **kwargs)
            results.append(result)
        return results
    
    @abstractmethod
    async def list_voices(self) -> list[dict[str, Any]]:
        """
        列出可用的声音
        
        Returns:
            list: 声音列表，每个元素包含 id, name, language 等信息
        """
        pass
