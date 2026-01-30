"""
Mock 音频生成 Provider

用于测试和演示的模拟 Provider。
"""

import asyncio
import uuid
from typing import Any

from multimodal_agent.logging_config import get_logger
from multimodal_agent.services.providers.base import ProviderInfo, ProviderType
from multimodal_agent.services.providers.registry import register_provider
from multimodal_agent.services.providers.audio.base import (
    AudioProvider,
    AudioGenerationConfig,
    AudioGenerationResult,
)

logger = get_logger(__name__)


@register_provider(ProviderType.AUDIO, "mock", is_default=True)
class MockAudioProvider(AudioProvider):
    """
    Mock 音频生成 Provider
    
    用于测试，返回模拟的音频 URL。
    """
    
    SUPPORTED_MODELS = ["mock-tts", "mock-elevenlabs", "mock-azure"]
    
    MOCK_VOICES = [
        {"id": "xiaoming", "name": "小明", "language": "zh-CN", "gender": "male"},
        {"id": "xiaohong", "name": "小红", "language": "zh-CN", "gender": "female"},
        {"id": "narrator", "name": "旁白", "language": "zh-CN", "gender": "neutral"},
        {"id": "john", "name": "John", "language": "en-US", "gender": "male"},
        {"id": "emma", "name": "Emma", "language": "en-US", "gender": "female"},
    ]
    
    def __init__(self, delay: float = 0.5):
        super().__init__()
        self.delay = delay
    
    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="mock",
            provider_type=ProviderType.AUDIO,
            description="Mock 音频生成 Provider（测试用）",
            models=self.SUPPORTED_MODELS,
        )
    
    async def generate(
        self,
        prompt: str,
        config: AudioGenerationConfig | None = None,
        **kwargs: Any,
    ) -> AudioGenerationResult:
        """模拟生成音频（TTS）"""
        config = config or AudioGenerationConfig()
        model = config.model or "mock-tts"
        
        logger.info(
            "mock_audio_generation_started",
            model=model,
            text_length=len(prompt),
            voice=config.voice,
        )
        
        # 模拟延迟
        await asyncio.sleep(self.delay)
        
        # 估算音频时长（假设每秒约5个字）
        estimated_duration = len(prompt) / 5.0 / config.speed
        
        # 生成模拟 URL
        request_id = str(uuid.uuid4())
        
        result = AudioGenerationResult(
            request_id=request_id,
            content=f"https://cdn.mock-audio.com/{request_id}.{config.format}",
            provider="mock",
            model=model,
            duration=estimated_duration,
            format=config.format,
            voice=config.voice,
            text=prompt,
            metadata={
                "language": config.language,
                "speed": config.speed,
                "pitch": config.pitch,
                "sample_rate": config.sample_rate,
            },
        )
        
        logger.info(
            "mock_audio_generation_completed",
            request_id=result.request_id,
            duration=result.duration,
        )
        
        return result
    
    async def list_voices(self) -> list[dict[str, Any]]:
        """列出可用声音"""
        return self.MOCK_VOICES
