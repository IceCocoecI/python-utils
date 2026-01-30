"""
音频生成 Provider 模块
"""

from multimodal_agent.services.providers.audio.base import (
    AudioProvider,
    AudioGenerationResult,
    AudioGenerationConfig,
)
from multimodal_agent.services.providers.audio.mock_provider import MockAudioProvider

__all__ = [
    "AudioProvider",
    "AudioGenerationResult",
    "AudioGenerationConfig",
    "MockAudioProvider",
]
