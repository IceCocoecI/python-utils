"""
视频生成 Provider 模块
"""

from multimodal_agent.services.providers.video.base import (
    VideoProvider,
    VideoGenerationResult,
    VideoGenerationConfig,
)
from multimodal_agent.services.providers.video.mock_provider import MockVideoProvider

__all__ = [
    "VideoProvider",
    "VideoGenerationResult",
    "VideoGenerationConfig",
    "MockVideoProvider",
]
