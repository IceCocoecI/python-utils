"""
Service 层

封装与外部厂商 API 的交互，提供统一的异步接口。

架构：
1. 基础服务层 (base.py) - 通用 HTTP 客户端
2. Provider 层 (providers/) - 策略模式，支持多模型切换
   - text/    文本生成 Provider
   - image/   图片生成 Provider
   - video/   视频生成 Provider
   - audio/   音频生成 Provider
"""

from multimodal_agent.services.base import BaseHTTPService, ServiceException

# 旧版服务（向后兼容）
from multimodal_agent.services.text_service import TextGenerationService
from multimodal_agent.services.image_service import ImageGenerationService
from multimodal_agent.services.video_service import VideoGenerationService

# 新版 Provider 系统
from multimodal_agent.services.providers import (
    ProviderRegistry,
    get_provider,
    register_provider,
    list_providers,
)
from multimodal_agent.services.providers.base import (
    ProviderType,
    ProviderInfo,
    BaseProvider,
    GenerationConfig,
    GenerationResult,
)

__all__ = [
    # 基础类
    "BaseHTTPService",
    "ServiceException",
    
    # 旧版服务（向后兼容）
    "TextGenerationService",
    "ImageGenerationService",
    "VideoGenerationService",
    
    # 新版 Provider 系统
    "ProviderRegistry",
    "get_provider",
    "register_provider",
    "list_providers",
    "ProviderType",
    "ProviderInfo",
    "BaseProvider",
    "GenerationConfig",
    "GenerationResult",
]
