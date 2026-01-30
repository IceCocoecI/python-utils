"""
Provider 模块

支持多模型的策略模式实现。每种类别（文本、图片、视频、音频）都可以有多个 Provider 实现。
"""

from multimodal_agent.services.providers.registry import (
    ProviderRegistry,
    get_provider,
    register_provider,
    list_providers,
)

__all__ = [
    "ProviderRegistry",
    "get_provider",
    "register_provider",
    "list_providers",
]
