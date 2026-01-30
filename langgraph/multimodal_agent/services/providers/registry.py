"""
Provider 注册表

管理所有 Provider 的注册、查找和实例化。
支持通过配置动态切换 Provider。
"""

from typing import Type, Any
from functools import lru_cache

from multimodal_agent.logging_config import get_logger
from multimodal_agent.services.providers.base import (
    BaseProvider,
    ProviderType,
    ProviderInfo,
)

logger = get_logger(__name__)


class ProviderRegistry:
    """
    Provider 注册表
    
    单例模式，管理所有 Provider 的注册和获取。
    """
    
    _instance: "ProviderRegistry | None" = None
    _initialized: bool = False
    
    def __new__(cls) -> "ProviderRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._providers: dict[ProviderType, dict[str, Type[BaseProvider]]] = {
                ProviderType.TEXT: {},
                ProviderType.IMAGE: {},
                ProviderType.VIDEO: {},
                ProviderType.AUDIO: {},
            }
            self._defaults: dict[ProviderType, str] = {}
            self._instances: dict[str, BaseProvider] = {}
            ProviderRegistry._initialized = True
    
    def register(
        self,
        provider_class: Type[BaseProvider],
        provider_type: ProviderType,
        name: str,
        is_default: bool = False,
    ) -> None:
        """
        注册 Provider
        
        Args:
            provider_class: Provider 类
            provider_type: Provider 类型
            name: Provider 名称
            is_default: 是否设为默认
        """
        if name in self._providers[provider_type]:
            logger.warning(
                "provider_already_registered",
                name=name,
                provider_type=provider_type.value,
            )
        
        self._providers[provider_type][name] = provider_class
        
        if is_default or provider_type not in self._defaults:
            self._defaults[provider_type] = name
        
        logger.info(
            "provider_registered",
            name=name,
            provider_type=provider_type.value,
            is_default=is_default,
        )
    
    def get(
        self,
        provider_type: ProviderType,
        name: str | None = None,
        use_cache: bool = True,
        **init_kwargs: Any,
    ) -> BaseProvider:
        """
        获取 Provider 实例
        
        Args:
            provider_type: Provider 类型
            name: Provider 名称，为空则使用默认
            use_cache: 是否使用缓存实例（默认 True）
            **init_kwargs: Provider 初始化参数
            
        Returns:
            BaseProvider: Provider 实例
            
        Raises:
            ValueError: Provider 不存在
            
        Note:
            如果提供了 init_kwargs 且 use_cache=True，将忽略缓存创建新实例。
            这是因为不同的 init_kwargs 可能需要不同配置的实例。
        """
        if name is None:
            name = self._defaults.get(provider_type)
            if name is None:
                raise ValueError(f"No default provider for type: {provider_type.value}")
        
        if name not in self._providers[provider_type]:
            available = list(self._providers[provider_type].keys())
            raise ValueError(
                f"Provider '{name}' not found for type '{provider_type.value}'. "
                f"Available: {available}"
            )
        
        provider_class = self._providers[provider_type][name]
        
        # 如果有自定义参数，总是创建新实例
        if init_kwargs:
            return provider_class(**init_kwargs)
        
        # 使用缓存
        if use_cache:
            cache_key = f"{provider_type.value}:{name}"
            if cache_key not in self._instances:
                self._instances[cache_key] = provider_class()
            return self._instances[cache_key]
        
        # 不使用缓存，创建新实例
        return provider_class()
    
    def list_providers(
        self,
        provider_type: ProviderType | None = None,
    ) -> dict[str, list[ProviderInfo]]:
        """
        列出所有 Provider
        
        Args:
            provider_type: 可选，过滤指定类型
            
        Returns:
            dict: Provider 信息字典
        """
        result: dict[str, list[ProviderInfo]] = {}
        
        types_to_list = [provider_type] if provider_type else list(ProviderType)
        
        for ptype in types_to_list:
            providers_info = []
            for name, provider_class in self._providers[ptype].items():
                try:
                    # 临时创建实例获取信息
                    instance = provider_class()
                    info = instance.info
                    info.is_default = (self._defaults.get(ptype) == name)
                    providers_info.append(info)
                except Exception as e:
                    logger.warning(
                        "failed_to_get_provider_info",
                        name=name,
                        error=str(e),
                    )
            result[ptype.value] = providers_info
        
        return result
    
    def set_default(self, provider_type: ProviderType, name: str) -> None:
        """
        设置默认 Provider
        
        Args:
            provider_type: Provider 类型
            name: Provider 名称
        """
        if name not in self._providers[provider_type]:
            raise ValueError(f"Provider '{name}' not found for type '{provider_type.value}'")
        
        self._defaults[provider_type] = name
        logger.info(
            "default_provider_set",
            provider_type=provider_type.value,
            name=name,
        )
    
    def clear_cache(self) -> None:
        """清除实例缓存"""
        self._instances.clear()


# 全局注册表实例
_registry = ProviderRegistry()


def register_provider(
    provider_type: ProviderType,
    name: str,
    is_default: bool = False,
):
    """
    装饰器：注册 Provider
    
    Usage:
        @register_provider(ProviderType.TEXT, "openai", is_default=True)
        class OpenAITextProvider(BaseProvider):
            ...
    """
    def decorator(cls: Type[BaseProvider]) -> Type[BaseProvider]:
        _registry.register(cls, provider_type, name, is_default)
        return cls
    return decorator


def get_provider(
    provider_type: ProviderType,
    name: str | None = None,
    **kwargs: Any,
) -> BaseProvider:
    """
    获取 Provider 实例
    
    Args:
        provider_type: Provider 类型
        name: Provider 名称，为空则使用默认
        **kwargs: 初始化参数
        
    Returns:
        BaseProvider: Provider 实例
    """
    return _registry.get(provider_type, name, **kwargs)


def list_providers(
    provider_type: ProviderType | None = None,
) -> dict[str, list[ProviderInfo]]:
    """
    列出所有 Provider
    """
    return _registry.list_providers(provider_type)


def set_default_provider(provider_type: ProviderType, name: str) -> None:
    """
    设置默认 Provider
    """
    _registry.set_default(provider_type, name)
