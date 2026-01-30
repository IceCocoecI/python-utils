"""
Provider 基类定义

定义所有 Provider 的通用接口和抽象基类。
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from enum import Enum

from pydantic import BaseModel, Field

T = TypeVar("T")


class ProviderType(str, Enum):
    """Provider 类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class ProviderInfo(BaseModel):
    """Provider 信息"""
    
    name: str = Field(description="Provider 名称")
    provider_type: ProviderType = Field(description="Provider 类型")
    description: str = Field(default="", description="Provider 描述")
    models: list[str] = Field(default_factory=list, description="支持的模型列表")
    is_default: bool = Field(default=False, description="是否为默认 Provider")


class GenerationConfig(BaseModel):
    """通用生成配置"""
    
    model: str | None = Field(default=None, description="使用的模型名称")
    timeout: int = Field(default=60, description="超时时间（秒）")
    extra_params: dict[str, Any] = Field(default_factory=dict, description="额外参数")


class GenerationResult(BaseModel):
    """通用生成结果"""
    
    request_id: str = Field(description="请求 ID")
    content: str | list[str] = Field(description="生成的内容或 URL")
    provider: str = Field(description="使用的 Provider 名称")
    model: str = Field(description="使用的模型")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")


class BaseProvider(ABC, Generic[T]):
    """
    Provider 抽象基类
    
    所有具体的 Provider 实现都必须继承此类。
    """
    
    def __init__(self):
        self._info: ProviderInfo | None = None
    
    @property
    @abstractmethod
    def info(self) -> ProviderInfo:
        """获取 Provider 信息"""
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> T:
        """
        生成内容
        
        Args:
            prompt: 生成提示
            config: 生成配置
            **kwargs: 额外参数
            
        Returns:
            T: 生成结果
        """
        pass
    
    @abstractmethod
    async def generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> list[T]:
        """
        批量生成内容
        
        Args:
            prompts: 生成提示列表
            config: 生成配置
            **kwargs: 额外参数
            
        Returns:
            list[T]: 生成结果列表
        """
        pass
    
    async def close(self) -> None:
        """关闭 Provider（释放资源）"""
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.info.name})>"
