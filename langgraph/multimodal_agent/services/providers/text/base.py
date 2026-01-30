"""
文本生成 Provider 基类
"""

from abc import abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from multimodal_agent.services.providers.base import (
    BaseProvider,
    ProviderInfo,
    ProviderType,
    GenerationConfig,
    GenerationResult,
)


class TextGenerationConfig(GenerationConfig):
    """文本生成配置"""
    
    max_tokens: int = Field(default=2048, ge=1, le=128000, description="最大 token 数")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-P 采样")
    system_prompt: str | None = Field(default=None, description="系统提示词")


class TextGenerationResult(GenerationResult):
    """文本生成结果"""
    
    content: str = Field(description="生成的文本内容")
    usage: dict[str, int] = Field(default_factory=dict, description="Token 使用量")
    finish_reason: str = Field(default="stop", description="结束原因")


class TextProvider(BaseProvider[TextGenerationResult]):
    """
    文本生成 Provider 抽象基类
    
    所有文本生成 Provider 都必须继承此类。
    """
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.TEXT
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: TextGenerationConfig | None = None,
        **kwargs: Any,
    ) -> TextGenerationResult:
        """生成文本"""
        pass
    
    async def generate_batch(
        self,
        prompts: list[str],
        config: TextGenerationConfig | None = None,
        **kwargs: Any,
    ) -> list[TextGenerationResult]:
        """批量生成文本（默认串行实现）"""
        results = []
        for prompt in prompts:
            result = await self.generate(prompt, config, **kwargs)
            results.append(result)
        return results
    
    async def chat(
        self,
        messages: list[dict[str, str]],
        config: TextGenerationConfig | None = None,
        **kwargs: Any,
    ) -> TextGenerationResult:
        """
        对话模式
        
        Args:
            messages: 对话消息列表 [{"role": "user/assistant/system", "content": "..."}]
            config: 生成配置
            **kwargs: 额外参数
            
        Returns:
            TextGenerationResult: 生成结果
        """
        # 默认实现：将消息拼接为单个 prompt
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return await self.generate(prompt, config, **kwargs)
