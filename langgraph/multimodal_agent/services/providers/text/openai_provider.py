"""
OpenAI 文本生成 Provider

支持 OpenAI GPT 系列模型。
"""

import asyncio
import uuid
from typing import Any

import httpx

from multimodal_agent.config import get_config
from multimodal_agent.logging_config import get_logger
from multimodal_agent.services.providers.base import ProviderInfo, ProviderType
from multimodal_agent.services.providers.registry import register_provider
from multimodal_agent.services.providers.text.base import (
    TextProvider,
    TextGenerationConfig,
    TextGenerationResult,
)

logger = get_logger(__name__)


@register_provider(ProviderType.TEXT, "openai", is_default=True)
class OpenAITextProvider(TextProvider):
    """
    OpenAI 文本生成 Provider
    
    支持模型：
    - gpt-4o
    - gpt-4o-mini
    - gpt-4-turbo
    - gpt-3.5-turbo
    """
    
    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-mini", 
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ]
    
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        super().__init__()
        config = get_config()
        self.api_key = api_key or config.openai.api_key.get_secret_value()
        self.base_url = base_url or config.openai.api_base
        self._client: httpx.AsyncClient | None = None
    
    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="openai",
            provider_type=ProviderType.TEXT,
            description="OpenAI GPT 系列模型",
            models=self.SUPPORTED_MODELS,
        )
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(60.0),
            )
        return self._client
    
    async def generate(
        self,
        prompt: str,
        config: TextGenerationConfig | None = None,
        **kwargs: Any,
    ) -> TextGenerationResult:
        """生成文本"""
        config = config or TextGenerationConfig()
        model = config.model or "gpt-4o-mini"
        
        logger.info(
            "openai_text_generation_started",
            model=model,
            prompt_length=len(prompt),
        )
        
        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            client = await self._get_client()
            response = await client.post(
                "/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    **config.extra_params,
                },
            )
            response.raise_for_status()
            data = response.json()
            
            result = TextGenerationResult(
                request_id=data.get("id", str(uuid.uuid4())),
                content=data["choices"][0]["message"]["content"],
                provider="openai",
                model=data["model"],
                usage=data.get("usage", {}),
                finish_reason=data["choices"][0].get("finish_reason", "stop"),
            )
            
            logger.info(
                "openai_text_generation_completed",
                request_id=result.request_id,
                content_length=len(result.content),
            )
            
            return result
            
        except httpx.HTTPError as e:
            logger.error("openai_text_generation_failed", error=str(e))
            raise
    
    async def chat(
        self,
        messages: list[dict[str, str]],
        config: TextGenerationConfig | None = None,
        **kwargs: Any,
    ) -> TextGenerationResult:
        """对话模式"""
        config = config or TextGenerationConfig()
        model = config.model or "gpt-4o-mini"
        
        try:
            client = await self._get_client()
            response = await client.post(
                "/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    **config.extra_params,
                },
            )
            response.raise_for_status()
            data = response.json()
            
            return TextGenerationResult(
                request_id=data.get("id", str(uuid.uuid4())),
                content=data["choices"][0]["message"]["content"],
                provider="openai",
                model=data["model"],
                usage=data.get("usage", {}),
                finish_reason=data["choices"][0].get("finish_reason", "stop"),
            )
            
        except httpx.HTTPError as e:
            logger.error("openai_chat_failed", error=str(e))
            raise
    
    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
