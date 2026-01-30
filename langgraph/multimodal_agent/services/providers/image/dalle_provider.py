"""
DALL-E 图片生成 Provider

支持 OpenAI DALL-E 系列模型。
"""

import uuid
from typing import Any

import httpx

from multimodal_agent.config import get_config
from multimodal_agent.logging_config import get_logger
from multimodal_agent.services.providers.base import ProviderInfo, ProviderType
from multimodal_agent.services.providers.registry import register_provider
from multimodal_agent.services.providers.image.base import (
    ImageProvider,
    ImageGenerationConfig,
    ImageGenerationResult,
)

logger = get_logger(__name__)


@register_provider(ProviderType.IMAGE, "dalle", is_default=True)
class DallEImageProvider(ImageProvider):
    """
    DALL-E 图片生成 Provider
    
    支持模型：
    - dall-e-3
    - dall-e-2
    """
    
    SUPPORTED_MODELS = ["dall-e-3", "dall-e-2"]
    
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        super().__init__()
        config = get_config()
        self.api_key = api_key or config.openai.api_key.get_secret_value()
        self.base_url = base_url or config.openai.api_base
        self._client: httpx.AsyncClient | None = None
    
    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="dalle",
            provider_type=ProviderType.IMAGE,
            description="OpenAI DALL-E 图片生成",
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
                timeout=httpx.Timeout(120.0),
            )
        return self._client
    
    async def generate(
        self,
        prompt: str,
        config: ImageGenerationConfig | None = None,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """生成图片"""
        config = config or ImageGenerationConfig()
        model = config.model or "dall-e-3"
        
        logger.info(
            "dalle_image_generation_started",
            model=model,
            prompt_length=len(prompt),
            n=config.n,
        )
        
        try:
            client = await self._get_client()
            
            payload = {
                "model": model,
                "prompt": prompt,
                "size": config.size,
                "quality": config.quality,
                "n": config.n,
                **config.extra_params,
            }
            
            # DALL-E 3 支持 style 参数
            if model == "dall-e-3":
                payload["style"] = config.style if config.style in ["natural", "vivid"] else "vivid"
            
            response = await client.post("/images/generations", json=payload)
            response.raise_for_status()
            data = response.json()
            
            # 提取所有图片 URL
            urls = [item["url"] for item in data["data"]]
            content = urls[0] if len(urls) == 1 else urls
            
            # 获取优化后的提示（如果有）
            revised_prompt = data["data"][0].get("revised_prompt") if data["data"] else None
            
            result = ImageGenerationResult(
                request_id=str(uuid.uuid4()),
                content=content,
                provider="dalle",
                model=model,
                revised_prompt=revised_prompt,
                size=config.size,
                style=config.style,
                metadata={
                    "quality": config.quality,
                    "count": len(urls),
                },
            )
            
            logger.info(
                "dalle_image_generation_completed",
                request_id=result.request_id,
                count=len(urls),
            )
            
            return result
            
        except httpx.HTTPError as e:
            logger.error("dalle_image_generation_failed", error=str(e))
            raise
    
    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
