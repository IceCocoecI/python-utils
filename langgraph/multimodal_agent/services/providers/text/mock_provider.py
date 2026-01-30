"""
Mock 文本生成 Provider

用于测试和演示的模拟 Provider。
"""

import asyncio
import uuid
from typing import Any

from multimodal_agent.logging_config import get_logger
from multimodal_agent.services.providers.base import ProviderInfo, ProviderType
from multimodal_agent.services.providers.registry import register_provider
from multimodal_agent.services.providers.text.base import (
    TextProvider,
    TextGenerationConfig,
    TextGenerationResult,
)

logger = get_logger(__name__)


@register_provider(ProviderType.TEXT, "mock")
class MockTextProvider(TextProvider):
    """
    Mock 文本生成 Provider
    
    用于测试，不需要 API Key。
    """
    
    SUPPORTED_MODELS = ["mock-gpt", "mock-fast"]
    
    def __init__(self, delay: float = 0.5):
        super().__init__()
        self.delay = delay
    
    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="mock",
            provider_type=ProviderType.TEXT,
            description="Mock 文本生成 Provider（测试用）",
            models=self.SUPPORTED_MODELS,
        )
    
    async def generate(
        self,
        prompt: str,
        config: TextGenerationConfig | None = None,
        **kwargs: Any,
    ) -> TextGenerationResult:
        """模拟生成文本"""
        config = config or TextGenerationConfig()
        model = config.model or "mock-gpt"
        
        logger.info(
            "mock_text_generation_started",
            model=model,
            prompt_length=len(prompt),
        )
        
        # 模拟延迟
        await asyncio.sleep(self.delay)
        
        # 生成模拟内容
        mock_content = self._generate_mock_content(prompt)
        
        result = TextGenerationResult(
            request_id=str(uuid.uuid4()),
            content=mock_content,
            provider="mock",
            model=model,
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(mock_content.split()),
                "total_tokens": len(prompt.split()) + len(mock_content.split()),
            },
            finish_reason="stop",
        )
        
        logger.info(
            "mock_text_generation_completed",
            request_id=result.request_id,
            content_length=len(result.content),
        )
        
        return result
    
    def _generate_mock_content(self, prompt: str) -> str:
        """生成模拟内容"""
        prompt_lower = prompt.lower()
        
        # 漫剧相关
        if "剧本" in prompt or "故事" in prompt:
            return """【漫剧剧本】

场景一：清晨的咖啡馆
角色：小明（主角）、咖啡师
小明走进咖啡馆，阳光透过玻璃窗洒在地板上。

小明：（内心独白）新的一天，新的开始。
咖啡师：早上好！还是老样子吗？
小明：是的，一杯美式，谢谢。

场景二：办公室
角色：小明、同事小红
小明坐在工位上，屏幕上显示着一个神秘的消息。

小明：（惊讶）这是...
小红：怎么了？
小明：我收到了一封奇怪的邮件..."""
        
        if "分镜" in prompt:
            return """【分镜脚本】

分镜1: 
- 镜头：全景
- 场景：城市日出，高楼林立
- 描述：橙红色的阳光照亮城市天际线

分镜2:
- 镜头：中景
- 场景：咖啡馆门口
- 描述：主角推门而入，门铃叮咚作响

分镜3:
- 镜头：特写
- 场景：咖啡杯
- 描述：热气腾腾的咖啡，拉花呈心形

分镜4:
- 镜头：正面中景
- 场景：主角面部
- 描述：主角微笑，眼中带着期待"""
        
        # 默认响应
        return f"""这是基于您的提示生成的内容：

{prompt[:100]}...

我已经分析了您的需求，并生成了相应的文本内容。这段内容是由 Mock Provider 生成的，
用于演示和测试目的。在实际使用中，您可以切换到真实的 Provider（如 OpenAI、Claude 等）
来获得更高质量的生成结果。

如需更多帮助，请告诉我具体的需求。"""
