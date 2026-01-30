"""
文本生成服务

模拟调用大模型接口进行文本生成。
"""

import asyncio
import uuid
from typing import Any

from pydantic import BaseModel, Field

from multimodal_agent.config import TextServiceConfig, get_config
from multimodal_agent.logging_config import get_logger
from multimodal_agent.services.base import BaseHTTPService, ServiceException

logger = get_logger(__name__)


class TextGenerationRequest(BaseModel):
    """文本生成请求"""
    
    prompt: str = Field(description="生成提示")
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    model: str = Field(default="gpt-4o")


class TextGenerationResponse(BaseModel):
    """文本生成响应"""
    
    request_id: str = Field(description="请求 ID")
    content: str = Field(description="生成的文本内容")
    model: str = Field(description="使用的模型")
    usage: dict[str, int] = Field(default_factory=dict, description="Token 使用量")
    finish_reason: str = Field(default="stop", description="结束原因")


class TextGenerationService(BaseHTTPService[TextGenerationResponse]):
    """
    文本生成服务
    
    模拟调用大模型 API 进行文本生成。
    在生产环境中，这里会调用真实的 API（如 OpenAI、Claude 等）。
    """
    
    def __init__(self, config: TextServiceConfig | None = None):
        """
        初始化服务
        
        Args:
            config: 服务配置，如果不提供则从环境变量加载
        """
        if config is None:
            config = get_config().text_service
        super().__init__(config)
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        model: str = "gpt-4o",
        **kwargs: Any,
    ) -> TextGenerationResponse:
        """
        生成文本内容
        
        Args:
            prompt: 生成提示
            max_tokens: 最大 token 数
            temperature: 温度参数
            model: 模型名称
            **kwargs: 额外参数
            
        Returns:
            TextGenerationResponse: 生成结果
            
        Raises:
            ServiceException: 生成失败时抛出
        """
        request = TextGenerationRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
        )
        
        logger.info(
            "text_generation_started",
            prompt_length=len(prompt),
            model=model,
            max_tokens=max_tokens,
        )
        
        try:
            # ============================================
            # MOCK IMPLEMENTATION
            # 在生产环境中，这里应该调用真实的 API
            # ============================================
            response = await self._mock_generate(request)
            
            logger.info(
                "text_generation_completed",
                request_id=response.request_id,
                content_length=len(response.content),
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "text_generation_failed",
                error=str(e),
                prompt_length=len(prompt),
            )
            raise
    
    async def _mock_generate(
        self,
        request: TextGenerationRequest,
    ) -> TextGenerationResponse:
        """
        模拟文本生成（Mock）
        
        在生产环境中，这个方法应该被替换为真实的 API 调用。
        
        Args:
            request: 生成请求
            
        Returns:
            TextGenerationResponse: 模拟的生成结果
        """
        # 模拟网络延迟
        await asyncio.sleep(0.5)
        
        # 模拟生成内容
        mock_content = self._generate_mock_content(request.prompt)
        
        return TextGenerationResponse(
            request_id=str(uuid.uuid4()),
            content=mock_content,
            model=request.model,
            usage={
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(mock_content.split()),
                "total_tokens": len(request.prompt.split()) + len(mock_content.split()),
            },
            finish_reason="stop",
        )
    
    def _generate_mock_content(self, prompt: str) -> str:
        """
        生成模拟内容
        
        Args:
            prompt: 用户提示
            
        Returns:
            str: 模拟生成的文本
        """
        # 根据提示生成相关的模拟响应
        templates = {
            "文章": "这是一篇关于 {topic} 的精彩文章。文章从多个角度深入分析了该主题，包括背景介绍、核心概念、实际应用和未来展望。希望这篇文章能够帮助您更好地理解相关内容。",
            "故事": "从前有一个神奇的王国，那里住着各种各样的生物。有一天，一位勇敢的探险家来到了这片土地，开始了一段不可思议的冒险旅程...",
            "代码": "```python\ndef main():\n    # 这是一个示例代码\n    print('Hello, World!')\n    return 0\n\nif __name__ == '__main__':\n    main()\n```",
            "总结": "根据您的请求，以下是主要内容的总结：\n1. 核心要点分析\n2. 关键信息提取\n3. 建议和结论",
            "default": f"这是基于您的提示「{prompt[:50]}...」生成的文本内容。我们的 AI 模型已经分析了您的需求，并生成了相应的回复。如果您需要更具体的内容，请提供更详细的指示。",
        }
        
        prompt_lower = prompt.lower()
        for key, template in templates.items():
            if key in prompt_lower or key in prompt:
                if "{topic}" in template:
                    topic = prompt[:30] if len(prompt) > 30 else prompt
                    return template.format(topic=topic)
                return template
        
        return templates["default"]
    
    async def _real_generate(
        self,
        request: TextGenerationRequest,
    ) -> TextGenerationResponse:
        """
        真实的 API 调用实现
        
        这是生产环境中应该使用的方法。
        
        Args:
            request: 生成请求
            
        Returns:
            TextGenerationResponse: 生成结果
        """
        response = await self.post(
            "/chat/completions",
            json={
                "model": request.model,
                "messages": [
                    {"role": "user", "content": request.prompt}
                ],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            },
        )
        
        data = response.json()
        
        return TextGenerationResponse(
            request_id=data.get("id", str(uuid.uuid4())),
            content=data["choices"][0]["message"]["content"],
            model=data["model"],
            usage=data.get("usage", {}),
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
        )
