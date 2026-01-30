"""
基础 HTTP 服务模块

提供异步 HTTP 客户端的基类，封装通用的请求逻辑。
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic

import httpx

from multimodal_agent.config import ServiceConfig
from multimodal_agent.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class ServiceException(Exception):
    """服务层异常"""
    
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body
        self.original_exception = original_exception
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.status_code:
            parts.append(f"status_code={self.status_code}")
        if self.response_body:
            parts.append(f"response={self.response_body[:200]}")
        return " | ".join(parts)


class BaseHTTPService(ABC, Generic[T]):
    """
    HTTP 服务基类
    
    提供统一的异步 HTTP 请求接口，包含：
    - 超时设置
    - 认证头部
    - 错误处理
    - 重试逻辑
    """
    
    def __init__(self, config: ServiceConfig):
        """
        初始化服务
        
        Args:
            config: 服务配置
        """
        self.config = config
        self._client: httpx.AsyncClient | None = None
    
    @property
    def base_url(self) -> str:
        """服务基础 URL"""
        return self.config.base_url
    
    @property
    def timeout(self) -> float:
        """请求超时时间（秒）"""
        return float(self.config.timeout)
    
    @property
    def headers(self) -> dict[str, str]:
        """请求头部"""
        return {
            "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    
    async def _get_client(self) -> httpx.AsyncClient:
        """
        获取或创建 HTTP 客户端
        
        Returns:
            httpx.AsyncClient: 异步 HTTP 客户端
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client
    
    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        发送 HTTP 请求
        
        Args:
            method: HTTP 方法
            endpoint: API 端点
            **kwargs: 额外的请求参数
            
        Returns:
            httpx.Response: HTTP 响应
            
        Raises:
            ServiceException: 请求失败时抛出
        """
        client = await self._get_client()
        
        logger.debug(
            "sending_http_request",
            method=method,
            endpoint=endpoint,
            base_url=self.base_url,
        )
        
        try:
            response = await client.request(method, endpoint, **kwargs)
            
            logger.debug(
                "http_response_received",
                status_code=response.status_code,
                endpoint=endpoint,
            )
            
            # 检查响应状态
            if response.status_code >= 400:
                raise ServiceException(
                    message=f"HTTP request failed: {response.status_code}",
                    status_code=response.status_code,
                    response_body=response.text,
                )
            
            return response
            
        except httpx.TimeoutException as e:
            logger.error(
                "http_timeout",
                endpoint=endpoint,
                timeout=self.timeout,
            )
            raise ServiceException(
                message=f"Request timeout after {self.timeout}s",
                original_exception=e,
            ) from e
            
        except httpx.ConnectError as e:
            logger.error(
                "http_connection_error",
                endpoint=endpoint,
                error=str(e),
            )
            raise ServiceException(
                message=f"Connection failed: {str(e)}",
                original_exception=e,
            ) from e
            
        except httpx.HTTPError as e:
            logger.error(
                "http_error",
                endpoint=endpoint,
                error=str(e),
            )
            raise ServiceException(
                message=f"HTTP error: {str(e)}",
                original_exception=e,
            ) from e
    
    async def post(
        self,
        endpoint: str,
        json: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        发送 POST 请求
        
        Args:
            endpoint: API 端点
            json: JSON 请求体
            **kwargs: 额外参数
            
        Returns:
            httpx.Response: HTTP 响应
        """
        return await self._request("POST", endpoint, json=json, **kwargs)
    
    async def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        发送 GET 请求
        
        Args:
            endpoint: API 端点
            params: 查询参数
            **kwargs: 额外参数
            
        Returns:
            httpx.Response: HTTP 响应
        """
        return await self._request("GET", endpoint, params=params, **kwargs)
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> T:
        """
        生成内容（抽象方法）
        
        Args:
            prompt: 生成提示
            **kwargs: 额外参数
            
        Returns:
            T: 生成结果
        """
        pass
