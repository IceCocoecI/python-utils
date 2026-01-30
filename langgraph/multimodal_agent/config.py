"""
配置管理模块

使用 pydantic-settings 管理环境变量和应用配置。
支持从 .env 文件和环境变量加载配置。
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAIConfig(BaseSettings):
    """OpenAI API 配置 - 用于 Router 意图识别"""
    
    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API Key"
    )
    api_base: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API Base URL"
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        description="用于意图识别的模型名称"
    )


class ServiceConfig(BaseSettings):
    """外部服务通用配置"""
    
    base_url: str
    api_key: SecretStr = Field(default=SecretStr("mock-api-key"))
    timeout: int = Field(default=30, ge=1, le=300)


class TextServiceConfig(ServiceConfig):
    """文本生成服务配置"""
    
    model_config = SettingsConfigDict(
        env_prefix="TEXT_SERVICE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    base_url: str = Field(
        default="https://api.example.com/v1/text",
        description="文本生成服务 Base URL"
    )
    timeout: int = Field(default=30)


class ImageServiceConfig(ServiceConfig):
    """图片生成服务配置"""
    
    model_config = SettingsConfigDict(
        env_prefix="IMAGE_SERVICE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    base_url: str = Field(
        default="https://api.example.com/v1/image",
        description="图片生成服务 Base URL"
    )
    timeout: int = Field(default=60)


class VideoServiceConfig(ServiceConfig):
    """视频生成服务配置"""
    
    model_config = SettingsConfigDict(
        env_prefix="VIDEO_SERVICE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    base_url: str = Field(
        default="https://api.example.com/v1/video",
        description="视频生成服务 Base URL"
    )
    timeout: int = Field(default=120)


class LoggingConfig(BaseSettings):
    """日志配置"""
    
    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="日志级别"
    )
    format: Literal["json", "console"] = Field(
        default="json",
        description="日志格式"
    )


class AppConfig(BaseSettings):
    """应用总配置"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    text_service: TextServiceConfig = Field(default_factory=TextServiceConfig)
    image_service: ImageServiceConfig = Field(default_factory=ImageServiceConfig)
    video_service: VideoServiceConfig = Field(default_factory=VideoServiceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


@lru_cache()
def get_config() -> AppConfig:
    """
    获取应用配置单例
    
    Returns:
        AppConfig: 应用配置实例
    """
    return AppConfig()
