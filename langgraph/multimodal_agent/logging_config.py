"""
日志配置模块

使用 structlog 实现结构化日志记录。
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from multimodal_agent.config import get_config


def setup_logging() -> None:
    """
    配置结构化日志系统
    
    根据配置选择 JSON 或 Console 格式输出。
    """
    config = get_config()
    log_level = getattr(logging, config.logging.level)
    
    # 共享的处理器链
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.ExtraAdder(),
    ]
    
    if config.logging.format == "json":
        # JSON 格式 - 适合生产环境
        processors: list[Processor] = shared_processors + [
            structlog.processors.JSONRenderer()
        ]
    else:
        # Console 格式 - 适合开发环境
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    
    # 配置 structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # 配置标准库 logging（用于第三方库）
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    获取带有模块名的日志记录器
    
    Args:
        name: 模块名称
        
    Returns:
        structlog.stdlib.BoundLogger: 绑定的日志记录器
    """
    return structlog.get_logger(name)


class LogContext:
    """日志上下文管理器，用于追踪请求链路"""
    
    @staticmethod
    def bind(**kwargs: Any) -> None:
        """绑定上下文变量"""
        structlog.contextvars.bind_contextvars(**kwargs)
    
    @staticmethod
    def unbind(*keys: str) -> None:
        """解绑上下文变量"""
        structlog.contextvars.unbind_contextvars(*keys)
    
    @staticmethod
    def clear() -> None:
        """清除所有上下文变量"""
        structlog.contextvars.clear_contextvars()
