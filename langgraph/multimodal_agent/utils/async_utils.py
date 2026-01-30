"""
异步工具模块

提供在同步上下文中安全运行异步代码的工具。
"""

import asyncio
from typing import TypeVar, Coroutine, Any
import concurrent.futures

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    安全地在同步上下文中运行异步协程
    
    处理以下场景：
    1. 没有事件循环：直接使用 asyncio.run()
    2. 已有事件循环：在新线程中创建独立的事件循环运行
    
    Args:
        coro: 要运行的协程
        
    Returns:
        T: 协程的返回值
        
    Raises:
        Exception: 协程执行中的任何异常
    
    Example:
        async def fetch_data():
            return await some_async_operation()
        
        # 在同步函数中调用
        result = run_async(fetch_data())
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 没有运行中的事件循环
        loop = None
    
    if loop is not None:
        # 已有事件循环，在新线程中运行
        # 这是因为不能在已有事件循环中直接调用 asyncio.run()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        # 没有事件循环，直接运行
        return asyncio.run(coro)


async def gather_with_limit(
    *coros: Coroutine[Any, Any, T],
    limit: int = 5,
) -> list[T]:
    """
    带并发限制的 asyncio.gather
    
    Args:
        *coros: 协程列表
        limit: 最大并发数
        
    Returns:
        list[T]: 结果列表
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def limited_coro(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*[limited_coro(c) for c in coros])
