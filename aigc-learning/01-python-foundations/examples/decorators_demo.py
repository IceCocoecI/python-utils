"""装饰器综合示例
运行：python decorators_demo.py
"""
from __future__ import annotations

import functools
import time
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def timer(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        print(f"[timer] {func.__name__}: {dt*1000:.2f} ms")
        return result
    return wrapper


def retry(n_times: int = 3, delay: float = 0.5):
    """带参数的装饰器（装饰器工厂）。"""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_err: Exception | None = None
            for attempt in range(1, n_times + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    print(f"[retry] {func.__name__} attempt {attempt} failed: {e}")
                    time.sleep(delay)
            assert last_err is not None
            raise last_err
        return wrapper
    return decorator


class CallCounter:
    """类装饰器：保存调用次数。"""

    def __init__(self, func: Callable[..., R]) -> None:
        functools.update_wrapper(self, func)
        self._func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self._func(*args, **kwargs)


@timer
@CallCounter
def slow_square(x: int) -> int:
    time.sleep(0.05)
    return x * x


@retry(n_times=3, delay=0.2)
def flaky_api(x: int) -> int:
    import random
    if random.random() < 0.6:
        raise RuntimeError("network error")
    return x + 1


@functools.lru_cache(maxsize=None)
def fib(n: int) -> int:
    """缓存版斐波那契——没有缓存会爆炸。"""
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


if __name__ == "__main__":
    print("== 计时 + 计数 ==")
    for x in range(5):
        slow_square(x)
    print("call count:", slow_square.count)

    print("\n== 重试装饰器 ==")
    try:
        print("result:", flaky_api(10))
    except RuntimeError as e:
        print("final failure:", e)

    print("\n== lru_cache ==")
    print("fib(100) =", fib(100))
    print("cache_info:", fib.cache_info())
