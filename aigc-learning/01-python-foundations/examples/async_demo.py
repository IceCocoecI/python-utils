"""异步编程示例：模拟并发调用 LLM API。
运行：python async_demo.py
"""
from __future__ import annotations

import asyncio
import random
import time
from contextlib import asynccontextmanager


async def fake_llm_call(prompt: str, latency_range: tuple[float, float] = (0.3, 1.2)) -> str:
    """模拟 LLM API：带随机延迟 + 偶尔失败。"""
    await asyncio.sleep(random.uniform(*latency_range))
    if random.random() < 0.1:
        raise RuntimeError(f"rate limited on prompt: {prompt[:20]}")
    return f"<answer to: {prompt}>"


async def with_retry(coro_fn, *args, retries: int = 3, **kwargs):
    """通用重试包装。"""
    for attempt in range(retries):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt * 0.1)


async def bounded_batch_call(
    prompts: list[str],
    concurrency: int = 5,
) -> list[str]:
    """限流并发调用。"""
    sem = asyncio.Semaphore(concurrency)

    async def one(prompt: str) -> str:
        async with sem:
            return await with_retry(fake_llm_call, prompt)

    return await asyncio.gather(*(one(p) for p in prompts))


async def stream_tokens(prompt: str):
    """异步生成器：模拟流式输出。"""
    full = f"<streamed answer to {prompt}>".split()
    for token in full:
        await asyncio.sleep(0.05)
        yield token + " "


@asynccontextmanager
async def fake_http_client():
    """异步上下文管理器。"""
    print("  [client] opening connection")
    try:
        yield {"base_url": "https://api.example.com"}
    finally:
        print("  [client] closing connection")


async def producer_consumer_demo():
    """asyncio.Queue：vLLM 调度思想的玩具版。"""
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=5)

    async def producer():
        for i in range(10):
            await queue.put(f"prompt_{i}")
            print(f"  [prod] enqueue prompt_{i}")
        await queue.put(None)

    async def consumer():
        while True:
            item = await queue.get()
            if item is None:
                break
            result = await fake_llm_call(item, latency_range=(0.1, 0.3))
            print(f"  [cons] got {result}")

    await asyncio.gather(producer(), consumer())


async def main():
    print("== 串行 vs 并发 ==")
    prompts = [f"question {i}" for i in range(8)]

    t0 = time.perf_counter()
    serial = []
    for p in prompts:
        serial.append(await with_retry(fake_llm_call, p))
    print(f"serial took {time.perf_counter() - t0:.2f}s, got {len(serial)} results")

    t0 = time.perf_counter()
    results = await bounded_batch_call(prompts, concurrency=4)
    print(f"concurrent (lim=4) took {time.perf_counter() - t0:.2f}s, got {len(results)} results")

    print("\n== 流式输出 ==")
    async for tok in stream_tokens("hi"):
        print(tok, end="", flush=True)
    print()

    print("\n== 异步上下文管理器 ==")
    async with fake_http_client() as client:
        print(f"  using {client['base_url']}")

    print("\n== 生产者-消费者 ==")
    await producer_consumer_demo()


if __name__ == "__main__":
    asyncio.run(main())
