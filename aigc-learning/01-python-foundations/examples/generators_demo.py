"""生成器示例：模拟 AIGC 数据流水线。
运行：python generators_demo.py
"""
from __future__ import annotations

import random
import time
from collections.abc import Iterator


def raw_samples(n: int) -> Iterator[dict]:
    """模拟从磁盘流式读取样本。"""
    for i in range(n):
        time.sleep(0.01)
        yield {"id": i, "text": f"sample_{i}", "score": random.random()}


def filter_high_score(stream: Iterator[dict], threshold: float) -> Iterator[dict]:
    """流式过滤：yield from 保持惰性。"""
    for sample in stream:
        if sample["score"] >= threshold:
            yield sample


def add_length(stream: Iterator[dict]) -> Iterator[dict]:
    """流式增强字段。"""
    for sample in stream:
        sample["length"] = len(sample["text"])
        yield sample


def batched(stream: Iterator[dict], batch_size: int) -> Iterator[list[dict]]:
    """流式分批。"""
    batch: list[dict] = []
    for sample in stream:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def fibonacci() -> Iterator[int]:
    """无限生成器——惰性很重要。"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


def flatten(nested):
    """yield from 实现递归展开。"""
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


if __name__ == "__main__":
    print("== 流水线：读取 -> 过滤 -> 增强 -> 分批 ==")
    pipeline = batched(
        add_length(filter_high_score(raw_samples(50), threshold=0.5)),
        batch_size=4,
    )
    for i, batch in enumerate(pipeline):
        if i >= 3:
            break
        print(f"batch {i}: size={len(batch)}, head={batch[0]}")

    print("\n== 无限斐波那契（取前 10） ==")
    fib_stream = fibonacci()
    print([next(fib_stream) for _ in range(10)])

    print("\n== 递归展开 ==")
    nested = [1, [2, [3, [4, 5]], 6], 7]
    print(list(flatten(nested)))
