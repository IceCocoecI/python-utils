"""NumPy 基础示例：数组创建 / 索引 / 广播 / 向量化。
运行：python numpy_basics.py
"""
from __future__ import annotations

import time

import numpy as np


def demo_creation():
    print("== 1. 数组创建 ==")
    print("zeros:", np.zeros((2, 3)))
    print("arange:", np.arange(10))
    print("linspace:", np.linspace(0, 1, 5))
    rng = np.random.default_rng(42)
    print("randn:", rng.standard_normal((2, 3)))
    print("dtype int32:", np.array([1, 2, 3], dtype=np.int32).dtype)


def demo_indexing():
    print("\n== 2. 切片与索引 ==")
    a = np.arange(24).reshape(2, 3, 4)
    print("a[0, 1, 2]:", a[0, 1, 2])
    print("a[..., 0]:", a[..., 0])

    x = np.array([1, -2, 3, -4, 5])
    print("x > 0:", x[x > 0])

    one_hot = np.eye(5)[np.array([0, 2, 4])]
    print("one_hot:\n", one_hot)


def demo_broadcast():
    print("\n== 3. 广播：ImageNet 归一化 ==")
    x = np.random.rand(2, 3, 4, 4).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    x_norm = (x - mean) / std
    print(f"x shape: {x.shape}, x_norm shape: {x_norm.shape}")
    print(f"per-channel mean after norm: {x_norm.mean(axis=(0, 2, 3))}")


def demo_vectorization():
    print("\n== 4. 向量化 vs for 循环 ==")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1_000_000)

    t0 = time.perf_counter()
    result_loop = np.empty_like(x)
    for i in range(len(x)):
        result_loop[i] = x[i] ** 2 + 3 * x[i]
    t_loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    result_vec = x ** 2 + 3 * x
    t_vec = time.perf_counter() - t0

    print(f"for loop: {t_loop*1000:.1f} ms")
    print(f"vectorized: {t_vec*1000:.1f} ms")
    print(f"speedup: {t_loop/t_vec:.1f}x")


def demo_reductions():
    print("\n== 5. 归约操作 ==")
    a = np.random.randn(3, 4)
    print("sum(axis=0):", a.sum(axis=0))
    print("mean(axis=-1, keepdims):\n", a.mean(axis=-1, keepdims=True))
    print("argmax(axis=1):", a.argmax(axis=1))


def demo_softmax():
    print("\n== 6. 手写 softmax ==")
    logits = np.random.randn(3, 5)
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    print(f"row sums = {probs.sum(axis=-1)}  (should be 1)")


def demo_matmul():
    print("\n== 7. 矩阵乘法与批量矩阵乘 ==")
    a = np.random.randn(3, 4)
    b = np.random.randn(4, 5)
    print("a @ b shape:", (a @ b).shape)

    A = np.random.randn(8, 3, 4)
    B = np.random.randn(8, 4, 5)
    print("batch matmul shape:", (A @ B).shape)


def demo_view_vs_copy():
    print("\n== 8. view vs copy ==")
    a = np.arange(10)
    b = a[2:5]
    b[0] = 999
    print("slice is a view -> a:", a)

    a = np.arange(10)
    c = a[[2, 3, 4]]
    c[0] = 999
    print("fancy index is a copy -> a:", a)


if __name__ == "__main__":
    demo_creation()
    demo_indexing()
    demo_broadcast()
    demo_vectorization()
    demo_reductions()
    demo_softmax()
    demo_matmul()
    demo_view_vs_copy()
