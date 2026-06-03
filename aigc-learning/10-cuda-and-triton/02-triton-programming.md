# 02 · Triton 编程

> CUDA 太底层，PyTorch 太高层——Triton 刚好在中间。
> 用 Python 写 GPU kernel，性能接近手写 CUDA，开发效率提升 10 倍。

---

## 1. 为什么需要 Triton？

```
开发效率                                     性能
    ↑                                         ↑
    │  PyTorch ●                              │                    ● CUDA
    │                                         │
    │            ● Triton                     │          ● Triton
    │                                         │
    │                    ● CUDA               │  ● PyTorch
    └──────────────────────→                  └──────────────────────→

Triton 的定位：接近 CUDA 的性能 + 接近 PyTorch 的开发体验
```

| 维度 | PyTorch | Triton | CUDA C++ |
|---|---|---|---|
| 语言 | Python | Python | C++ |
| 抽象层次 | Tensor 操作 | Block-level 操作 | Thread-level 操作 |
| 性能 | 1x（基准） | 0.7-1.0x vs CUDA | 1x（手动优化后） |
| 开发时间 | 分钟 | 小时 | 天-周 |
| 自动优化 | 无 | Tiling、Coalescing、Shared Memory | 全手动 |
| Fusion | 无（需 torch.compile） | 手动 fusion | 手动 fusion |

**Triton 自动帮你做了什么**：
- 共享内存管理 → 自动
- 内存合并访问 → 自动
- Bank conflict 避免 → 自动
- Thread/Warp 调度 → 自动

**你需要做什么**：
- 定义 block 粒度的算法逻辑
- 指定 block size 等调优参数
- 处理边界条件

---

## 2. Triton 编程模型

### 2.1 核心概念：Program 而非 Thread

CUDA 中你思考的是"每个线程做什么"。Triton 中你思考的是"**每个 program（block）做什么**"。

```
CUDA 思维：                    Triton 思维：
"Thread 42 负责 a[42]"         "Program 0 负责 a[0:128]"
"Thread 43 负责 a[43]"         "Program 1 负责 a[128:256]"
   ...（管理 10000 个线程）         ...（管理几十个 program）
```

### 2.2 最小 Triton Kernel

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,  # 编译期常量
):
    pid = tl.program_id(axis=0)  # 当前 program 的 ID
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # [block_start, block_start+BLOCK_SIZE)

    mask = offsets < n_elements  # 边界掩码

    x = tl.load(x_ptr + offsets, mask=mask)  # 加载一个 block 的数据
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)  # 写回


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# 使用
a = torch.randn(100000, device="cuda")
b = torch.randn(100000, device="cuda")
c = add(a, b)
assert torch.allclose(c, a + b)
```

### 2.3 关键 API 一览

```python
# 身份
pid = tl.program_id(axis=0)          # 当前 program 在 axis 维的 ID
num_programs = tl.num_programs(axis=0)

# 索引生成
offs = tl.arange(0, BLOCK_SIZE)       # [0, 1, 2, ..., BLOCK_SIZE-1]

# 内存操作
data = tl.load(ptr + offs, mask=mask, other=0.0)  # 加载，越界位置填 other
tl.store(ptr + offs, data, mask=mask)              # 存储

# 数学运算
tl.exp(x)
tl.log(x)
tl.sqrt(x)
tl.abs(x)
tl.maximum(x, y)
tl.minimum(x, y)
tl.where(cond, x, y)                 # 条件选择

# 归约
tl.sum(x, axis=0)
tl.max(x, axis=0)
tl.min(x, axis=0)

# 矩阵乘法（Tensor Core）
tl.dot(a, b)                          # 小矩阵乘法，映射到 Tensor Core

# 原子操作
tl.atomic_add(ptr + offs, val, mask=mask)
tl.atomic_max(ptr + offs, val, mask=mask)
```

---

## 3. Fused Softmax：理解 Fusion 的威力

Softmax 是 AIGC 模型中最高频的操作之一。PyTorch 原生实现需要多次遍历数据：

```python
# PyTorch naive softmax：3 次全局内存遍历
def naive_softmax(x):
    x_max = x.max(dim=-1, keepdim=True).values   # 遍历 1: 求 max
    x = x - x_max                                 # 遍历 2: 减 max
    numerator = torch.exp(x)                       # 遍历 3: exp
    denominator = numerator.sum(dim=-1, keepdim=True)  # 遍历 4: sum
    return numerator / denominator                 # 遍历 5: div
    # 5 次读写 HBM！
```

Triton fused softmax **一次遍历搞定**：

```python
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        output, x,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
```

**为什么 fusion 有效？**
- Naive：5 次读写 HBM × N 个元素 = 10N 次 HBM 访问
- Fused：1 次读 + 1 次写 HBM = 2N 次 HBM 访问
- **HBM 访问减少 5 倍**，对 memory-bound 操作来说几乎等于 5 倍加速

---

## 4. Autotuning：让 Triton 自动找最优配置

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=4),
    ],
    key=["M", "N", "K"],  # 当这些值变化时重新搜索
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

**`@triton.autotune` 的工作方式**：
1. 在第一次调用时，尝试所有 `configs`
2. 对每个 config benchmark 多次
3. 选出最快的，后续调用直接用这个 config
4. 当 `key` 参数值变化时（如矩阵尺寸变了），重新搜索

---

## 5. FlashAttention 的核心思想

FlashAttention 是理解 Triton 实战的最佳案例。

### 5.1 标准 Attention 的问题

```python
# 标准实现：需要 O(N²) 内存存 S 和 P
Q, K, V = ...  # [batch, heads, seq_len, head_dim]
S = Q @ K.T / sqrt(d)   # [N, N] 注意力矩阵  ← 写入 HBM
P = softmax(S)           # [N, N]              ← 读/写 HBM
O = P @ V                # [N, d]              ← 读 HBM

# seq_len=4096, heads=32: S 矩阵 = 4096² × 32 × 4B ≈ 2 GB
```

### 5.2 FlashAttention 的 Tiling 策略

```
标准 Attention：
  整个 S 矩阵 [N×N] 存在 HBM 中

FlashAttention：
  ┌──────────────────────────────────┐
  │  Q 分成 Br 大小的块               │
  │  K, V 分成 Bc 大小的块            │
  │                                   │
  │  For each Q_block:                │
  │    For each K_block, V_block:     │
  │      在 SRAM 中计算 S_block       │
  │      在 SRAM 中计算 P_block       │
  │      累积 O_block += P_block × V  │
  │                                   │
  │  S 矩阵从不完整存在于 HBM 中       │
  └──────────────────────────────────┘
```

### 5.3 Online Softmax Trick

Tiling 的难点在于 softmax 需要全行的 max 和 sum，但我们一次只能看到一小块。**Online softmax** 解决了这个问题：

```python
# Online softmax: 逐 block 更新统计量
m_i = -inf       # running max
l_i = 0          # running sum of exp
O_i = 0          # running output

for j in range(num_kv_blocks):
    S_ij = Q_i @ K_j.T / sqrt(d)       # 当前块的注意力分数
    m_ij = max(S_ij)                     # 当前块的 max
    m_new = max(m_i, m_ij)               # 全局 max 更新

    # 修正之前累积的结果（因为 max 变了）
    l_i = l_i * exp(m_i - m_new) + sum(exp(S_ij - m_new))
    O_i = O_i * exp(m_i - m_new) + exp(S_ij - m_new) @ V_j

    m_i = m_new

O_i = O_i / l_i  # 最终归一化
```

**关键洞察**：即使分块计算，最终结果与标准 softmax **完全一致（exact）**，不是近似。

### 5.4 简化的 Triton FlashAttention（单头、forward only）

```python
@triton.jit
def flash_attention_fwd_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    k_ptrs = K + k_offset + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
    v_ptrs = V + v_offset + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX)

    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < N_CTX)
        qk = tl.dot(q, tl.trans(k))
        qk *= 1.0 / tl.sqrt(tl.cast(BLOCK_DMODEL, tl.float32))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < N_CTX)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    acc = acc / l_i[:, None]
    o_ptrs = Out + o_offset + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)
```

---

## 6. 与 PyTorch 集成：`torch.autograd.Function`

要让 Triton kernel 支持反向传播，需要用 `torch.autograd.Function`：

```python
class TritonSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        output = torch.empty_like(x)
        softmax_kernel[(n_rows,)](
            output, x,
            x.stride(0), output.stride(0),
            n_cols, BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (output,) = ctx.saved_tensors
        # softmax backward: grad_input = output * (grad_output - sum(grad_output * output))
        grad_input = output * (grad_output - (grad_output * output).sum(dim=-1, keepdim=True))
        return grad_input


# 使用
triton_softmax = TritonSoftmax.apply
y = triton_softmax(x)  # 可以 .backward()
```

---

## 7. 调试 Triton Kernels

### 7.1 数值正确性验证

```python
import triton.testing

def test_softmax():
    x = torch.randn(128, 1024, device="cuda")
    y_triton = softmax(x)
    y_torch = torch.softmax(x, dim=-1)
    triton.testing.assert_close(y_triton, y_torch, atol=1e-4, rtol=1e-4)
    print("✅ Softmax test passed!")
```

### 7.2 使用 tl.device_print

```python
@triton.jit
def debug_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:  # 只在第一个 program 打印
        offs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs)
        tl.device_print("x values", x)
```

### 7.3 常见 Debug 流程

```python
# 1. 先用小尺寸测试
x_small = torch.randn(4, 8, device="cuda")
y = my_kernel_wrapper(x_small)
print(y)

# 2. 对比 PyTorch 参考实现
y_ref = reference_impl(x_small)
print(torch.allclose(y, y_ref, atol=1e-4))

# 3. 逐步增大尺寸
for size in [16, 64, 256, 1024, 4096]:
    x = torch.randn(size, size, device="cuda")
    y = my_kernel_wrapper(x)
    y_ref = reference_impl(x)
    assert torch.allclose(y, y_ref, atol=1e-3), f"Failed at size {size}"

# 4. 测试边界条件
x_edge = torch.randn(7, 13, device="cuda")  # 非 2 的幂
y = my_kernel_wrapper(x_edge)
```

---

## 8. 性能对比与 Benchmarking

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M": 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.softmax(x, dim=-1), quantiles=quantiles
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: softmax(x), quantiles=quantiles
        )
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(show_plots=True, print_data=True)
```

---

## 9. 真实世界的 Triton Kernels

### 9.1 Unsloth

[Unsloth](https://github.com/unslothai/unsloth) 用 Triton 重写了 LLM 微调中的关键 kernel：

```python
# Unsloth 的 Triton kernels 包括：
# - Cross Entropy Loss（fused，避免 logits 物化）
# - RoPE（旋转位置编码）
# - RMS LayerNorm
# - SwiGLU activation

# 这些 kernel 的共同特点：
# 1. 都是 memory-bound 操作
# 2. PyTorch 原生需要多次 HBM 读写
# 3. Fused Triton 版本减少 2-5x 内存访问
# 结果：微调速度提升 2x，显存减少 50%
```

### 9.2 vLLM

[vLLM](https://github.com/vllm-project/vllm) 的核心 kernel 用 Triton 和 CUDA 实现：

```python
# vLLM 的关键 GPU kernels：
# - PagedAttention: 虚拟内存管理 KV cache
# - Rotary Embedding: RoPE 计算
# - Activation kernels: SiLU & Mul fused
# - Layernorm kernels: RMSNorm fused

# PagedAttention 的核心思想：
# KV cache 不再连续存储，而是分成固定大小的 "page"
# 每个 page 通过 block table 索引
# 允许非连续内存分配 → 消除 KV cache 碎片 → 提升 batch throughput
```

### 9.3 xformers

[xformers](https://github.com/facebookresearch/xformers) 提供高效 Transformer 组件：

```python
# Memory-efficient attention（Triton 和 CUDA 两种后端）
from xformers.ops import memory_efficient_attention

output = memory_efficient_attention(query, key, value, attn_bias=None)
```

---

## 10. 常见坑

### 10.1 BLOCK_SIZE 必须是 2 的幂

```python
# ❌ 编译错误
BLOCK_SIZE: tl.constexpr = 100

# ✅ 必须是 2 的幂
BLOCK_SIZE: tl.constexpr = 128

# 动态计算最近的 2 的幂
BLOCK_SIZE = triton.next_power_of_2(n_cols)
```

### 10.2 忘记 mask 导致越界

```python
# ❌ 当 n_elements 不是 BLOCK_SIZE 的倍数时，越界读写
x = tl.load(x_ptr + offsets)

# ✅ 始终加 mask
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```

### 10.3 数据类型不匹配

```python
# ❌ fp16 累加会损失精度
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
acc += tl.dot(a, b)  # 累加在 fp16，精度可能出问题

# ✅ 用 fp32 累加，最后再转
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
acc += tl.dot(a, b)
result = acc.to(tl.float16)
```

### 10.4 Grid 函数写错

```python
# ❌ grid 大小算错导致数据没算完
grid = (n_elements // BLOCK_SIZE,)

# ✅ 向上取整
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
# 等价于 (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
```

### 10.5 Triton 版本兼容性

```python
# Triton API 在不同版本间有 breaking changes
# 常见问题：
# - tl.load/tl.store 的 mask 参数名变化
# - tl.dot 的 allow_tf32 参数
# - tl.trans 在某些版本叫 tl.permute

# 建议：锁定 triton 版本（跟 PyTorch 配套的版本）
# pip install torch  # 自带对应的 triton 版本
```

### 10.6 JIT 编译慢

```python
# 第一次运行 Triton kernel 会触发 JIT 编译（几秒到几十秒）
# 后续有缓存，但参数变化时可能重新编译

# 技巧：预热
with torch.no_grad():
    _ = my_kernel_wrapper(torch.randn(16, 16, device="cuda"))

# autotune 会让编译更慢（要尝试所有 configs）
# 生产环境建议固定最优 config，去掉 autotune
```

---

## 小结

| 概念 | 要点 |
|---|---|
| 编程模型 | Program-level（block 粒度），不是 thread-level |
| 核心 API | `tl.load` / `tl.store` + `mask` 处理边界 |
| Fusion | 多个 elementwise 操作合并，减少 HBM 访问 |
| FlashAttention | Tiling + Online Softmax，O(N) 内存 |
| Autotuning | `@triton.autotune` 自动搜索最优 block size |
| PyTorch 集成 | `torch.autograd.Function` 包装 Triton kernel |
| 调试 | `triton.testing.assert_close` + 小尺寸先验证 |

**一句话总结**：Triton 让你用 Python 的开发效率，达到接近 CUDA 的 GPU 性能——这是 2024-2026 年写自定义 kernel 的首选方案。

下一节学习性能分析——如何用 profiler 找到真正的瓶颈，而不是盲目优化。
