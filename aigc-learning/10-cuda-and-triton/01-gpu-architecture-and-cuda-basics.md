# 01 · GPU 架构与 CUDA 基础

> GPU 不是"更快的 CPU"——它是一台完全不同的机器。
> 理解它的架构和编程模型，是写出高性能 kernel 的前提。

---

## 1. GPU vs CPU：两种不同的设计哲学

```
┌──────── CPU ────────┐          ┌──────────── GPU ──────────────┐
│                     │          │                               │
│  ┌───┐ ┌───┐       │          │  ┌─┐┌─┐┌─┐┌─┐ ┌─┐┌─┐┌─┐┌─┐ │
│  │ C │ │ C │       │          │  │c││c││c││c│ │c││c││c││c│ │
│  │ O │ │ O │  大   │          │  └─┘└─┘└─┘└─┘ └─┘└─┘└─┘└─┘ │
│  │ R │ │ R │  缓存  │          │  ┌─┐┌─┐┌─┐┌─┐ ┌─┐┌─┐┌─┐┌─┐ │
│  │ E │ │ E │       │          │  │c││c││c││c│ │c││c││c││c│ │
│  └───┘ └───┘       │          │  └─┘└─┘└─┘└─┘ └─┘└─┘└─┘└─┘ │
│  少量大核心          │          │  ... × 数千个小核心 ...        │
│  复杂控制逻辑        │          │  简单控制，海量并行             │
│  低延迟             │          │  高吞吐                       │
└─────────────────────┘          └───────────────────────────────┘
```

| 维度 | CPU | GPU |
|---|---|---|
| 核心数 | 4-128（大核心） | 数千-数万（小核心） |
| 设计目标 | 低延迟、单线程性能 | 高吞吐、大规模并行 |
| 控制逻辑 | 复杂（分支预测、乱序执行） | 简单（SIMT） |
| 缓存 | 大（L1/L2/L3 共几十 MB） | 小（L1/L2 几 MB） |
| 适合任务 | 复杂逻辑、分支多 | 数据并行、同一操作重复万次 |

**SIMT (Single Instruction, Multiple Threads)**：GPU 的核心执行模型。一个 warp（32 个线程）在同一时刻执行**同一条指令**，但操作不同的数据。

```
Warp (32 threads) 执行 ADD 指令：
  Thread 0:  a[0] + b[0]
  Thread 1:  a[1] + b[1]
  Thread 2:  a[2] + b[2]
  ...
  Thread 31: a[31] + b[31]
  ─── 一个时钟周期完成 32 次加法 ───
```

**Branch Divergence（分支分歧）**——SIMT 的代价：

```c
// 如果 warp 内有些线程走 if，有些走 else
if (threadIdx.x % 2 == 0) {
    do_A();  // 偶数线程执行，奇数线程空等
} else {
    do_B();  // 奇数线程执行，偶数线程空等
}
// 两个分支串行执行，吞吐减半！
```

---

## 2. GPU 内存层级

这是写 GPU kernel 时**最重要**的知识。绝大部分性能优化都围绕内存访问。

```
┌────────────────────────────────────────────────────┐
│                   GPU 内存层级                       │
│                                                     │
│   寄存器 (Registers)          ← 最快，每线程私有      │
│   │  延迟: ~1 cycle                                 │
│   │  带宽: ~TB/s                                    │
│   ▼                                                 │
│   共享内存 (Shared Memory)    ← 同一 block 内共享     │
│   │  延迟: ~5 cycles          (可编程的 L1)           │
│   │  大小: 48-228 KB/SM                              │
│   ▼                                                 │
│   L1 Cache                   ← 与共享内存共享物理空间  │
│   L2 Cache                   ← 全局共享, 几 MB-几十MB │
│   │  延迟: ~50-200 cycles                            │
│   ▼                                                 │
│   全局内存 (Global Memory / HBM)  ← 最慢，但最大      │
│     延迟: ~400-600 cycles                            │
│     带宽: 1-3 TB/s (A100-H100)                      │
│     大小: 40-192 GB                                  │
└────────────────────────────────────────────────────┘
```

**核心法则**：尽可能让数据待在高层级（寄存器 > 共享内存 > 全局内存），减少对 HBM 的访问次数。

---

## 3. 内存带宽与算力：Roofline 模型

**Roofline 模型**是判断 kernel 瓶颈的最重要工具。

```
性能                    ┌───────────── 计算峰值 (Compute Bound)
(FLOPS)                │
    │                 ╱│
    │               ╱  │
    │             ╱    │
    │           ╱      │
    │         ╱        │
    │       ╱ ← 带宽墙 (Memory Bound)
    │     ╱
    │   ╱
    │ ╱
    └──────────────────────── 算术强度 (FLOPs/Byte)

算术强度 = 计算量(FLOPs) / 数据搬运量(Bytes)
```

| 操作类型 | 算术强度 | 瓶颈 | 例子 |
|---|---|---|---|
| Elementwise | 极低 (< 1) | 内存带宽 | ReLU、LayerNorm、Dropout |
| Reduction | 低 | 内存带宽 | Softmax、Sum |
| GEMM (大矩阵) | 高 (> 100) | 算力 | Linear、Attention 的 Q×K |
| Conv2D (大 batch) | 高 | 算力 | 卷积层 |

**关键洞察**：LLM 推理的大部分操作（LayerNorm、RoPE、Softmax、残差加）都是 **memory-bound** 的。这就是 **operator fusion** 如此重要的原因——减少中间结果写回 HBM 的次数。

---

## 4. NVIDIA GPU 代际：AIGC 相关

| GPU | 架构 | FP16 TFLOPS | BF16 Tensor Core TFLOPS | HBM 容量 | HBM 带宽 | 年份 |
|---|---|---|---|---|---|---|
| A100 SXM | Ampere | 312 | 312 | 40/80 GB | 2.0 TB/s | 2020 |
| H100 SXM | Hopper | 989 | 989 | 80 GB | 3.35 TB/s | 2023 |
| H200 | Hopper | 989 | 989 | 141 GB | 4.8 TB/s | 2024 |
| B200 | Blackwell | 2250 | 2250 | 192 GB | 8.0 TB/s | 2025 |

关键趋势：
- **算力增长速度 >> 带宽增长速度** → memory-bound 问题越来越严重
- **Tensor Core 是主力**：FP16/BF16 GEMM 在 Tensor Core 上跑，比 CUDA Core 快 10x+
- **HBM 容量决定能装多大模型**：H200 的 141GB 可以放下 70B FP16 模型

---

## 5. CUDA 编程模型：Grid → Block → Thread

```
┌─────────────────── Grid ───────────────────┐
│                                            │
│  ┌─── Block(0,0) ───┐  ┌─── Block(1,0) ──┐│
│  │ T(0,0) T(1,0) .. │  │ T(0,0) T(1,0).. ││
│  │ T(0,1) T(1,1) .. │  │ T(0,1) T(1,1).. ││
│  │ ...               │  │ ...              ││
│  └───────────────────┘  └─────────────────┘│
│  ┌─── Block(0,1) ───┐  ┌─── Block(1,1) ──┐│
│  │ ...               │  │ ...              ││
│  └───────────────────┘  └─────────────────┘│
└────────────────────────────────────────────┘

一个 Grid 有多个 Block。
一个 Block 有多个 Thread。
一个 Warp = 32 个连续 Thread。
一个 Block 最多 1024 个 Thread。
同一个 Block 内的 Thread 可以共享 Shared Memory 并同步。
不同 Block 之间不能直接通信（除了原子操作和全局内存）。
```

**索引体系**：

```c
// 每个线程知道自己在哪
int tx = threadIdx.x;   // 线程在 block 内的 x 坐标
int ty = threadIdx.y;   // 线程在 block 内的 y 坐标
int bx = blockIdx.x;    // block 在 grid 内的 x 坐标
int by = blockIdx.y;    // block 在 grid 内的 y 坐标
int bdx = blockDim.x;   // block 的 x 维度大小
int bdy = blockDim.y;   // block 的 y 维度大小
int gdx = gridDim.x;    // grid 的 x 维度大小

// 计算全局唯一线程 ID（最常用的 1D 版本）
int global_id = blockIdx.x * blockDim.x + threadIdx.x;
```

---

## 6. CUDA Kernel 基础

### 6.1 函数修饰符

```c
__global__ void kernel(...) { }   // GPU 上执行，CPU 调用（这就是 kernel）
__device__ void helper(...) { }   // GPU 上执行，GPU 调用（device 辅助函数）
__host__   void cpu_fn(...) { }   // CPU 上执行，CPU 调用（默认）
__shared__ float smem[256];       // 声明共享内存
```

### 6.2 第一个 Kernel：Vector Add

```c
// vector_add.cu

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {       // 边界检查：线程数可能超过数据量
        c[idx] = a[idx] + b[idx];
    }
}

// 调用
int n = 1000000;
int block_size = 256;
int grid_size = (n + block_size - 1) / block_size;  // 向上取整
vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
```

**`<<<grid_size, block_size>>>`** 是 CUDA 特有的 kernel launch 语法，指定 Grid 和 Block 的维度。

### 6.3 Naive 矩阵乘法

```c
// C = A × B, A: [M, K], B: [K, N], C: [M, N]
__global__ void matmul_naive(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 调用
dim3 block(16, 16);              // 256 threads per block
dim3 grid((N + 15) / 16, (M + 15) / 16);
matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
```

这个 naive 版本的性能大约只有 cuBLAS 的 **1-5%**。接下来讲怎么优化。

---

## 7. 内存合并访问（Memory Coalescing）

GPU 从全局内存读数据是以 **128 字节（32 个 float）** 为单位的事务。如果一个 warp 的 32 个线程访问连续的内存地址，只需要一次事务；如果地址分散，需要多次事务。

```c
// ✅ 合并访问：相邻线程访问相邻地址
// Thread 0 访问 a[0], Thread 1 访问 a[1], ...
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = a[idx];   // 一个 warp 一次事务搞定

// ❌ 非合并访问：stride 访问
float val = a[idx * stride];  // stride > 1 时，地址不连续

// ❌ 非合并访问：行主序矩阵的列访问
// Thread 0 访问 A[0][col], Thread 1 访问 A[1][col], ...
float val = A[threadIdx.x * N + col];  // 间隔 N 个元素，极其浪费带宽
```

**实践建议**：矩阵乘法中，让同一个 warp 的线程读取矩阵的同一行（连续内存），而不是同一列。

---

## 8. 共享内存与 Tiling

共享内存是 **可编程的 L1 缓存**，延迟只有全局内存的 1/100。Tiling 是最常用的优化模式。

### 8.1 Tiled 矩阵乘法

```c
#define TILE_SIZE 16

__global__ void matmul_tiled(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 协作加载：每个线程搬一个元素到共享内存
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        __syncthreads();  // 等所有线程加载完毕

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();  // 等所有线程计算完毕再加载下一个 tile
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**为什么 Tiling 有效？**
- 不 tiling：每个元素从 HBM 读 K 次（A 的每一行被 N 个线程读，B 的每一列被 M 个线程读）
- Tiling：每个元素从 HBM 读 1 次到共享内存，然后从共享内存读 TILE_SIZE 次
- 全局内存访问减少了约 TILE_SIZE 倍

### 8.2 Bank Conflict

共享内存被分成 32 个 bank（对应 32 个 warp 线程），同一个 warp 的不同线程访问同一个 bank 时会产生 **bank conflict**，访问串行化。

```c
// ✅ 无 bank conflict：连续访问
__shared__ float smem[32];
float val = smem[threadIdx.x];  // 每个线程访问不同 bank

// ❌ 有 bank conflict：stride=2 访问
float val = smem[threadIdx.x * 2];  // 2 路 bank conflict

// 常见修复：padding
__shared__ float smem[32][33];  // 多加一列，错开 bank
```

---

## 9. Warp-Level 原语

Warp 内的 32 个线程可以直接交换数据，不需要共享内存。

```c
// Warp Shuffle: 线程间直接传值
float val = __shfl_sync(0xFFFFFFFF, my_val, src_lane);
// 从 warp 内 src_lane 号线程拿 my_val 的值

float val = __shfl_xor_sync(0xFFFFFFFF, my_val, mask);
// 与 lane_id XOR mask 的线程交换值

// Warp Reduce（求和）的经典实现
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;  // 所有线程都得到求和结果
}
```

**为什么 warp reduce 只需要 5 步就能对 32 个值求和？**
- 第 1 步 (offset=16)：线程 0 和 16 交换并相加，线程 1 和 17 交换并相加 ...
- 第 2 步 (offset=8)：线程 0 和 8 交换 ...
- 经过 log2(32) = 5 步，所有值的和就出来了

---

## 10. 同步机制

```c
__syncthreads();
// Block 内全部线程的 barrier。
// 调用前所有共享内存写入对 block 内所有线程可见。
// ⚠️ 不能放在条件分支里（不是所有线程都走到这里会死锁）。

// Cooperative Groups (CUDA 9+): 更灵活的同步
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel() {
    cg::thread_block block = cg::this_thread_block();
    block.sync();  // 等效于 __syncthreads()

    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    float sum = cg::reduce(warp, val, cg::plus<float>());
}
```

---

## 11. 错误处理

CUDA API 调用不会抛异常——你需要主动检查返回值。

```c
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// 使用
CUDA_CHECK(cudaMalloc(&d_ptr, size));
CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));

// Kernel launch 后检查错误（kernel 是异步的，需要同步后才能拿到错误）
my_kernel<<<grid, block>>>(...);
CUDA_CHECK(cudaGetLastError());      // 检查 launch 错误
CUDA_CHECK(cudaDeviceSynchronize()); // 等待完成 + 检查执行错误
```

---

## 12. 在 PyTorch 中加载自定义 CUDA Kernel

`torch.utils.cpp_extension` 让你可以在 Python 中直接编译和加载 CUDA 代码：

```python
import torch
from torch.utils.cpp_extension import load

# JIT 编译并加载（第一次慢，后续有缓存）
my_cuda = load(
    name="my_cuda_kernels",
    sources=["my_kernel.cu", "bindings.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)

# 使用
a = torch.randn(1024, device="cuda")
b = torch.randn(1024, device="cuda")
c = my_cuda.vector_add(a, b)
```

一个最小可运行的 CUDA extension 结构：

```
my_extension/
├── my_kernel.cu       # CUDA kernel 实现
└── bindings.cpp       # pybind11 绑定
```

```cpp
// bindings.cpp
#include <torch/extension.h>

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add_cuda, "Vector add (CUDA)");
}
```

```cpp
// my_kernel.cu
#include <torch/extension.h>

__global__ void vector_add_kernel(const float* a, const float* b,
                                   float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(),
        c.data_ptr<float>(), n
    );
    return c;
}
```

---

## 13. CUDA 内存管理基础

```c
// 分配与释放设备内存
float *d_data;
CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));
CUDA_CHECK(cudaFree(d_data));

// Host ↔ Device 数据传输
CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(float),
                       cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(h_data, d_data, n * sizeof(float),
                       cudaMemcpyDeviceToHost));

// 异步传输（需要 pinned memory + stream）
float *h_pinned;
CUDA_CHECK(cudaMallocHost(&h_pinned, n * sizeof(float)));  // Pinned memory
cudaStream_t stream;
CUDA_CHECK(cudaStreamCreate(&stream));
CUDA_CHECK(cudaMemcpyAsync(d_data, h_pinned, n * sizeof(float),
                            cudaMemcpyHostToDevice, stream));
```

在 PyTorch 中，这些通常被 `tensor.to(device)` 和 `tensor.cuda()` 封装了。但理解底层有助于：
- 排查 `cudaErrorMemoryAllocation` 错误
- 理解 `pin_memory=True` 在 DataLoader 中的作用
- 理解 CUDA stream 与异步执行

---

## 14. 常见坑

### 14.1 忘记边界检查

```c
// ❌ 没有边界检查：n=1000 但启动了 1024 个线程
__global__ void kernel(float* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] = 0;  // idx=1000~1023 越界！

// ✅ 加边界检查
    if (idx < n) a[idx] = 0;
}
```

### 14.2 在条件分支中调用 `__syncthreads()`

```c
// ❌ 死锁：不是所有线程都能到达 __syncthreads()
if (threadIdx.x < 16) {
    smem[threadIdx.x] = data;
    __syncthreads();  // 线程 16-31 永远等不到 barrier
}

// ✅ 把 sync 放在分支外面
if (threadIdx.x < 16) {
    smem[threadIdx.x] = data;
}
__syncthreads();
```

### 14.3 忽略 CUDA 错误

```c
// ❌ kernel 失败了但你不知道
my_kernel<<<grid, block>>>(...);
// 程序继续跑，后面的结果全是垃圾

// ✅ 始终检查
my_kernel<<<grid, block>>>(...);
CUDA_CHECK(cudaGetLastError());
```

### 14.4 Block size 选不好

```c
// ❌ block size = 1024，每个线程用 64 个寄存器 → 只能放 1 个 block/SM → 低占用率
// ❌ block size = 32，太少线程，无法隐藏内存延迟

// ✅ 经验值：128 或 256 通常是好的起点
// 用 CUDA Occupancy Calculator 或 Nsight Compute 确定最优值
```

### 14.5 从 Python 直接看 CUDA 错误

```python
# PyTorch 中的 CUDA 错误通常延迟报告
# 在出错的 kernel 之后的任何 CUDA 操作才会报错
# 打开同步模式来定位真正出错的行：
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
```

### 14.6 共享内存大小超限

```c
// A100 每个 SM 最多 164 KB 共享内存（可配置）
// 但默认每个 block 最多 48 KB
// 需要更多？使用动态配置：
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
kernel<<<grid, block, 98304>>>(args...);
```

---

## 小结

| 概念 | 要点 |
|---|---|
| SIMT | 32 线程一个 warp，执行同一指令 |
| 内存层级 | 寄存器 → 共享内存 → L2 → HBM，速度差 100x+ |
| Roofline | 算术强度决定瓶颈是算力还是带宽 |
| Coalescing | 相邻线程访问相邻地址，一次事务搞定 |
| Tiling | 用共享内存做数据复用，减少 HBM 访问 |
| Warp Shuffle | 线程间直接传值，比共享内存更快 |
| 错误处理 | CUDA_CHECK 宏 + cudaGetLastError |

**一句话总结**：GPU 编程的核心就是**管理数据移动**。计算是"免费"的，搬数据才是瓶颈。

下一节学习 Triton——用 Python 写 GPU kernel，不再需要手动管理线程和共享内存。
