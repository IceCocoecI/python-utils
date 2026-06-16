# 00 · GPU 性能理论框架

> 本文是模块 10 的理论底座。
> CUDA、Triton、Profiler、自定义算子看起来工具很多，但底层都围绕同一个问题：
> 如何让数据以更少的搬运、更高的并行度、更好的硬件利用率完成计算。

---

## 1. 本模块真正解决什么问题？

在 AIGC 工程里，性能瓶颈通常不只是“模型太大”。更常见的是：

| 现象 | 可能原因 |
|---|---|
| GPU 利用率低 | CPU 数据加载慢、kernel 太碎、同步过多 |
| 显存够但速度慢 | memory-bound、HBM 读写过多、没有 fusion |
| 显存爆 | 激活、KV cache、attention 中间矩阵、碎片 |
| 多卡不线性加速 | 通信瓶颈、负载不均、pipeline bubble |
| 自定义 kernel 慢 | 访存不合并、occupancy 低、共享内存冲突 |

模块 10 要建立的不是“会写一个 kernel”的能力，而是性能判断能力：

```text
先测量 → 判断瓶颈类型 → 选择优化手段 → 再测量验证
```

### 1.1 IO-aware 是大模型性能优化的核心思想

很多初学者以为 GPU 优化主要是“少算 FLOPs”。在 AIGC 场景里，更常见的瓶颈是数据搬运：

```text
HBM ↔ cache/shared memory/register ↔ compute unit
```

GPU 算力增长很快，但 HBM 带宽、显存容量、跨卡通信和 kernel launch 开销不会同等增长。
因此大量关键优化的目标不是改变数学结果，而是让同一份数据少搬几次、搬得更连续、在高速存储中复用更多次。

| 技术 | 表面现象 | 本质优化 |
|---|---|---|
| FlashAttention | attention 更省显存、更快 | 不 materialize 完整 attention matrix，减少 HBM 读写 |
| Operator Fusion | 多个算子合成一个 kernel | 中间结果留在寄存器/共享内存，少写回 HBM |
| Tiling | 把大矩阵分块 | 让数据块在 shared memory/register 中重复使用 |
| PagedAttention | KV cache 支持高并发和长上下文 | 改变 KV 的物理布局，减少碎片和无效预留 |
| ZeRO/FSDP | 大模型能跨卡训练 | 切分参数、梯度、优化器状态，用通信换显存 |

这是一条非常重要的理论主线：**AI 系统的能力上限，往往由数据移动而不是公式复杂度决定**。
所以看性能论文或 kernel 代码时，先问四个问题：

1. 它减少了哪一级内存读写？
2. 它增加了哪些计算或通信作为代价？
3. 它适合训练、prefill、decode，还是只适合特定 shape？
4. 它是否保持精确结果，还是引入了近似或量化误差？

#### 深度解读：为什么大模型性能经常输给数据移动？

一个算子在数学上可能很简单，但在 GPU 上慢，常常是因为每次计算前后都要读写大量数据。
例如几个 elementwise 操作的 FLOPs 很少，看起来“便宜”，但如果每一步都单独启动 kernel，并把中间结果写回 HBM，就会反复付出内存带宽和 launch 开销。

这解释了很多 AIGC 性能技术的共同逻辑：

- FlashAttention 不近似 attention，而是改变计算顺序，避免把完整 `(T, T)` attention matrix 写进 HBM。
- Fusion 不一定减少数学运算，却能减少中间 tensor 的读写。
- Tiling 把数据块搬到 shared memory/register 后重复使用，提高算术强度。
- KV cache 用显存换计算，避免每个 decode step 重算历史 token。
- PagedAttention 改变 KV cache 的物理分配方式，让在线请求更少浪费显存。
- ZeRO/FSDP 用通信换显存，让单卡不再保存所有训练状态。

所以性能优化不能只看“理论 FLOPs 少了多少”，还要看：

```text
实际搬了多少 bytes？
是否产生大中间 tensor？
是否能复用已加载的数据？
是否让 GPU 等 CPU、等通信、等小 kernel？
```

对 LLM 来说，prefill 常常更像大矩阵计算问题，decode 常常更像权重和 KV cache 的带宽问题。
同一个模型、同一张 GPU，在不同 batch size、prompt 长度、输出长度和并发模式下，瓶颈可能完全不同。
这也是为什么 profiling 必须按真实 workload 做，而不是只测一个孤立 kernel。

---

## 2. GPU 的执行模型

GPU 面向吞吐，而不是单线程低延迟。

```text
Grid
  └── Block
        └── Thread

SM 执行多个 Block
Warp = 32 个线程的调度单位
```

核心概念：

| 概念 | 含义 |
|---|---|
| SM | Streaming Multiprocessor，GPU 上的主要执行单元 |
| Thread | 最小编程单位 |
| Warp | 32 个线程组成的执行批次 |
| Block | 一组线程，共享 shared memory，可同步 |
| Grid | 一次 kernel launch 的全部 block |

### 2.1 SIMT

GPU 使用 SIMT（Single Instruction, Multiple Threads）模型：

```text
同一个 warp 内的线程在同一时刻执行同一条指令，
只是操作不同的数据。
```

这带来两个直接后果：

- 数据并行任务很适合 GPU。
- warp 内分支不一致会造成 branch divergence，吞吐下降。

### 2.2 Kernel launch 有固定开销

每次从 Python / CPU 发起一个 GPU kernel 都有调度开销。
大量很小的 elementwise 操作会让 GPU 时间碎片化：

```text
x = x + bias
x = gelu(x)
x = dropout(x)
x = residual + x
```

如果每一步都是单独 kernel，就会反复读写 HBM。
这就是 operator fusion 的动机。

---

## 3. GPU 内存层级

性能优化大部分都在优化内存访问。

```text
寄存器 Register       每线程私有，最快，容量最小
共享内存 Shared Mem   每 block 共享，可手动管理
L1 Cache              每 SM 局部缓存
L2 Cache              全 GPU 共享缓存
HBM / Global Memory   容量最大，延迟最高
CPU Memory            跨 PCIe / NVLink，更慢
```

核心原则：

1. 尽量少访问 HBM。
2. 访问 HBM 时尽量连续、合并。
3. 能复用的数据尽量放进寄存器或 shared memory。
4. 不要为了省一次计算制造大量内存读写。

---

## 4. Roofline 模型

Roofline 用来判断一个算子是 compute-bound 还是 memory-bound。

```text
算术强度 = FLOPs / Bytes moved
```

| 类型 | 特征 | 例子 | 优化方向 |
|---|---|---|---|
| Compute-bound | 算术强度高，受算力限制 | 大 GEMM、Conv | Tensor Core、tiling、并行度 |
| Memory-bound | 算术强度低，受带宽限制 | LayerNorm、RoPE、Softmax、Elementwise | fusion、减少 HBM、合并访存 |
| Latency-bound | 工作太小，launch/同步开销明显 | 小 batch、小 shape 算子 | 合并 kernel、batching、CUDA Graph |
| Communication-bound | 多卡通信限制 | all-reduce、all-gather | overlap、分片策略、拓扑优化 |

LLM 中最重的 Linear/GEMM 通常偏 compute-bound；
decode 阶段很多操作因为 batch 小、每步只生成一个 token，更容易 memory-bound 或 latency-bound。

---

## 5. AIGC 常见算子的性能直觉

| 算子 | 常见瓶颈 | 关键优化 |
|---|---|---|
| Linear / GEMM | 算力、Tensor Core 利用率 | 合适矩阵尺寸、混合精度、cuBLAS/CUTLASS |
| Attention | 中间矩阵和 HBM 读写 | FlashAttention、block tiling、online softmax |
| LayerNorm / RMSNorm | memory-bound | fusion、向量化加载 |
| RoPE | memory-bound | fused RoPE、减少中间 tensor |
| Softmax | reduction + memory-bound | fused softmax、online softmax |
| KV Cache | 显存容量、碎片、读取带宽 | PagedAttention、连续布局、量化 |
| Sampling | 小算子碎片 | fused logits processor、batching |

---

## 6. FlashAttention 的理论动机

标准 attention：

```text
S = QK^T
P = softmax(S)
O = PV
```

问题是 `S` 和 `P` 都是 `(seq_len, seq_len)` 级别的中间矩阵。
长序列下，它们会造成巨大的 HBM 读写和显存占用。

FlashAttention 的核心不是近似 attention，而是**保持精确结果，同时改变计算顺序**：

```text
把 Q/K/V 分块
  ↓
每次只计算一个 block
  ↓
用 online softmax 维护全局归一化
  ↓
避免完整 materialize attention matrix
```

它优化的是 IO，而不是数学公式。

---

## 7. PagedAttention 的理论动机

LLM 推理时，每个请求都有 KV cache。
普通连续内存分配会遇到两个问题：

1. 请求长度不同，导致内存碎片。
2. 动态 batch 中请求进入/退出频繁，cache 管理复杂。

PagedAttention 借鉴虚拟内存思想：

```text
KV cache 被切成固定大小 page/block
每个请求维护 block table
逻辑上连续，物理上可以不连续
```

它解决的不是 attention 公式，而是**在线推理服务中的 KV cache 内存管理**。

---

## 8. Operator Fusion 为什么有效？

假设有三个 elementwise 操作：

```text
y = gelu(x + bias)
z = dropout(y) + residual
```

如果分成多个 kernel，流程是：

```text
读 x/bias → 写中间结果 → 读中间结果 → 写 y → 读 y/residual → 写 z
```

fusion 后：

```text
读 x/bias/residual → 在寄存器中完成计算 → 写 z
```

对 memory-bound 操作来说，少写回 HBM 往往比少算几次乘加更重要。

---

## 9. Tiling 的理论意义

Tiling 的目标是让数据在高速存储层级中被多次复用。

以矩阵乘法为例：

```text
C = A @ B
```

朴素实现会反复从 HBM 读取 A/B。
Tiled 实现把 A/B 的小块搬到 shared memory 或寄存器中，让多个线程重复使用。

```text
HBM → Shared Memory → Register → Compute → HBM
```

Tiling 是 GEMM、卷积、FlashAttention 等高性能 kernel 的共同基础。

---

## 10. Occupancy 不是越高越好

Occupancy 表示一个 SM 上活跃 warp 的比例。
它能帮助隐藏内存延迟，但不是唯一目标。

影响 occupancy 的因素：

- 每个线程使用的寄存器数量；
- 每个 block 使用的 shared memory；
- block size；
- kernel 的并行粒度。

常见误区：

- 只追求 occupancy，结果 register spilling，性能更差。
- block size 盲目设大，导致资源占满但吞吐不升。
- 没有先确认瓶颈类型，就调整 launch 参数。

更稳的判断：

```text
如果 memory latency 是瓶颈，提高 occupancy 可能有帮助。
如果 HBM bandwidth 已经打满，提高 occupancy 不一定有用。
如果 compute-bound，重点看 Tensor Core 和指令吞吐。
```

---

## 11. Profiling 的判断流程

优化前必须先测量。推荐流程：

```text
1. 确认结果正确
2. 用 torch.profiler 看 CPU/GPU 时间分布
3. 判断是否 CPU-bound、GPU-bound、memory-bound、communication-bound
4. 对热点 kernel 做 Nsight Systems / Nsight Compute
5. 修改一个变量
6. 再测量，确认收益
```

### 11.1 常见信号

| 观察 | 可能结论 |
|---|---|
| GPU idle 很多 | CPU 数据加载、同步、调度瓶颈 |
| kernel 数量很多且很短 | launch overhead / fusion 机会 |
| HBM 带宽接近上限 | memory-bound |
| Tensor Core 利用低 | shape 不友好、dtype 不对、GEMM 太小 |
| all-reduce 时间长 | 通信瓶颈 |
| 显存碎片高 | 动态分配、KV cache 管理问题 |

---

## 12. 从理论映射到本模块文档

| 理论问题 | 对应文档 |
|---|---|
| GPU 如何执行线程、访问内存、组织 kernel？ | [01-gpu-architecture-and-cuda-basics](./01-gpu-architecture-and-cuda-basics.md) |
| Triton 如何用 block 编程模型表达 fused kernel？ | [02-triton-programming](./02-triton-programming.md) |
| 如何测量 CPU/GPU 时间线、显存、kernel 指标？ | [03-performance-profiling](./03-performance-profiling.md) |
| 如何把自定义 CUDA/Triton 算子接入 PyTorch？ | [04-custom-operators-and-extensions](./04-custom-operators-and-extensions.md) |

---

## 13. 工程判断清单

- [ ] 这段代码是否已经用 profiler 证明是瓶颈？
- [ ] 瓶颈是 CPU、GPU、内存带宽、通信，还是 kernel launch？
- [ ] 这个算子是 compute-bound 还是 memory-bound？
- [ ] 是否存在大量短小 kernel，可以 fusion？
- [ ] HBM 读写是否能减少？
- [ ] 数据访问是否 coalesced？
- [ ] 是否可以通过 tiling 提高数据复用？
- [ ] 是否有不必要的 CPU-GPU 同步？
- [ ] mixed precision 是否真正启用了 Tensor Core？
- [ ] 优化后是否做了数值一致性测试和性能回归测试？
