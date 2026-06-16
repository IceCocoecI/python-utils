# 模块 10：深水区工程 —— CUDA 与 Triton

> 当你用 `torch.compile` 已经不够快，当你需要写一个 FlashAttention 变体，当你想理解 vLLM 的 paged attention kernel 到底在做什么——
> 你需要下到 GPU 编程这一层。

---

## 为什么 AIGC 工程师需要懂 GPU 编程？

大多数 AIGC 工程师的日常在 Python 层：调用 `transformers`、写训练循环、用 `vLLM` 部署。但越来越多的关键场景要求你**理解甚至编写 GPU kernel**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AIGC 工程的性能层级                                │
│                                                                     │
│   Python 层     │  PyTorch API / HuggingFace / vLLM                 │
│   ─────────────────────────────────────────────────────             │
│   编译器层      │  torch.compile / Triton JIT                       │
│   ─────────────────────────────────────────────────────             │
│   Kernel 层     │  Triton kernels / CUDA C++ kernels                │
│   ─────────────────────────────────────────────────────             │
│   硬件层        │  SM / Tensor Core / HBM / NVLink                  │
│                                                                     │
│          ↑ 越往下，性能天花板越高，但开发成本也越高 ↑                  │
└─────────────────────────────────────────────────────────────────────┘
```

**性能差距有多大？**

| 操作 | PyTorch naive | 优化后 (Triton/CUDA) | 加速比 |
|---|---|---|---|
| Attention (seq=4096) | 1x | 2-4x (FlashAttention) | 2-4x |
| LayerNorm + Dropout 融合 | 1x | 1.5-2x | 1.5-2x |
| Softmax | 1x | 2-3x (fused Triton) | 2-3x |
| PagedAttention (推理) | 1x (无法实现) | vLLM kernel | ∞ |

**你不需要成为 CUDA 专家**，但你需要：
1. 能读懂 kernel 代码（debug 和 PR review）
2. 用 Triton 写中等复杂度的 fused kernel
3. 用 profiler 定位性能瓶颈
4. 能把自定义算子集成到 PyTorch 训练/推理管线

---

## 学习内容

| # | 文档 | 核心话题 |
|---|---|---|
| 00 | [gpu-performance-theory](./00-gpu-performance-theory.md) | GPU 执行模型 / 内存层级 / Roofline / FlashAttention / PagedAttention / fusion / profiling |
| 01 | [gpu-architecture-and-cuda-basics](./01-gpu-architecture-and-cuda-basics.md) | GPU 架构 / 内存层级 / CUDA 编程模型 / kernel 编写 / 内存优化 |
| 02 | [triton-programming](./02-triton-programming.md) | Triton 编程模型 / fused kernel / FlashAttention / autotuning |
| 03 | [performance-profiling](./03-performance-profiling.md) | torch.profiler / Nsight Systems / Nsight Compute / 优化工作流 |
| 04 | [custom-operators-and-extensions](./04-custom-operators-and-extensions.md) | pybind11 / cpp_extension / torch.library / FlexAttention |

---

## 前置知识

- 模块 02：PyTorch 基础（Tensor、autograd、nn.Module）
- 模块 05：分布式训练基础（理解 GPU 显存布局有帮助）
- 基本的 C/C++ 语法（不需要精通，但需要能读）

---

## 理论与实践怎么组织

本模块建议按三层学习：

| 层次 | 要回答的问题 | 对应材料 |
|---|---|---|
| 理论层 | GPU 如何执行线程？为什么大模型推理常常受 HBM、kernel launch、KV cache 限制？ | `00-gpu-performance-theory.md` |
| Kernel 层 | CUDA 和 Triton 如何表达 thread/block、tiling、fusion、shared memory、autotuning？ | `01`、`02`、`04` 文档 |
| Profiling 层 | 如何用 torch.profiler / Nsight 判断瓶颈类型，并验证优化收益？ | `03-performance-profiling.md` |

学习顺序建议：

1. 先读 `00`，建立 compute-bound、memory-bound、fusion、tiling 的判断框架。
2. 再读 `01` 和 `02`，理解 CUDA 与 Triton 的编程模型差异。
3. 学 `03` 后再做优化，避免在没有测量的情况下改 kernel。
4. 最后读 `04`，把自定义算子接入 PyTorch 和 `torch.compile`。

---

## 示例代码（`examples/`）

| 文件 | 说明 | 是否需要 GPU |
|---|---|---|
| [`torch_profiler_demo.py`](./examples/torch_profiler_demo.py) | 用 `torch.profiler` 分析一个 tiny training step，输出耗时表，可选写 TensorBoard trace | 否，GPU 下信息更完整 |

运行：

```bash
conda run -n aigc python aigc-learning/10-cuda-and-triton/examples/torch_profiler_demo.py --steps 5
```

默认使用 CPU，并隐藏部分 PyTorch profiler 在 CPU-only 环境下的底层设备探测日志。GPU 环境可显式增加 `--device cuda`；需要内存统计时再加 `--profile-memory`；调试 profiler 本身时可加 `--show-profiler-stderr`。

生成 trace：

```bash
conda run -n aigc python aigc-learning/10-cuda-and-triton/examples/torch_profiler_demo.py --steps 5 --trace
tensorboard --logdir aigc-learning/10-cuda-and-triton/examples/outputs/profiler_demo
```

---

## 推荐配套资源

### 核心文档

| 资源 | 说明 |
|---|---|
| [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | NVIDIA 官方 CUDA 编程指南 |
| [Triton Documentation](https://triton-lang.org/) | Triton 官方文档与教程 |
| [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/) | 官方示例：vector add → matmul → FlashAttention |
| [CUDA by Example (书)](https://developer.nvidia.com/cuda-example) | 经典入门书 |
| [PyTorch C++ Extension Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html) | 官方自定义算子教程 |
| [PyTorch Custom Operators](https://pytorch.org/docs/stable/library.html) | torch.library 文档 |

### 核心论文

| 论文 | 要点 |
|---|---|
| [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) | IO-aware exact attention，tiling + recomputation |
| [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691) | 更好的 work partitioning，接近硬件峰值 |
| [FlashAttention-3 (Shah et al., 2024)](https://arxiv.org/abs/2407.08691) | H100 异步 + FP8 |
| [Triton (Tillet et al., 2019)](https://arxiv.org/abs/1907.00587) | Triton 编译器设计 |
| [PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180) | vLLM 的核心：虚拟内存管理 KV cache |
| [Roofline Model (Williams et al., 2009)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyworky.pdf) | 性能分析的基础框架 |

### 核心代码库

| 库 | 用途 |
|---|---|
| [openai/triton](https://github.com/openai/triton) | Triton 编译器 |
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | FlashAttention CUDA 实现 |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | LLM 推理引擎（大量 Triton/CUDA kernel） |
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | 用 Triton 加速 LLM 微调 |
| [facebookresearch/xformers](https://github.com/facebookresearch/xformers) | 高效 Transformer 组件 |
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | CUDA 模板库（高性能 GEMM） |

---

## 硬件与环境建议

```
最低配置（能跑 Triton kernel 和 profiler）：
  - 1× NVIDIA GPU（Ampere 及以上：A100 / RTX 3090 / RTX 4090）
  - CUDA Toolkit 12.x
  - PyTorch 2.4+（带 Triton）

推荐配置（CUDA C++ 开发）：
  - 1× A100-80GB 或 H100
  - CUDA Toolkit 12.4+
  - nsight-systems / nsight-compute

关键依赖：
  pip install torch triton
  pip install ninja             # 加速 C++ extension 编译

  # 可选：profiling 工具
  pip install torch-tb-profiler
  # nsight-systems 和 nsight-compute 从 NVIDIA 官网安装
```

---

## 自检清单

- [ ] 画出 GPU 的内存层级：寄存器 → 共享内存 → L1/L2 → HBM，延迟和带宽各是多少？
- [ ] 解释 SIMT 模型：一个 warp 有多少线程？为什么 branch divergence 会降低性能？
- [ ] 什么是内存合并访问（coalesced access）？写一个反例。
- [ ] 共享内存的 bank conflict 是什么？如何避免？
- [ ] A100 和 H100 的 HBM 带宽分别是多少？Tensor Core 算力差多少？
- [ ] 用 Roofline 模型判断一个 kernel 是 compute-bound 还是 memory-bound。
- [ ] 解释 Triton 的编程模型：为什么用 block pointer 而不是 thread？
- [ ] 写一个 Triton fused softmax kernel，并和 PyTorch 原生对比性能。
- [ ] FlashAttention 的核心思想是什么？为什么它是 IO-aware 的？
- [ ] 什么是 online softmax trick？为什么 FlashAttention 需要它？
- [ ] 用 `torch.profiler` 生成一份 Chrome trace，能看到 CPU/GPU 时间线。
- [ ] 用 Nsight Systems 分析一个训练 step，找到 GPU idle 时间。
- [ ] 解释 operator fusion 为什么能提升性能（从内存带宽角度）。
- [ ] 用 `torch.utils.cpp_extension.load()` 加载一个自定义 CUDA kernel。
- [ ] 用 `torch.library` 注册一个 custom op，使其对 `torch.compile` 可见。
- [ ] FlexAttention 的 `score_mod` 和 `block_mask` 分别解决什么问题？
