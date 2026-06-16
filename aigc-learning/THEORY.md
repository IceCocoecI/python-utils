# AIGC 理论知识地图

> 本文件是整个 `aigc-learning` 的理论导航入口。
> 它不替代各模块教程，而是帮你按“概念主线”复习：先建立心智模型，再进入代码、框架和工程实践。

---

## 1. 为什么需要单独的理论导航？

当前仓库的主体是“边学边跑”的实践教程：每章讲一个工具、框架或工程场景，并配套示例代码。
这种组织方式适合动手，但在复习理论时会遇到三个问题：

1. 同一个概念会跨模块出现，例如 Attention 同时出现在 PyTorch、Transformer、推理优化、CUDA kernel 中。
2. 有些底层理论被夹在实践文档里，例如 RAG 评估、Agent 架构、GPU Roofline。
3. 面试或排查问题时，往往需要按概念检索，而不是按工具检索。

因此建议采用“两层结构”：

| 层次 | 作用 | 典型文件 |
|---|---|---|
| 理论入口 | 建立概念地图、术语、设计取舍、常见误区 | `THEORY.md`、各模块 `00-...-theory.md` |
| 实践教程 | 展示 API、代码模板、运行命令、工程经验 | `01-...md`、`examples/` |

---

## 2. 理论阅读主线

如果目标是系统补理论，推荐按下面顺序阅读。

### 主线 0：Python 工程基础

1. [01 · Python 工程理论框架](./01-python-foundations/00-python-engineering-theory.md)
2. [01 · 现代 Python 基础](./01-python-foundations/01-modern-python-basics.md)
3. [01 · 装饰器 / 生成器 / 上下文管理器](./01-python-foundations/02-advanced-features.md)
4. [01 · 类型注解与 Pydantic](./01-python-foundations/04-type-hints.md)
5. [01 · 工程化最佳实践](./01-python-foundations/05-engineering-best-practices.md)

这条线回答的是：

- 如何用数据结构、类型、协议和上下文管理器表达工程边界？
- 生成器、async、装饰器分别适合解决哪类问题？
- 测试、日志、异常、profiling 如何让算法代码可维护？

### 主线 A：深度学习与生成模型

1. [02 · 深度学习核心理论](./02-deep-learning-libraries/00-deep-learning-theory.md)
2. [02 · Transformer 原理白话速览](./02-deep-learning-libraries/06-transformer-principles-overview.md)
3. [02 · Transformer 架构深度剖析](./02-deep-learning-libraries/05-transformer-from-scratch.md)
4. [06 · 微调与对齐理论框架](./06-finetuning-and-alignment/00-finetuning-and-alignment-theory.md)
5. [09 · 前沿 AIGC 模型理论框架](./09-frontier-models/00-frontier-models-theory.md)
6. 模块 09 的模型架构专题：
   [LLM](./09-frontier-models/01-llm-architectures.md)、
   [图像生成](./09-frontier-models/02-image-generation.md)、
   [多模态](./09-frontier-models/03-multimodal-models.md)

你应该重点建立这些概念之间的关系：

```text
Tensor / Autograd / Optimizer
        ↓
Transformer / Diffusion / Flow Matching
        ↓
Pretrain / SFT / Preference Alignment
        ↓
Inference / Serving / Application
```

Transformer 相关内容分层阅读：

| 层次 | 文档 | 作用 |
|---|---|---|
| 概念速览 | [02/06 Transformer 原理白话速览](./02-deep-learning-libraries/06-transformer-principles-overview.md) | 第一遍建立直觉 |
| 公式与实现 | [02/05 Transformer 架构深度剖析](./02-deep-learning-libraries/05-transformer-from-scratch.md) | 手写 Attention、RoPE、mask、KV cache |
| 现代架构 | [09/01 LLM 架构全解](./09-frontier-models/01-llm-architectures.md) | 理解 LLaMA/Qwen/DeepSeek/MoE 等变体 |

### 主线 B：数据、训练与分布式工程

1. [03 · 数据处理与科学计算理论框架](./03-data-and-scientific-computing/00-data-and-scientific-computing-theory.md)
2. [04 · 训练工程化理论框架](./04-training-engineering/00-training-engineering-theory.md)
3. [05 · 分布式训练理论框架](./05-distributed-training/00-distributed-training-theory.md)
4. [07 · 推理与部署理论框架](./07-inference-and-deployment/00-inference-and-deployment-theory.md)

这条线回答的是：

- 数据为什么要被约束成明确的 shape、dtype、layout、range、distribution？
- 训练为什么必须记录 experiment、run、config、artifact？
- DDP、FSDP、ZeRO、TP、PP、EP 分别在切什么？
- 推理系统为什么要区分 prefill、decode、batching、KV cache、调度？

### 主线 C：LLM 应用系统

1. [08 · LLM 应用理论框架](./08-llm-applications/00-llm-applications-theory.md)
2. [08 · RAG 基础](./08-llm-applications/01-rag-fundamentals.md)
3. [08 · 向量数据库](./08-llm-applications/02-vector-databases.md)
4. [08 · 编排框架](./08-llm-applications/03-orchestration-frameworks.md)
5. [08 · Agent 工程](./08-llm-applications/04-agent-engineering.md)

这条线的核心不是“会调 API”，而是理解一个 LLM 应用系统的闭环：

```text
输入 → 上下文构造 → 检索/工具 → 模型生成 → 校验/评估 → 反馈改进
```

### 主线 D：性能与 GPU 底层

1. [10 · GPU 性能理论框架](./10-cuda-and-triton/00-gpu-performance-theory.md)
2. [10 · GPU 架构与 CUDA 基础](./10-cuda-and-triton/01-gpu-architecture-and-cuda-basics.md)
3. [10 · Triton 编程](./10-cuda-and-triton/02-triton-programming.md)
4. [10 · 性能分析与优化](./10-cuda-and-triton/03-performance-profiling.md)
5. [10 · 自定义算子与 PyTorch 扩展](./10-cuda-and-triton/04-custom-operators-and-extensions.md)

这条线回答的是：

- 为什么同样的数学公式，写法不同会差数倍性能？
- 为什么 LLM 推理 decode 阶段常常 memory-bound？
- FlashAttention、PagedAttention、operator fusion 本质上在优化什么？
- Profiling 到底应该看 CPU 时间、GPU 时间、内存带宽还是 kernel occupancy？

---

## 3. 模块理论地图

| 模块 | 理论入口 | 实践入口 | 备注 |
|---|---|---|---|
| 01 Python 基础 | [00 理论](./01-python-foundations/00-python-engineering-theory.md) | [01 README](./01-python-foundations/README.md) | 数据结构、协议、生成器、async、类型、测试、profiling |
| 02 深度学习核心库 | [00 理论](./02-deep-learning-libraries/00-deep-learning-theory.md) | [02 README](./02-deep-learning-libraries/README.md) | 已包含深度学习底座与 Transformer 速览 |
| 03 数据与科学计算 | [00 理论](./03-data-and-scientific-computing/00-data-and-scientific-computing-theory.md) | [03 README](./03-data-and-scientific-computing/README.md) | shape / dtype / layout / range / distribution 是主线 |
| 04 训练工程化 | [00 理论](./04-training-engineering/00-training-engineering-theory.md) | [04 README](./04-training-engineering/README.md) | experiment / run / config / artifact 是主线 |
| 05 分布式训练 | [00 理论](./05-distributed-training/00-distributed-training-theory.md) | [05 README](./05-distributed-training/README.md) | 从“切数据”到“切参数/激活/计算” |
| 06 微调与对齐 | [00 理论](./06-finetuning-and-alignment/00-finetuning-and-alignment-theory.md) | [06 README](./06-finetuning-and-alignment/README.md) | PEFT、量化、SFT、偏好优化 |
| 07 推理与部署 | [00 理论](./07-inference-and-deployment/00-inference-and-deployment-theory.md) | [07 README](./07-inference-and-deployment/README.md) | latency / throughput / batching / KV cache |
| 08 LLM 应用 | [00 理论](./08-llm-applications/00-llm-applications-theory.md) | [08 README](./08-llm-applications/README.md) | RAG、向量检索、编排、Agent、安全评估 |
| 09 前沿模型 | [00 理论](./09-frontier-models/00-frontier-models-theory.md) | [09 README](./09-frontier-models/README.md) | 跨模态生成模型、token/latent/patch、Scaling 与架构选型 |
| 10 CUDA 与 Triton | [00 理论](./10-cuda-and-triton/00-gpu-performance-theory.md) | [10 README](./10-cuda-and-triton/README.md) | GPU 执行模型、Roofline、memory-bound 优化 |

---

## 4. 理论文档的阅读方法

每个 `00-...-theory.md` 建议按四步使用：

1. 先读“本模块解决什么问题”，确认学习目标。
2. 读核心概念图，建立整体结构。
3. 对照实践文档运行示例，确认理论能解释代码。
4. 用自检问题复述，不要只停留在“看懂”。

一个判断标准：读完理论后，应该能解释“为什么这么设计”，而不仅是“怎么调用 API”。

---

## 5. 推荐的理论文档模板

后续新增理论文档时，建议统一采用这个结构：

```text
# 00 · xxx 理论框架

## 1. 本模块真正解决什么问题？
## 2. 核心概念图
## 3. 关键机制 / 公式 / 心智模型
## 4. 主要设计取舍
## 5. 常见误区
## 6. 如何映射到本模块代码和实践文档
## 7. 自检问题
```

这个模板的边界很重要：

- 理论文档讲概念、机制、取舍、排错判断。
- 实践文档讲 API、代码、命令、工具配置。
- `examples/` 负责把知识点落成可以运行的小脚本。

---

## 6. 当前补齐状态

第一阶段补齐：

- 模块 08：[LLM 应用理论框架](./08-llm-applications/00-llm-applications-theory.md)
- 模块 10：[GPU 性能理论框架](./10-cuda-and-triton/00-gpu-performance-theory.md)
- 根目录：当前理论总索引

第二阶段补齐：

- 模块 01：[Python 工程理论框架](./01-python-foundations/00-python-engineering-theory.md)
- 模块 09：[前沿 AIGC 模型理论框架](./09-frontier-models/00-frontier-models-theory.md)
- 模块 02/09：Transformer 内容分层与交叉引用
