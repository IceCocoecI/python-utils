# 05 · 并行策略全景：从 DP 到 3D 并行

> 目标：理解训练超大模型（100B+）所需的全部并行策略——
> Data / Tensor / Pipeline / Expert / Sequence Parallelism，以及它们的组合（3D 并行）。
> 这是分布式训练的"屠龙之术"，理解原理后，读懂 Megatron-LM 的代码不再是难事。

---

## 1. 为什么一种并行不够？

### 1.1 数据并行的天花板

DDP / FSDP / ZeRO 本质都是**数据并行 (Data Parallelism)**——
每张卡处理不同的数据，模型（或分片）相同。

数据并行的限制：
- **单层太大**：即使 ZeRO-3 能分片参数，forward 时还是要在单卡上临时聚合完整的一层。
  如果某一层的参数 + 激活就超过单卡显存，数据并行救不了。
- **通信瓶颈**：ZeRO-3 每次 forward/backward 都要 All-Gather 和 Reduce-Scatter，
  当模型极大时通信量爆炸。

本章对应可运行示例：

```bash
cd aigc-learning/05-distributed-training/examples
conda run -n aigc python parallelism_planner.py --total-gpus 128 --tp 8 --pp 4
```

这个示例不模拟 Megatron-LM，而是验证 `总 GPU 数 = DP × TP × PP × EP` 的组合关系，并输出每个维度的设计提醒。TP/PP/EP 的真实性能必须在对应硬件拓扑上 benchmark，不能只靠公式判断。

### 1.2 超大模型需要多维并行

```
                    LLaMA-3 405B 训练方案
                    ├── DP = 数据并行（不同数据）
                    ├── TP = 张量并行（切分矩阵）
                    └── PP = 流水线并行（切分层）

                    DeepSeek-V3 671B MoE
                    ├── DP × TP × PP
                    └── EP = 专家并行（切分 expert）
```

| 并行方式 | 切分维度 | 通信模式 | 节省内容 |
|---|---|---|---|
| Data Parallelism (DP) | 数据 | All-Reduce / Reduce-Scatter | 不省显存（ZeRO 除外） |
| Tensor Parallelism (TP) | 层内矩阵 | All-Reduce / All-Gather | 每层的参数 + 激活 |
| Pipeline Parallelism (PP) | 层间 | P2P Send/Recv | 总参数 + 激活 |
| Expert Parallelism (EP) | MoE 专家 | All-to-All | 专家参数 |
| Sequence Parallelism (SP) | 序列长度 | All-Gather / Reduce-Scatter | 激活 |

---

## 2. Data Parallelism (DP)

前几章已经详细讲过，这里总结核心要点：

```
GPU 0: data_0 → model → grad_0 ─┐
GPU 1: data_1 → model → grad_1 ─┤ All-Reduce → avg_grad → update
GPU 2: data_2 → model → grad_2 ─┤
GPU 3: data_3 → model → grad_3 ─┘
```

**优点**：
- 实现简单（DDP 几行代码）
- 线性加速比接近理想
- 通信量与模型大小成正比，与数据无关

**缺点**：
- 每卡需要能容纳完整模型（DDP）或完整一层（ZeRO-3/FSDP）
- 不能降低单条数据的 latency

**适用**：模型放得下单卡或几张卡（< 30B），主要瓶颈是数据量大。

---

## 3. Tensor Parallelism (TP)

### 3.1 核心思想

把单个矩阵运算切分到多张卡上。最经典的实现来自 **Megatron-LM**。

以一个 Linear 层 `Y = XA` 为例（X 形状 `[B, d]`，A 形状 `[d, h]`）：

**列切分 (Column Parallel)**：

```
A 按列切成 2 份：
A = [A₁ | A₂]

GPU 0: Y₁ = X @ A₁    shape: [B, h/2]
GPU 1: Y₂ = X @ A₂    shape: [B, h/2]

结果：Y = [Y₁ | Y₂]   shape: [B, h]
```

**行切分 (Row Parallel)**：

```
A 按行切成 2 份，X 也相应切分：
A = [A₁]    X = [X₁ | X₂]
    [A₂]

GPU 0: Z₁ = X₁ @ A₁   shape: [B, h]
GPU 1: Z₂ = X₂ @ A₂   shape: [B, h]

结果：Y = Z₁ + Z₂      (All-Reduce)
```

### 3.2 Transformer 中的 TP

Megatron-LM 把 Transformer 的 MLP 和 Attention 层做了精巧的 TP 设计：

```
MLP 层：
  input
    │
    ├─ Column Parallel ─→ Linear(d, 4d/tp) → GELU    （每卡 4d/tp）
    │
    ├─ Row Parallel ────→ Linear(4d/tp, d) → All-Reduce → output
    │

Attention 层：
  input
    │
    ├─ Column Parallel ─→ Q, K, V 各 (d, d/tp)  （每卡 n_heads/tp 个头）
    │
    ├─ Attention 计算（每卡独立）
    │
    ├─ Row Parallel ────→ O_proj(d/tp, d) → All-Reduce → output
    │
```

**关键洞察**：Column Parallel 的输出可以直接喂给 Row Parallel，中间不需要通信。
只在 Row Parallel 的输出做一次 All-Reduce。这样一个 Transformer Block 只需 **2 次 All-Reduce**。

### 3.3 通信分析

| 操作 | 通信量 | 发生次数 (per block) |
|---|---|---|
| All-Reduce (MLP) | 2 × B × seq × d | 1 |
| All-Reduce (Attn) | 2 × B × seq × d | 1 |

通信量与 `tp_size` 无关，但 **All-Reduce 延迟** 与 `tp_size` 正相关。

### 3.4 TP 的要求

- **高带宽互联**：TP 的 All-Reduce 在每个 micro-batch 的每层都要执行。
  必须用 NVLink / NVSwitch（300–900 GB/s），PCIe（~64 GB/s）会成为严重瓶颈。
- **通常 TP 只在单机内做**（同一台机器的 GPU 之间走 NVLink）。
- 模型参数数量需要能被 `tp_size` 整除（特别是 `n_heads` 和 `hidden_dim`）。

### 3.5 PyTorch 原生 TP

PyTorch 2.x 引入了 `torch.distributed.tensor`（DTensor）支持原生 TP：

```python
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

parallelize_module(
    model.layers[0].mlp,
    device_mesh["tp"],
    {
        "gate_proj": ColwiseParallel(),
        "down_proj": RowwiseParallel(),
    },
)
```

---

## 4. Pipeline Parallelism (PP)

### 4.1 核心思想

把模型的层拆分到不同 GPU 上，数据像"流水线"一样流过：

```
Stage 0 (GPU 0): Layer 0–5
Stage 1 (GPU 1): Layer 6–11
Stage 2 (GPU 2): Layer 12–17
Stage 3 (GPU 3): Layer 18–23
```

```
简单 PP（无 micro-batch）：
GPU 0: [  fwd  ][              idle              ][  bwd  ]
GPU 1:          [  fwd  ][       idle       ][  bwd  ]
GPU 2:                   [  fwd  ][ idle ][  bwd  ]
GPU 3:                            [ fwd  ][  bwd  ]

                     大量空闲时间 = "气泡" (bubble)
```

### 4.2 Micro-batching 减少气泡

**GPipe** 方案：把一个 mini-batch 切成 M 个 micro-batch，流水线调度：

```
GPipe (M=4 micro-batches, 4 stages):

GPU 0: [f1][f2][f3][f4][        ][b4][b3][b2][b1]
GPU 1:     [f1][f2][f3][f4][    ][b4][b3][b2][b1]
GPU 2:         [f1][f2][f3][f4][][b4][b3][b2][b1]
GPU 3:             [f1][f2][f3][f4][b4][b3][b2][b1]

气泡比例 ≈ (P-1) / M    P=stages, M=micro-batches
```

当 M >> P 时，气泡趋近于 0。

### 4.3 1F1B 调度

**1F1B (One Forward One Backward)**：交替执行 forward 和 backward，减少峰值显存。

```
1F1B Schedule (M=4, P=4):

GPU 0: [f1][f2][f3][f4][b1][b2][b3][b4]
GPU 1:     [f1][f2][f3][b1][f4][b2][b3][b4]
GPU 2:         [f1][f2][b1][f3][b2][f4][b3][b4]
GPU 3:             [f1][b1][f2][b2][f3][b3][f4][b4]
```

**优势**：同一时刻只需保存少量 micro-batch 的激活，峰值显存大幅下降。

### 4.4 PP 的通信

| 通信 | 类型 | 频率 |
|---|---|---|
| 层间激活传递 | P2P Send/Recv | 每个 micro-batch 每个 stage 边界 |
| 通信量 | B × seq × d (一个张量) | 远小于 DP 的 All-Reduce |

**PP 的通信量远小于 TP**——适合跨机器（节点间带宽较低）。

### 4.5 PP 的局限

- **负载均衡难**：每个 stage 的计算量要尽量相同，否则最慢的 stage 成为瓶颈。
  Embedding 和 LM Head 通常较轻，要注意分配。
- **气泡不可避免**：即使 1F1B，M/P 比例小时气泡仍然可观。
- **代码复杂度高**：需要管理 micro-batch 调度、激活传递、梯度反传。

---

## 5. Expert Parallelism (EP)

### 5.1 MoE 架构回顾

Mixture-of-Experts (MoE) 模型的 FFN 层包含多个"专家"，每个 token 只路由到 top-k 个专家：

```
                input token
                    │
                  Router
               ┌────┴────┐
           Expert 0   Expert 1   Expert 2   Expert 3
           (active)              (active)
               └────┬────┘
               Combine output
```

### 5.2 EP 的做法

把不同的专家放在不同的 GPU 上：

```
GPU 0: Expert 0, Expert 1
GPU 1: Expert 2, Expert 3
GPU 2: Expert 4, Expert 5
GPU 3: Expert 6, Expert 7
```

每个 token 需要路由到对应专家所在的 GPU → **All-to-All 通信**。

```
Token 分发（All-to-All）：
GPU 0 上的 token 去 Expert 5 → 发给 GPU 2
GPU 2 上的 token 去 Expert 1 → 发给 GPU 0

计算后再通过 All-to-All 把结果发回来。
```

### 5.3 通信分析

All-to-All 通信量 = `B × seq × top_k × d / ep_size`

**特点**：
- 所有 GPU 之间都需要通信（不是邻居间的 P2P）
- 路由不均匀时，部分 GPU 负载重
- 需要 load balancing loss 来鼓励均匀路由

### 5.4 实际使用

MoE 的 EP 通常和 DP + TP 组合：

```
DeepSeek-V3 (671B, 256 experts):
  EP_size = 64     每组 4 个专家在一张 GPU
  TP_size = 4      注意力层用 TP
  DP_size = 8      数据并行
  总 GPU = 64 × 4 × 8 = 2048
```

---

## 6. Sequence Parallelism (SP)

### 6.1 为什么需要 SP？

TP 切分了参数，但 LayerNorm 和 Dropout 等操作的激活仍然是完整的。
当序列长度很长时（32K、128K、1M），激活值的显存可能超过参数显存。

SP 把序列维度切分到多张卡上。

### 6.2 Megatron-SP

与 TP 配合使用，在 TP 区域内（All-Reduce 的输入/输出）沿序列维度分片：

```
标准 TP:
  LayerNorm(全 seq) → TP_region(全 seq) → All-Reduce → Dropout(全 seq)

Megatron-SP:
  LayerNorm(seq/tp) → All-Gather → TP_region(全 seq) → Reduce-Scatter → Dropout(seq/tp)
```

**效果**：TP 区域外的激活从 `[B, seq, d]` 降到 `[B, seq/tp, d]`。

### 6.3 Ring Attention

用于超长序列（100K+）。每张卡持有一段序列的 Q，K/V 在卡之间以环形传递：

```
Ring Attention (4 GPUs, seq=16K → 每卡 4K):

GPU 0: Q₀ × K₀,V₀ → send K₀,V₀ to GPU 1, recv K₃,V₃
       Q₀ × K₃,V₃ → send K₃,V₃ to GPU 1, recv K₂,V₂
       Q₀ × K₂,V₂ → send K₂,V₂ to GPU 1, recv K₁,V₁
       Q₀ × K₁,V₁ → done, combine all partial attentions
```

**优点**：可以处理任意长的序列，显存和序列长度线性关系。
**通信**：每步 P2P 传 KV，可以和计算重叠。

### 6.4 DeepSpeed Ulysses

另一种 SP 方案：把 Q、K、V 沿序列维度切分，用 All-to-All 通信：

```
Ulysses SP (tp_size=4, seq=16K):

切分 Q/K/V:   每卡 [B, seq/4, n_heads, d_head]
All-to-All:   每卡 [B, seq, n_heads/4, d_head]  （变成按 head 切分）
各卡独立做 Attention
All-to-All:   恢复按 seq 切分
```

**优点**：通信一步到位（2 次 All-to-All），不需要像 Ring Attention 那样多轮传递。
**缺点**：All-to-All 需要高带宽互联。

---

## 7. 3D 并行：DP × TP × PP

训练 100B+ 模型的标准方案是组合多种并行。以 LLaMA-3 405B 为例：

### 7.1 架构

```
              ┌─────────── PP Stage 0 ──────────┐ ┌── PP Stage 1 ──┐  ...
              │                                  │ │                │
  DP 0: [GPU 0,1,2,3] (TP=4)   →  P2P  → [GPU 16,17,18,19]  →  ...
  DP 1: [GPU 4,5,6,7] (TP=4)   →  P2P  → [GPU 20,21,22,23]  →  ...
  DP 2: [GPU 8,9,10,11] (TP=4) →  P2P  → [GPU 24,25,26,27]  →  ...
  DP 3: [GPU 12,13,14,15] (TP=4) → P2P → [GPU 28,29,30,31]  →  ...
```

### 7.2 各维度的分工

| 维度 | 切分什么 | 通信类型 | 带宽要求 | 通常放在 |
|---|---|---|---|---|
| TP | 层内矩阵 | All-Reduce | 极高 | **单机内** (NVLink) |
| PP | 层间 | P2P | 中等 | 机器间 |
| DP | 数据 | All-Reduce | 中等 | 最外层 |

### 7.3 关键设计原则

**① TP 在节点内**：因为 All-Reduce 频率高、对带宽敏感，必须用 NVLink。

**② PP 跨节点**：P2P 通信量相对小，可以走节点间网络（InfiniBand）。

**③ DP 在最外层**：DP 的 All-Reduce 只在每个 step 结束时做一次，对带宽要求最低。

**④ 总 GPU 数 = DP × TP × PP**：

```
假设 128 张 GPU:
  TP = 8  (单机 8 卡 NVLink)
  PP = 4  (4 个流水线 stage)
  DP = 4  (4 路数据并行)
  Total = 8 × 4 × 4 = 128
```

### 7.4 LLaMA-3 405B 的训练配置

根据 Meta 的技术报告，LLaMA-3 405B 使用了：

| 参数 | 值 |
|---|---|
| 总 GPU | 16,384 × H100 |
| TP | 8（单机内） |
| PP | 16 |
| DP | 128 |
| Context Parallelism | 用于 128K 长序列 |
| Batch Size | 约 16M tokens/step |

---

## 8. Context Parallelism

PyTorch 2.4+ 引入了 Context Parallelism (CP)，专门为长上下文训练设计：

```python
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
)
```

CP 将输入序列在 context 维度切分，配合 Ring Attention 或类似机制完成注意力计算。

**适用**：训练长上下文模型（128K+ tokens），如 LLaMA-3 的 128K 版本。

---

## 9. 工具：Megatron-LM 与 Megatron-Core

### 9.1 Megatron-LM

NVIDIA 的 3D 并行训练框架，是 LLaMA、GPT-NeoX 等大模型的训练底座。

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
```

特点：
- 原生支持 TP + PP + DP + SP + EP
- 高度优化的 CUDA kernel
- 完整的数据处理流水线
- 学习曲线陡峭，代码量大

### 9.2 Megatron-Core

Megatron-LM 的核心并行原语，解耦为独立库：

```python
from megatron.core import parallel_state
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.pipeline_parallel import get_forward_backward_func
```

特点：
- 可以集成到自己的训练框架中
- 提供 TP / PP / SP / EP 的基础组件
- 比完整 Megatron-LM 更容易上手

### 9.3 `torch.distributed.tensor` (DTensor)

PyTorch 原生的张量并行支持，未来方向：

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))
```

用 DeviceMesh 可以同时定义 DP + TP 的通信拓扑。

---

## 10. 决策框架

### 10.1 按模型规模选方案

| 模型规模 | 推荐方案 | 说明 |
|---|---|---|
| < 1B | DDP | 单卡或几卡搞定 |
| 1B – 13B | DDP 或 FSDP | ZeRO-2/3 也行 |
| 13B – 70B | FSDP / ZeRO-3 | 可能需要 TP=2 或 4 |
| 70B – 200B | TP + FSDP 或 TP + PP + DP | 需要多机 |
| 200B+ | 3D 并行 (TP + PP + DP) | Megatron-LM 级别 |
| MoE 100B+ | 3D + EP | 如 DeepSeek、Mixtral |

### 10.2 按硬件选方案

| 硬件 | TP 上限 | PP 建议 | DP 策略 |
|---|---|---|---|
| 单机 4 × 4090 (PCIe) | TP=1（PCIe 太慢） | 不建议 | DDP/FSDP |
| 单机 8 × A100 (NVLink) | TP=8 | 通常不需要 | DDP/FSDP |
| 多机 × 8 A100/H100 | TP=8（机内） | PP=N（跨机） | FSDP/ZeRO |

### 10.3 成本模型

```
总训练时间 ∝ (计算时间 + 通信时间) / 并行效率

计算时间 ∝ 6 × 参数量 × token数 / (GPU数 × GPU算力)

通信开销:
  DP:  2 × model_size per step（All-Reduce）
  TP:  2 × hidden_size × seq_len × batch per layer（All-Reduce）
  PP:  hidden_size × seq_len × batch per micro-batch（P2P）
```

| 场景 | 计算/通信比 | 策略 |
|---|---|---|
| 计算密集（大 batch，长 seq） | 高 | 可以容忍更多通信 → TP + PP |
| 通信密集（小 batch，短 seq） | 低 | 减少通信 → 纯 DP/FSDP |

---

## 11. 实际案例

### 11.1 LLaMA-2 70B 训练

```
硬件：2000 × A100 80GB
TP = 8, PP = 4, DP = ~62
训练数据：2T tokens
训练时间：~170 万 A100 小时
```

### 11.2 DeepSeek-V3 671B MoE

```
硬件：2048 × H800
TP = ?, PP = ?, DP = ?, EP = ?
训练数据：14.8T tokens
训练时间：2.788M H800 小时
特点：使用 FP8 混合精度训练，大幅节省显存和通信带宽
```

### 11.3 你的 7B 微调项目

```
硬件：8 × A100 80GB (单机)
方案：FSDP (FULL_SHARD) + bf16 + activation checkpointing
等效于 ZeRO-3，每卡显存充裕
不需要 TP/PP——模型不够大
```

```
硬件：4 × 4090 24GB (单机, PCIe)
方案：DeepSpeed ZeRO-3 + CPU Offload + activation checkpointing
4090 没有 NVLink，不适合 TP
ZeRO-3 + CPU Offload 勉强可以全量训练
推荐 QLoRA 微调（更快更省）
```

---

## 12. 通信原语速查

| 原语 | 操作 | 输入 → 输出 | 用途 |
|---|---|---|---|
| `All-Reduce` | 聚合 + 广播 | 每卡 [N] → 每卡 [N] (聚合值) | DP 梯度同步, TP 输出聚合 |
| `All-Gather` | 拼接 + 广播 | 每卡 [N/P] → 每卡 [N] | FSDP 参数收集, SP 序列恢复 |
| `Reduce-Scatter` | 聚合 + 分片 | 每卡 [N] → 每卡 [N/P] (聚合值) | FSDP 梯度同步, Megatron-SP |
| `All-to-All` | 转置 + 分发 | 每卡 [P × N/P] → 每卡 [P × N/P] (转置) | EP token 路由, Ulysses SP |
| `P2P Send/Recv` | 点对点 | GPU_i → GPU_j | PP 激活传递 |
| `Broadcast` | 从一个到所有 | 一卡 [N] → 每卡 [N] | 参数初始化同步 |

---

## 13. 常见坑

| 坑 | 说明 |
|---|---|
| TP 跨机器 | NVLink 带宽 >> InfiniBand 带宽，TP 跨机会极慢 |
| PP 层分配不均 | 某个 stage 计算量远大于其他 → 流水线气泡增大 |
| DP 忘了 scale 学习率 | 全局 batch 变大，LR 要相应调整 |
| EP 路由不均衡 | 某些 expert 过载、某些空闲 → 需要 load balancing loss |
| 混用不同框架 | Megatron 的 TP + DeepSpeed 的 ZeRO → 版本兼容性问题 |
| 通信 overlap 失效 | 不当的 `torch.cuda.synchronize()` 会阻塞重叠 |

---

## 小结

| 并行方式 | 切什么 | 通信 | 带宽要求 | 典型位置 |
|---|---|---|---|---|
| DP | 数据 | All-Reduce | 中 | 最外层 |
| TP | 矩阵 | All-Reduce | 极高 | 机内 NVLink |
| PP | 层 | P2P | 低 | 跨机 |
| EP | 专家 | All-to-All | 高 | 按拓扑分 |
| SP | 序列 | All-Gather / RS | 高 | 随 TP |

**记住三条原则**：
1. TP 放机内（NVLink），PP 放跨机（InfiniBand），DP 放最外层。
2. 能用简单方案就不用复杂方案——先试 DDP/FSDP，不够再加 TP/PP。
3. `GPU 总数 = DP × TP × PP × EP`——根据硬件和模型规模分配各维度。

恭喜你完成了分布式训练模块的全部学习！
现在你已经具备了理解和参与大模型训练工程的理论基础。
