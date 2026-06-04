# 00 · 分布式训练理论框架

> 目标：先理解分布式训练的约束系统，再学习 DDP、FSDP、Accelerate、DeepSpeed 和 3D 并行。
> 工具会变，但核心问题始终是显存、计算、通信、同步、拓扑之间的权衡。

---

## 1. 分布式训练的本质

单卡训练可以简化成：

```
data -> forward -> loss -> backward -> optimizer.step()
```

分布式训练不是简单地“多开几个进程”，而是把这条链路拆到多个设备上执行，同时保证：

- 每个进程拿到正确的数据切片
- 梯度或参数在正确时机同步
- 优化器更新语义等价于目标训练算法
- checkpoint、日志、评估不会互相冲突
- 通信开销不会吞掉并行收益

一个分布式方案是否成立，取决于四个问题：

| 问题 | 典型答案 |
|---|---|
| 切什么？ | 数据、参数、梯度、优化器状态、层、矩阵、序列、专家 |
| 怎么通信？ | All-Reduce、All-Gather、Reduce-Scatter、All-to-All、P2P |
| 省什么？ | 显存、训练时间、单步延迟、人工工程复杂度 |
| 付出什么？ | 通信、同步、复杂度、调试成本、数值风险 |

---

## 2. 四个基础资源模型

### 2.1 显存模型

全量训练一个参数量为 `P` 的模型，AdamW + bf16 常见显存近似：

```
参数：P × 2 bytes
梯度：P × 2 bytes
优化器状态：P × 12 bytes   # fp32 master weight + exp_avg + exp_avg_sq
激活值：取决于 batch、seq_len、hidden、层数
通信 buffer / 碎片：额外开销
```

所以不含激活时，每个参数大约需要：

```
2 + 2 + 12 = 16 bytes / parameter
```

7B 模型全量训练只算参数、梯度、优化器状态就约 `7B × 16 = 112GB`（十进制），这就是为什么“7B 微调也可能需要分布式”。

### 2.2 计算模型

Transformer 训练常用近似：

```
训练 FLOPs ≈ 6 × 参数量 × token 数
```

并行后理想情况：

```
计算时间 ≈ 总 FLOPs / (GPU 数 × 单 GPU 有效算力)
```

现实中还要乘以并行效率。通信、等待、load imbalance、kernel 效率都会降低并行效率。

### 2.3 通信模型

通信时间可以粗略拆成：

```
通信时间 ≈ 启动延迟 × 通信次数 + 通信字节数 / 带宽
```

这解释了两个常见现象：

- 小张量频繁通信很慢，因为延迟占主导。
- 大模型跨机器通信很慢，因为带宽占主导。

### 2.4 拓扑模型

GPU 之间不是“全都一样快”：

| 连接 | 典型带宽特征 | 适合 |
|---|---|---|
| NVLink / NVSwitch | 高带宽、低延迟 | TP、频繁 All-Reduce |
| PCIe | 明显较慢 | DDP/FSDP，小规模通信 |
| InfiniBand / RoCE | 跨机高带宽 | PP、DP、FSDP/ZeRO 多机 |
| 普通以太网 | 带宽低、延迟高 | 调试，不适合大规模训练 |

并行策略必须尊重拓扑。比如 TP 每层都通信，通常应该放在单机 NVLink 内；PP 只传层间激活，跨机器更可接受。

---

## 3. 通信原语是分布式训练的指令集

| 原语 | 输入/输出 | 典型用途 |
|---|---|---|
| Broadcast | 一份数据复制给所有 rank | 初始化参数、同步配置 |
| All-Reduce | 所有 rank 的张量聚合后返回给所有 rank | DDP 梯度平均 |
| All-Gather | 每个 rank 的分片拼成完整张量并返回给所有 rank | FSDP 参数收集 |
| Reduce-Scatter | 聚合完整张量后每个 rank 只拿一片 | FSDP 梯度分片 |
| All-to-All | 每个 rank 给每个 rank 发送一片 | MoE 专家路由、Ulysses SP |
| Send/Recv | 点对点传输 | Pipeline stage 间传激活 |
| Barrier | 所有 rank 等待到同一点 | 保存 checkpoint 前后同步 |

DDP、FSDP、ZeRO、TP、PP 本质上都是这些原语的不同组合。

---

## 4. 数据并行：复制模型，切分数据

Data Parallelism 的语义：

```
rank 0: batch_0 -> model -> grad_0
rank 1: batch_1 -> model -> grad_1
...
All-Reduce average gradients
每个 rank 执行相同 optimizer.step()
```

DDP 的核心优点是语义简单、性能成熟。它的限制也很直接：每张卡仍然保存完整模型、完整梯度和完整优化器状态。

DDP 适合：

- 模型能放进单卡
- 主要想提升吞吐
- 训练代码还在快速迭代
- 需要最容易 debug 的分布式 baseline

---

## 5. ZeRO/FSDP：切分冗余状态

DDP 的冗余来自每张卡保存同一份状态：

```
参数 P、梯度 G、优化器状态 OS
```

ZeRO/FSDP 的思路是把这些状态按 rank 分片：

| 策略 | 参数 | 梯度 | 优化器状态 | 等价理解 |
|---|---|---|---|---|
| DDP / NO_SHARD | 全量 | 全量 | 全量 | 最简单，最占显存 |
| ZeRO-1 | 全量 | 全量 | 分片 | 省优化器 |
| ZeRO-2 / SHARD_GRAD_OP | 全量 | 分片 | 分片 | 再省梯度 |
| ZeRO-3 / FULL_SHARD | 分片 | 分片 | 分片 | 最省显存，通信最多 |

FSDP/ZeRO-3 的关键代价：

```
forward 前需要 All-Gather 参数
backward 后需要 Reduce-Scatter 梯度
```

所以它不是“免费省显存”。当模型不大时，DDP 可能更快；当模型放不下时，FSDP/ZeRO 才是必要选择。

---

## 6. Tensor Parallelism：切分层内矩阵

当单层太大，即使用 ZeRO-3 临时聚合完整层也放不下，就需要 Tensor Parallelism。

TP 切的是矩阵乘法：

- Column Parallel：按输出维切权重，每张卡产出一部分 hidden。
- Row Parallel：按输入维切权重，各卡结果需要 All-Reduce 求和。

TP 的特点：

- 能降低单层参数和激活显存。
- 每个 Transformer block 内都有通信。
- 对带宽极其敏感，通常要求 NVLink/NVSwitch。

TP 是大模型预训练中的核心技术，但不适合在普通 PCIe 多卡上随便开启。

---

## 7. Pipeline Parallelism：切分层间深度

PP 把模型层切到不同 stage：

```
stage 0: layers 0-7
stage 1: layers 8-15
stage 2: layers 16-23
```

它传的是 stage 边界的激活，通信频率低于 TP，适合跨机器。但 PP 有气泡：

```
bubble ratio ≈ (pipeline_stages - 1) / micro_batches
```

要减少气泡，需要更多 micro-batch 或更好的调度（如 1F1B），同时要保证每个 stage 计算量接近。

---

## 8. Expert 和 Sequence Parallelism

Expert Parallelism 用于 MoE：

- 不同 expert 放到不同 GPU。
- token 根据 router 发送到对应 expert。
- 核心通信是 All-to-All。
- 关键风险是 expert 负载不均衡。

Sequence Parallelism 用于长上下文：

- 沿序列维度切激活。
- 减少长序列下 LayerNorm、Dropout、Attention 相关激活显存。
- 常和 TP 或 Ring Attention / Ulysses 一起使用。

这两类并行通常是超大模型或长上下文模型才需要的高级方案。

---

## 9. 3D 并行：组合维度

大规模训练通常组合：

```
总 GPU 数 = DP × TP × PP × EP
```

经验原则：

1. TP 放机内，因为它通信频繁且需要高带宽。
2. PP 可以跨机，因为它主要传 stage 边界激活。
3. DP 放最外层，用于提升吞吐和扩展数据量。
4. EP 只在 MoE 中使用，必须关注 All-to-All 拓扑。

一个 128 GPU 例子：

```
TP = 8
PP = 4
DP = 4
Total = 8 × 4 × 4 = 128
```

这不是固定答案，而是一个拓扑友好的起点。

---

## 10. 训练语义与全局 batch

分布式训练会改变有效 batch：

```
global_batch = per_device_batch × world_size × gradient_accumulation_steps
```

这会影响：

- 学习率
- warmup 步数
- loss scale
- evaluation cadence
- checkpoint cadence

很多“分布式训练不收敛”的问题不是通信错了，而是全局 batch 改了但优化策略没改。

---

## 11. Debug 思维模型

分布式问题通常分四类：

| 类型 | 现象 | 排查方向 |
|---|---|---|
| 启动问题 | torchrun 起不来、端口占用、rank 不匹配 | rendezvous、MASTER_ADDR、MASTER_PORT |
| 同步问题 | hang、timeout | 是否所有 rank 走同一路径、collective 次数是否一致 |
| 数值问题 | loss 爆炸、不收敛 | LR、global batch、混合精度、梯度裁剪 |
| 性能问题 | 多卡比单卡慢 | batch 太小、通信过多、拓扑不匹配、I/O 瓶颈 |

调试顺序：

1. 先用 CPU + gloo 跑通逻辑。
2. 再用单机 2 GPU + NCCL。
3. 再扩到单机多 GPU。
4. 最后做多机。

不要直接在多机 64 卡上 debug 新训练脚本。

---

## 12. 本模块示例如何对应理论

| 理论问题 | 对应示例 |
|---|---|
| collective 到底做了什么？ | `examples/collectives_demo.py` |
| DDP 如何切数据、同步梯度、保存 checkpoint？ | `examples/ddp_cpu_train.py` |
| Accelerate 封装了哪些分布式细节？ | `examples/accelerate_train.py` |
| FSDP/ZeRO 省了多少参数/梯度/优化器显存？ | `examples/fsdp_memory_math.py` |
| DeepSpeed JSON 怎么表达 ZeRO 策略？ | `examples/deepspeed_config_builder.py` |
| DP/TP/PP/EP 怎么组合成总 GPU 数？ | `examples/parallelism_planner.py` |

当前 CPU 环境能验证通信语义、DDP 训练循环、Accelerate 编排和显存公式；不能验证 NCCL 性能、GPU FSDP 真实显存峰值或 DeepSpeed kernel 性能。这些需要真实多 GPU 环境。

---

## 小结

分布式训练的核心不是某个库，而是五个约束：

- 显存：单卡是否放得下模型、梯度、优化器、激活
- 计算：GPU 是否被喂满
- 通信：通信次数和通信字节是否可接受
- 同步：所有 rank 是否执行一致的 collective 语义
- 拓扑：并行维度是否匹配硬件连接

后续章节分别展开这些约束在 DDP、FSDP、Accelerate、DeepSpeed 和 3D 并行中的具体实现。
