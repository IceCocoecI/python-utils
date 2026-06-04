# 模块 05：分布式训练

> 单卡放不下、单卡跑太慢——这就是分布式训练的全部动机。
> 本模块带你从 DDP 入门到 3D 并行，覆盖 AIGC 工程师日常训练大模型所需的全部技能。

---

## 为什么这一章重要？

2020 年 GPT-3（175B）发布以来，模型规模增速远超单卡显存增速：

| 年份 | 代表模型 | 参数量 | 训练所需 GPU |
|---|---|---|---|
| 2018 | BERT-Large | 340M | 1×V100 可训 |
| 2020 | GPT-3 | 175B | 数百 A100 |
| 2023 | LLaMA-2 70B | 70B | 数十~百 A100 |
| 2024 | LLaMA-3 405B | 405B | 数千 H100 |
| 2024 | DeepSeek-V3 | 671B (MoE) | 数千 H800 |

即使你"只是"微调一个 7B 模型，在 bf16 下参数就占 14GB，加上优化器状态和激活值，
单张 24GB 的 4090 已经放不下。**分布式训练不是可选项，而是必修课。**

---

## 工具选型速查

> **原则**：能用简单的就不用复杂的。

| 场景 | 推荐方案 | 说明 |
|---|---|---|
| 单机多卡，模型放得下单卡 | **DDP** | 最成熟、最快、最好调试 |
| 单机多卡，模型放不下单卡 | **FSDP** / **DeepSpeed ZeRO-3** | 切分参数+优化器 |
| 多机多卡，中等规模 (7B–70B) | **FSDP** + **Accelerate** | HuggingFace 生态首选 |
| 多机多卡，超大规模 (100B+) | **3D 并行** (DP+TP+PP) | Megatron-LM / Megatron-Core |
| 想快速从单卡代码迁移 | **Accelerate** | 改动最小 |
| HuggingFace Trainer 用户 | **Trainer** + DeepSpeed/FSDP | 配置驱动，零代码改动 |

---

## 学习内容

| # | 文档 | 核心话题 |
|---|---|---|
| 00 | [distributed-training-theory](./00-distributed-training-theory.md) | 分布式训练理论：并行维度、通信原语、显存模型、性能模型 |
| 01 | [distributed-basics-and-ddp](./01-distributed-basics-and-ddp.md) | 分布式核心概念 / DDP / torchrun / 多机 |
| 02 | [fsdp](./02-fsdp.md) | FSDP / FSDP2 / 分片策略 / checkpoint |
| 03 | [accelerate](./03-accelerate.md) | HuggingFace Accelerate / 一键分布式 / DeepSpeed 集成 |
| 04 | [deepspeed](./04-deepspeed.md) | DeepSpeed / ZeRO-1/2/3 / CPU Offload / 推理加速 |
| 05 | [parallelism-strategies](./05-parallelism-strategies.md) | DP / TP / PP / EP / SP / 3D 并行 / 选型决策 |

---

## 示例代码（`examples/`）

当前机器的 `aigc` 环境有 PyTorch 2.6、Accelerate 1.13；CUDA 是编译可用但当前无 GPU，DeepSpeed 未安装。
因此示例分成两类：

| 文件 | 说明 | 当前环境是否可跑 |
|---|---|---|
| [`collectives_demo.py`](./examples/collectives_demo.py) | `all_reduce` / `broadcast` / `all_gather` 集合通信 demo | 是，CPU + gloo |
| [`ddp_cpu_train.py`](./examples/ddp_cpu_train.py) | 两进程 CPU DDP 训练，演示 `DistributedSampler`、rank 0 保存 | 是，CPU + gloo |
| [`accelerate_train.py`](./examples/accelerate_train.py) | Accelerate 训练循环，可单进程或 CPU 多进程启动 | 是 |
| [`fsdp_memory_math.py`](./examples/fsdp_memory_math.py) | DDP / FSDP / ZeRO 显存估算 | 是 |
| [`deepspeed_config_builder.py`](./examples/deepspeed_config_builder.py) | 生成 ZeRO-1/2/3 配置 JSON | 是，不要求安装 DeepSpeed |
| [`parallelism_planner.py`](./examples/parallelism_planner.py) | 计算 DP/TP/PP/EP 组合是否匹配总 GPU 数 | 是 |

### 在当前 `aigc` 环境运行

```bash
cd aigc-learning/05-distributed-training/examples

conda run -n aigc torchrun --standalone --nproc_per_node=2 collectives_demo.py
conda run -n aigc torchrun --standalone --nproc_per_node=2 ddp_cpu_train.py --epochs 1
conda run -n aigc accelerate launch --cpu --num_processes=2 accelerate_train.py --epochs 1

conda run -n aigc python fsdp_memory_math.py --params-billion 7 --world-size 4
conda run -n aigc python deepspeed_config_builder.py --stage 3 --offload
conda run -n aigc python parallelism_planner.py --total-gpus 128 --tp 8 --pp 4
```

如果你在受限沙箱里运行 `torchrun`，本地 TCP rendezvous 可能被拦截并报 `Operation not permitted`。真实终端通常不会有这个限制；在本次验证中，允许本地 TCPStore 后两个 `torchrun` 示例均已跑通。

---

## 理论与实践怎么组织

建议按四层学习：

| 层次 | 要回答的问题 | 对应材料 |
|---|---|---|
| 理论层 | 为什么并行训练受显存、通信、同步和拓扑共同约束？ | `00-distributed-training-theory.md` |
| 基础层 | `rank`、`world_size`、集合通信、DDP 训练循环怎么工作？ | `01` + `collectives_demo.py` + `ddp_cpu_train.py` |
| 分片层 | FSDP/ZeRO 具体省了哪部分显存，代价是什么？ | `02`、`04` + `fsdp_memory_math.py` |
| 编排层 | Accelerate、DeepSpeed、3D 并行如何降低工程复杂度？ | `03`、`05` + `accelerate_train.py` + `parallelism_planner.py` |

学习顺序建议：

1. 先读 `00`，建立显存模型、通信模型和并行维度。
2. 跑通 `collectives_demo.py`，理解集合通信不是抽象概念。
3. 跑通 `ddp_cpu_train.py`，理解 DDP 的数据切分、梯度同步、rank 0 I/O。
4. 跑 `fsdp_memory_math.py` 和 `deepspeed_config_builder.py`，把 FSDP/ZeRO 的显存收益量化。
5. 跑 `accelerate_train.py`，理解框架如何封装 DDP/FSDP/DeepSpeed 细节。

## 前置知识

开始本模块前，你应该已经掌握：

- PyTorch 基础（`nn.Module`、训练循环、AMP）→ 见模块 02
- Linux 基本操作（SSH、环境变量、多进程概念）
- GPU 显存概念（参数 / 梯度 / 优化器状态 / 激活值各占多少）

---

## 推荐配套资源

| 类型 | 资源 | 说明 |
|---|---|---|
| 文档 | [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html) | 官方分布式全景图 |
| 文档 | [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) | 官方 FSDP 教程 |
| 文档 | [HuggingFace Accelerate Docs](https://huggingface.co/docs/accelerate) | Accelerate 官方文档 |
| 文档 | [DeepSpeed Docs](https://www.deepspeed.ai/) | DeepSpeed 官方站点 |
| 文档 | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | NVIDIA 大模型训练框架 |
| 论文 | [ZeRO: Memory Optimizations](https://arxiv.org/abs/1910.02054) | DeepSpeed ZeRO 原论文 |
| 论文 | [Megatron-LM (2020)](https://arxiv.org/abs/1909.08053) | 张量并行原论文 |
| 论文 | [GPipe (2019)](https://arxiv.org/abs/1811.06965) | 流水线并行原论文 |
| 博客 | [Lillian Weng: Large Transformer Model Training](https://lilianweng.github.io/posts/2021-09-25-train-large/) | 全景综述 |
| 博客 | [HuggingFace: Model Parallelism](https://huggingface.co/docs/transformers/perf_train_gpu_many) | 多 GPU 训练指南 |

---

## 自检清单

学完本模块，你应该能自信地回答以下问题：

- [ ] `world_size`、`rank`、`local_rank` 分别表示什么？
- [ ] DDP 和 DP（`nn.DataParallel`）的核心区别是什么？为什么 DDP 更快？
- [ ] `torchrun` 比 `python -m torch.distributed.launch` 好在哪里？
- [ ] FSDP 和 DDP 在显存占用上的差异是什么？
- [ ] FSDP 的三种分片策略（`FULL_SHARD` / `SHARD_GRAD_OP` / `NO_SHARD`）各适合什么场景？
- [ ] DeepSpeed ZeRO-1 / 2 / 3 分别切分了什么？
- [ ] 用 Accelerate 把一个单卡训练脚本改成分布式需要改几行？
- [ ] Tensor Parallelism 的通信开销主要在哪里？
- [ ] Pipeline Parallelism 的"气泡"(bubble) 是什么？如何减少？
- [ ] 什么是 3D 并行？为什么训练 100B+ 模型需要它？
- [ ] 训练一个 7B 模型，你会选择哪种并行方案？70B 呢？
- [ ] `NCCL_TIMEOUT` 报错通常是什么原因导致的？
- [ ] 分布式训练中如何保证梯度同步的正确性？
- [ ] 跑通 `examples/` 中的 CPU 分布式 demo，并能解释每个输出。
- [ ] 解释为什么当前 CPU 环境能验证 DDP 机制，但不能验证 NCCL/FSDP/DeepSpeed GPU 性能。
