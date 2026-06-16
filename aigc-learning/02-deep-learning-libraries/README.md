# 模块 02：深度学习核心库

> 这是 AIGC 工程师真正意义上的"主战场"。
> 本模块围绕 **PyTorch + HuggingFace 三件套** 展开，这是 2026 年几乎所有开源 AIGC 项目的底座。

---

## 为什么是这两个生态？

| 生态 | 地位 |
|---|---|
| **PyTorch** | 事实上的深度学习框架（OpenAI/Meta/Tesla 都在用） |
| **Transformers** | LLM 训练/推理的 de-facto 标准库 |
| **Diffusers** | 扩散模型的 de-facto 标准库（SD/Flux/Qwen-Image 全家桶） |
| **Accelerate** | 一行代码搞定分布式训练/混合精度 |

掌握这 4 个库，基本覆盖了大部分 AIGC 岗位的日常工作。

---

## 学习内容

| # | 文档 | 核心话题 |
|---|---|---|
| 00 | [deep-learning-theory](./00-deep-learning-theory.md) | 张量/梯度/优化/归一化/Attention/扩散/显存预算/训练稳定性 |
| 01 | [pytorch-fundamentals](./01-pytorch-fundamentals.md) | Tensor / autograd / nn.Module / device |
| 02 | [pytorch-training-loop](./02-pytorch-training-loop.md) | Dataset / DataLoader / optimizer / AMP / 保存加载 / torch.compile / **显存预算** / **训练调试** |
| 03 | [huggingface-transformers](./03-huggingface-transformers.md) | Tokenizer / AutoModel / Trainer / PEFT (LoRA) |
| 04 | [huggingface-diffusers](./04-huggingface-diffusers.md) | Pipeline / Scheduler / UNet / VAE / 图像生成与微调 |
| 05 | [transformer-from-scratch](./05-transformer-from-scratch.md) | 多头注意力、位置编码（RoPE/ALiBi）、掩码、KV cache、FlashAttention |
| 06 | [transformer-principles-overview](./06-transformer-principles-overview.md) | Transformer 原理白话速览（无代码概念版，适合入门/快速复习） |

---

## 示例代码（`examples/`）

| 文件 | 说明 | 是否需要 GPU |
|---|---|---|
| `pytorch_basics.py` | Tensor 操作、autograd、device 管理 | 否（CPU 可跑） |
| `mlp_mnist.py` | 完整训练循环：MLP 在 MNIST/合成 MNIST 上分类 | 否 |
| `transformers_quickstart.py` | Tokenizer、生成、流式输出、pipeline；默认离线小模型 | 否 |
| `diffusers_quickstart.py` | 默认离线 Toy DDPM；可选运行 Stable Diffusion | Toy 模式否，SD 推荐 GPU |
| `transformer_from_scratch.py` | RMSNorm / RoPE / 因果注意力 / SwiGLU / KV cache / 采样 | 否 |

### 本地验证命令

当前环境为 `aigc` 时，建议先跑离线 smoke tests：

```bash
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/pytorch_basics.py
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/mlp_mnist.py --synthetic --epochs 1 --max-train-batches 3 --max-val-batches 2 --workers 0
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/transformers_quickstart.py
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/diffusers_quickstart.py --toy-steps 1 --toy-batch-size 2
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/transformer_from_scratch.py
```

需要真实模型/真实数据时再打开下载路径：

```bash
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/mlp_mnist.py --epochs 3
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/transformers_quickstart.py --real-model
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/diffusers_quickstart.py --stable-diffusion
```

---

## 理论与实践怎么组织

本模块建议按四层学习：

| 层次 | 要回答的问题 | 对应材料 |
|---|---|---|
| 理论层 | 张量、梯度、优化、归一化、Attention、扩散、显存预算之间是什么关系？ | `00-deep-learning-theory.md` |
| 概念速览层 | 不看代码时，如何用白话解释 Transformer 的整体机制？ | `06-transformer-principles-overview.md` |
| 实现层 | 如何从 PyTorch 基础一步步写出训练循环和 Transformer block？ | `01`、`02`、`05` 文档 |
| 框架层 | 如何把核心概念映射到 Transformers / Diffusers 的真实生态？ | `03`、`04` 文档和 `examples/` |

学习顺序建议：

1. 先读 `00`，建立深度学习底层概念。
2. 读 `06` 快速建立 Transformer 心智模型。
3. 跑 PyTorch 和 Transformer from scratch 示例，确认 shape、mask、KV cache。
4. 再进入 HuggingFace Transformers 和 Diffusers。

---

## Transformer 内容分层

本仓库有多处讲 Transformer，它们不是重复章节，而是不同抽象层：

| 层次 | 文档 | 重点 |
|---|---|---|
| 概念直觉 | [06-transformer-principles-overview](./06-transformer-principles-overview.md) | 不写代码，用白话解释 Attention、位置编码、架构变体 |
| 公式与实现 | [05-transformer-from-scratch](./05-transformer-from-scratch.md) | 手写 Attention、RoPE、mask、KV cache、FlashAttention 相关实现 |
| 深度学习底座 | [00-deep-learning-theory](./00-deep-learning-theory.md) | 把 Attention 放回张量、梯度、优化、显存和数值稳定性体系里 |
| 现代架构演进 | [09/01-llm-architectures](../09-frontier-models/01-llm-architectures.md) | 解释 LLaMA、Qwen、DeepSeek、MoE、长上下文等工程化变体 |

推荐路径：

```text
06 概念速览 → 05 从零实现 → 09/01 现代架构
```

如果你已经会实现基础 Transformer，可以跳过 `06`，直接从 `05` 或 `09/01` 进入。

---

## 推荐配套资源

### 官方
- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics)
- [PyTorch 60 Minute Blitz](https://docs.pytorch.org/tutorials/beginner/blitz/)
- [HuggingFace Transformers Course](https://huggingface.co/learn/nlp-course)
- [HuggingFace Diffusion Models Course](https://github.com/huggingface/diffusion-models-class)

### 开源项目（适合读源码）
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — ~300 行复现 GPT-2
- [karpathy/nanochat](https://github.com/karpathy/nanochat) — 单节点完整 ChatGPT 复现
- [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) — 配套视频，按 git commit 步步解读
- [huggingface/transformers](https://github.com/huggingface/transformers) — 注意查 `src/transformers/models/llama/modeling_llama.py`
- [huggingface/diffusers](https://github.com/huggingface/diffusers) — 注意查 `examples/` 目录下的训练脚本

### 视频
- [Andrej Karpathy: Neural Networks Zero to Hero](https://karpathy.ai/zero-to-hero.html) — 强烈推荐，从 micrograd 到 GPT

---

## 自检清单

学完本模块，你应该能：

- [ ] 手写一个完整的 PyTorch 训练循环（包含 train/eval/save）。
- [ ] 解释 `loss.backward()` 背后发生了什么。
- [ ] 用 `torch.cuda.amp` 开混合精度训练。
- [ ] 用 `transformers.AutoModelForCausalLM` 加载 LLM 并生成文本。
- [ ] 用 `peft` 给 LLM 添加 LoRA 适配器。
- [ ] 用 `diffusers.StableDiffusionPipeline` 从 prompt 生成图像。
- [ ] 解释 Scheduler 在扩散模型里的作用。
- [ ] 推导 Attention 的 shape（`Q·Kᵀ / √d_k · V`）并解释因果掩码的作用。
- [ ] 解释 RoPE 相比绝对位置编码的优势。
- [ ] 估算一个 7B 模型训练/推理所需的显存。
- [ ] 定位训练 loss NaN 的常见原因。
