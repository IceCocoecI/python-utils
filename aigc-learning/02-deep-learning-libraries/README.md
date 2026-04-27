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
| 01 | [pytorch-fundamentals](./01-pytorch-fundamentals.md) | Tensor / autograd / nn.Module / device |
| 02 | [pytorch-training-loop](./02-pytorch-training-loop.md) | Dataset / DataLoader / optimizer / AMP / 保存加载 / torch.compile / **显存预算** / **训练调试** |
| 03 | [huggingface-transformers](./03-huggingface-transformers.md) | Tokenizer / AutoModel / Trainer / PEFT (LoRA) |
| 04 | [huggingface-diffusers](./04-huggingface-diffusers.md) | Pipeline / Scheduler / UNet / VAE / 图像生成与微调 |
| 05 | [transformer-from-scratch](./05-transformer-from-scratch.md) | 多头注意力、位置编码（RoPE/ALiBi）、掩码、KV cache、FlashAttention |

---

## 示例代码（`examples/`）

| 文件 | 说明 | 是否需要 GPU |
|---|---|---|
| `pytorch_basics.py` | Tensor 操作、autograd、device 管理 | 否（CPU 可跑） |
| `mlp_mnist.py` | 完整训练循环：MLP 在 MNIST 上分类 | 否 |
| `transformers_quickstart.py` | 加载 LLM、编码解码、chat template | CPU 可跑小模型 |
| `diffusers_quickstart.py` | 加载 Stable Diffusion，文生图 | 推荐 GPU |

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
