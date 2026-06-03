# 模块 06：LLM 微调与对齐

> 预训练给了模型知识，**微调**教它做事，**对齐**让它做好事。
> 掌握 SFT + RLHF/DPO + PEFT，你就能把一个通用基座模型变成专属的业务助手。

---

## 为什么这一章是 AIGC 工程师的核心技能？

大模型时代的分工正在变化：

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM 训练流水线                             │
│                                                             │
│   Pretrain ──→ SFT ──→ Reward Model ──→ RLHF / DPO         │
│   (大厂做)     (你做)   (你做/可跳过)    (你做)               │
│                                                             │
│   ← 需要千卡 →  ← 8卡甚至单卡即可完成 →                      │
└─────────────────────────────────────────────────────────────┘
```

**绝大部分 AIGC 工程师的主战场在 SFT + 对齐这一段。** 而 PEFT/LoRA 让单卡也能微调 7B+ 模型，量化让推理部署成本降一个量级。

本模块覆盖的技能栈：

| 技能 | 对应工作 |
|---|---|
| PEFT / LoRA / QLoRA | 低成本微调 |
| 量化（GPTQ / AWQ / bitsandbytes） | 模型压缩与加速部署 |
| SFT 数据工程 + 训练 | 让模型"听话"做任务 |
| RLHF / DPO / GRPO | 让模型输出更安全、更有帮助 |

---

## 学习内容

| # | 文档 | 核心话题 |
|---|---|---|
| 01 | [peft-and-lora](./01-peft-and-lora.md) | LoRA 数学原理 / QLoRA / DoRA / peft 库实战 |
| 02 | [quantization](./02-quantization.md) | INT8 / INT4 / GPTQ / AWQ / GGUF / bitsandbytes |
| 03 | [sft-data-and-training](./03-sft-data-and-training.md) | 数据格式 / Chat Template / TRL SFTTrainer / 完整训练脚本 |
| 04 | [alignment-rlhf-dpo](./04-alignment-rlhf-dpo.md) | Reward Model / PPO / DPO / GRPO / TRL 实战 |

---

## 前置知识

- 模块 02：PyTorch 基础（Module、Optimizer、训练循环）
- 模块 04：Transformers 库基础用法（AutoModel, Trainer）
- 模块 05：分布式训练基本概念（可选但有帮助）

---

## 推荐配套资源

### 核心论文

| 论文 | 要点 |
|---|---|
| [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) | 低秩适配矩阵 |
| [QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314) | 4-bit + LoRA |
| [DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) | 跳过 Reward Model 的对齐 |
| [GRPO (Shao et al., 2024)](https://arxiv.org/abs/2402.03300) | DeepSeek 的组相对策略优化 |
| [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) | RLHF 经典 |
| [Constitutional AI (Bai et al., 2022)](https://arxiv.org/abs/2212.08073) | 自我纠正对齐 |

### 核心代码库

| 库 | 用途 |
|---|---|
| [huggingface/peft](https://github.com/huggingface/peft) | 所有 PEFT 方法的统一接口 |
| [huggingface/trl](https://github.com/huggingface/trl) | SFT / DPO / PPO / GRPO Trainer |
| [huggingface/transformers](https://github.com/huggingface/transformers) | 基座模型加载 |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | 量化 + 8-bit 优化器 |
| [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) | GPTQ 量化 |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | GGUF 量化 + CPU 推理 |

### 数据集参考

| 数据集 | 说明 |
|---|---|
| [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) | 大规模指令跟随 |
| [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | 高质量 SFT |
| [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | DPO 偏好数据 |
| [tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval) | 对齐评估 |

---

## 硬件与环境建议

```
最低配置（能跑 QLoRA 7B）：
  - 1× A100-40GB 或 1× RTX 4090 (24GB)
  - RAM 64GB+
  - SSD 500GB+

推荐配置（能跑 LoRA 70B / DPO 7B）：
  - 4× A100-80GB
  - RAM 256GB+

关键依赖：
  pip install torch transformers peft trl datasets
  pip install bitsandbytes accelerate
  pip install flash-attn --no-build-isolation
```

---

## 自检清单

- [ ] 解释 LoRA 的数学公式 `W' = W + BA`，为什么能省参数？
- [ ] rank=8 和 rank=64 分别适合什么场景？
- [ ] QLoRA 用了哪 3 个技术来省显存？
- [ ] GPTQ 和 AWQ 的核心思想有什么区别？
- [ ] INT4 量化后的模型质量损失大约是多少 perplexity？
- [ ] SFT 数据中 loss mask 的作用是什么？
- [ ] ChatML 和 Llama 的 chat template 有什么区别？
- [ ] 为什么 SFT 容易过拟合？如何缓解？
- [ ] DPO 相比 PPO 的核心简化是什么？
- [ ] GRPO 不需要 critic model，它用什么替代？
- [ ] 什么是 reward hacking？如何缓解？
- [ ] KL 散度在 RLHF 中的作用是什么？
- [ ] 用 TRL 训一个 DPO 模型需要哪几步？
- [ ] LoRA adapter 如何合并回基座模型？合并后有什么好处？
- [ ] 解释 catastrophic forgetting，微调时如何避免？
