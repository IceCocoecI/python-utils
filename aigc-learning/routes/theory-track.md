# 理论优先路线

> 目标：建立 AIGC 工程背后的理论骨架，能解释关键机制、读论文、做架构选型。
> 理论路线不是纯阅读；每个阶段都要跑一个最小代码例子来校准理解。

---

## T1：数据、张量、优化基础

**必读**

- [../03-data-and-scientific-computing/00-data-and-scientific-computing-theory.md](../03-data-and-scientific-computing/00-data-and-scientific-computing-theory.md)
- [../02-deep-learning-libraries/00-deep-learning-theory.md](../02-deep-learning-libraries/00-deep-learning-theory.md)
- [../03-data-and-scientific-computing/01-numpy-essentials.md](../03-data-and-scientific-computing/01-numpy-essentials.md)
- [../02-deep-learning-libraries/01-pytorch-fundamentals.md](../02-deep-learning-libraries/01-pytorch-fundamentals.md)

**要能回答**

- shape、dtype、layout、range 为什么是数据管线的不变量？
- 为什么深度学习代码最终大量落到矩阵乘法？
- autograd 里的计算图、叶子节点、梯度累积分别是什么？
- Cross Entropy、MSE 在 LLM 和扩散模型中分别扮演什么角色？
- AdamW 为什么成为默认优化器？

**最小实践**

```bash
conda run -n aigc python aigc-learning/03-data-and-scientific-computing/examples/numpy_basics.py
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/pytorch_basics.py
```

---

## T2：Transformer 原理

**必读**

- [../02-deep-learning-libraries/06-transformer-principles-overview.md](../02-deep-learning-libraries/06-transformer-principles-overview.md)
- [../02-deep-learning-libraries/05-transformer-from-scratch.md](../02-deep-learning-libraries/05-transformer-from-scratch.md)
- [../09-frontier-models/01-llm-architectures.md](../09-frontier-models/01-llm-architectures.md)

**要能回答**

- Self-Attention 为什么能建模 token 间关系？
- Multi-Head Attention 为什么不是简单重复计算？
- Decoder-only、Encoder-only、Encoder-Decoder 的适用场景有什么差异？
- RoPE、ALiBi、绝对位置编码的核心差异是什么？
- KV cache 为什么能降低自回归推理复杂度？
- GQA/MQA 如何降低 KV cache 显存？

**最小实践**

```bash
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/transformer_from_scratch.py
```

---

## T3：训练工程理论

**必读**

- [../04-training-engineering/00-training-engineering-theory.md](../04-training-engineering/00-training-engineering-theory.md)
- [../04-training-engineering/04-reproducibility.md](../04-training-engineering/04-reproducibility.md)
- [../05-distributed-training/00-distributed-training-theory.md](../05-distributed-training/00-distributed-training-theory.md)
- [../05-distributed-training/05-parallelism-strategies.md](../05-distributed-training/05-parallelism-strategies.md)

**要能回答**

- Experiment、Run、Config、Artifact 之间是什么关系？
- 可重复、可复现、可复制有什么区别？
- 数据并行、张量并行、流水线并行、专家并行分别切什么？
- DDP 为什么要做梯度 all-reduce？
- ZeRO/FSDP 为什么能省显存？代价是什么？

**最小实践**

```bash
conda run -n aigc python aigc-learning/04-training-engineering/examples/reproducible_train.py --epochs 1
conda run -n aigc python aigc-learning/05-distributed-training/examples/fsdp_memory_math.py
```

---

## T4：微调与对齐理论

**必读**

- [../06-finetuning-and-alignment/00-finetuning-and-alignment-theory.md](../06-finetuning-and-alignment/00-finetuning-and-alignment-theory.md)
- [../06-finetuning-and-alignment/01-peft-and-lora.md](../06-finetuning-and-alignment/01-peft-and-lora.md)
- [../06-finetuning-and-alignment/02-quantization.md](../06-finetuning-and-alignment/02-quantization.md)
- [../06-finetuning-and-alignment/04-alignment-rlhf-dpo.md](../06-finetuning-and-alignment/04-alignment-rlhf-dpo.md)

**要能回答**

- LoRA 为什么可以用低秩矩阵表达参数更新？
- QLoRA 的省显存来自哪里？
- SFT、Reward Model、PPO、DPO 的训练目标分别是什么？
- Preference data 和 instruction data 的差异是什么？
- 对齐训练为什么需要评估安全性和偏好一致性？

**最小实践**

```bash
conda run -n aigc python aigc-learning/06-finetuning-and-alignment/examples/dpo_loss_demo.py
conda run -n aigc python aigc-learning/06-finetuning-and-alignment/examples/quantization_sim.py
```

---

## T5：推理系统理论

**必读**

- [../07-inference-and-deployment/00-inference-and-deployment-theory.md](../07-inference-and-deployment/00-inference-and-deployment-theory.md)
- [../07-inference-and-deployment/01-llm-inference-engines.md](../07-inference-and-deployment/01-llm-inference-engines.md)
- [../07-inference-and-deployment/03-serving-frameworks.md](../07-inference-and-deployment/03-serving-frameworks.md)
- [../10-cuda-and-triton/03-performance-profiling.md](../10-cuda-and-triton/03-performance-profiling.md)

**要能回答**

- Prefill 和 decode 阶段的瓶颈为什么不同？
- TTFT、TPOT、QPS、吞吐、延迟之间怎么权衡？
- Continuous batching 为什么提升吞吐？
- PagedAttention 为什么要把 KV cache 做成类似虚拟内存？
- 量化对显存、速度、质量分别有什么影响？

**最小实践**

```bash
conda run -n aigc python aigc-learning/07-inference-and-deployment/examples/kv_cache_and_batching.py
```

---

## T6：前沿模型架构

**必读**

- [../09-frontier-models/01-llm-architectures.md](../09-frontier-models/01-llm-architectures.md)
- [../09-frontier-models/02-image-generation.md](../09-frontier-models/02-image-generation.md)
- [../09-frontier-models/03-multimodal-models.md](../09-frontier-models/03-multimodal-models.md)

**按方向选读**

- 视频生成：[../09-frontier-models/04-video-generation.md](../09-frontier-models/04-video-generation.md)
- 语音音频：[../09-frontier-models/05-speech-and-audio.md](../09-frontier-models/05-speech-and-audio.md)

**要能回答**

- LLaMA/Qwen/DeepSeek/Mistral 这类模型的关键架构差异是什么？
- MoE 的优势、成本和负载均衡问题是什么？
- DDPM、Latent Diffusion、DiT、Flow Matching 的演进逻辑是什么？
- CLIP/LLaVA/Qwen-VL 这类多模态模型如何把视觉信息接入 LLM？

---

## T7：GPU 与性能模型

**必读**

- [../10-cuda-and-triton/01-gpu-architecture-and-cuda-basics.md](../10-cuda-and-triton/01-gpu-architecture-and-cuda-basics.md)
- [../10-cuda-and-triton/02-triton-programming.md](../10-cuda-and-triton/02-triton-programming.md)
- [../10-cuda-and-triton/03-performance-profiling.md](../10-cuda-and-triton/03-performance-profiling.md)
- [../10-cuda-and-triton/04-custom-operators-and-extensions.md](../10-cuda-and-triton/04-custom-operators-and-extensions.md)

**要能回答**

- GPU 内存层级是什么？
- 什么是 compute-bound 和 memory-bound？
- 为什么 operator fusion 能提速？
- FlashAttention 的 IO-aware 思想是什么？
- Triton 为什么用 block 级抽象而不是直接暴露 CUDA thread？

**最小实践**

按你的硬件环境选择运行。没有 NVIDIA GPU 时，只完成 profiling 理论和 PyTorch CPU trace 即可。

