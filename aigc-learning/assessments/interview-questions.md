# AIGC 工程面试问题集

> 用于复盘、模拟面试和查漏补缺。建议先回答，再回模块文档核对。

---

## Python 与工程基础

1. 为什么自定义装饰器通常要用 `functools.wraps`？
2. generator 和 list 在大规模数据处理中的内存差异是什么？
3. `asyncio.gather` 和 `asyncio.create_task` 的区别是什么？
4. `Protocol` 相比 ABC 的优势是什么？
5. 为什么可变默认参数危险？如何修复？
6. 一个训练项目应该如何组织目录？
7. 你会如何给一个数据处理函数写单元测试？

---

## PyTorch 与训练

1. `loss.backward()` 背后发生了什么？
2. 为什么 PyTorch 梯度默认累积？
3. `model.train()` 和 `model.eval()` 改变了哪些层的行为？
4. `torch.no_grad()` 和 `torch.inference_mode()` 有什么区别？
5. AMP 为什么能提速和省显存？什么时候会引入数值问题？
6. 如何排查 loss NaN？
7. 如何估算一个 7B 模型训练需要多少显存？
8. `torch.compile` 适合什么场景？可能失败在哪里？

---

## Transformer 与 LLM

1. 请手推 Multi-Head Attention 的 shape。
2. 为什么 causal LM 训练时可以并行，而推理时通常自回归？
3. RoPE 的核心思想是什么？
4. KV cache 为什么能提速？显存如何估算？
5. GQA 和 MQA 的区别是什么？
6. temperature、top-k、top-p 分别控制什么？
7. 为什么 LLM 主流是 decoder-only？
8. FlashAttention 相比普通 attention 优化了什么？

---

## HuggingFace 与生态

1. `AutoTokenizer`、`AutoModelForCausalLM`、`pipeline` 各适合什么场景？
2. chat template 为什么重要？
3. `device_map="auto"` 和手动 `.to(device)` 为什么可能冲突？
4. `Trainer` 适合什么，不适合什么？
5. `datasets.map`、`filter`、`save_to_disk` 如何用于训练数据处理？
6. `safetensors` 相比 `torch.save` 的优势是什么？

---

## 微调与对齐

1. LoRA 为什么能减少训练参数量？
2. 如何选择 LoRA 的 target modules？
3. QLoRA 和 LoRA 的核心差异是什么？
4. SFT 数据中 system/user/assistant 的组织方式为什么重要？
5. DPO 和 RLHF/PPO 的区别是什么？
6. 如何评估一次微调是否真的变好？
7. 微调后模型出现过拟合或灾难性遗忘怎么办？
8. 量化为什么可能影响数学和代码能力？

---

## 分布式训练

1. DDP 为什么要 all-reduce 梯度？
2. FSDP 和 DeepSpeed ZeRO-3 的共同点是什么？
3. 数据并行、张量并行、流水线并行分别解决什么问题？
4. activation checkpointing 如何省显存？
5. 梯度累积和增大 batch size 是否等价？
6. 多机训练最容易遇到哪些通信和环境问题？
7. Accelerate 适合什么场景？

---

## 推理部署

1. Prefill 和 decode 为什么瓶颈不同？
2. TTFT 和 TPOT 分别是什么？
3. Continuous batching 为什么适合 LLM？
4. vLLM 的 PagedAttention 解决了什么问题？
5. 如何选择 vLLM、SGLang、TensorRT-LLM、llama.cpp？
6. 如何做 LLM 服务压测？
7. 如何控制推理成本？
8. OpenAI-compatible 服务要注意哪些兼容边界？

---

## RAG 与 Agent

1. RAG 解决了 LLM 的哪些问题？
2. chunk size 和 overlap 如何选择？
3. Dense retrieval、BM25、hybrid retrieval 的差异是什么？
4. reranker 放在 RAG 哪个位置？为什么有效？
5. 如何评估 RAG 系统？
6. 什么是 prompt injection？如何防护？
7. ReAct 和 Plan-and-Execute 的区别是什么？
8. 工具调用失败时 Agent 应该如何恢复？

---

## 前沿架构

1. MoE 的优势和问题是什么？
2. DeepSeek 系模型的 MLA 主要解决什么瓶颈？
3. Latent Diffusion 和 pixel-space diffusion 的差异是什么？
4. DiT 为什么成为图像生成的重要路线？
5. CLIP 的训练目标是什么？
6. LLaVA 如何把视觉特征接入 LLM？
7. 视频生成相比图像生成多了哪些挑战？
8. 神经音频 codec 对 TTS 有什么意义？

---

## CUDA、Triton 与性能

1. GPU 的内存层级是什么？
2. 什么是 warp divergence？
3. 什么是 memory coalescing？
4. 如何判断一个 kernel 是 compute-bound 还是 memory-bound？
5. operator fusion 为什么能提速？
6. Triton 和 CUDA C++ 的开发体验差异是什么？
7. `torch.profiler` 能看到什么，Nsight 又能看到什么？
8. 自定义算子如何接入 PyTorch 和 `torch.compile`？

