# 理论路线自检问题

> 配合 [../routes/theory-track.md](../routes/theory-track.md) 使用。
> 每个问题都应该能用自己的话解释，并能指出相关代码或公式。

---

## T1：数据、张量、优化

- [ ] shape、dtype、layout、range、distribution 为什么是数据管线的核心不变量？
- [ ] broadcasting 的规则是什么？什么时候会悄悄引入 bug？
- [ ] 为什么矩阵乘法是深度学习计算的核心？
- [ ] 动态计算图和静态计算图有什么区别？
- [ ] `loss.backward()` 后梯度存在哪里？
- [ ] 为什么梯度会累积？
- [ ] Cross Entropy 为什么适合分类和语言模型？
- [ ] AdamW 和 Adam 的 weight decay 有什么差异？
- [ ] fp16 和 bf16 的数值稳定性差异是什么？

---

## T2：Transformer

- [ ] Self-Attention 的核心公式是什么？每个矩阵 shape 是什么？
- [ ] 为什么 attention score 要除以 `sqrt(d_k)`？
- [ ] Multi-Head Attention 带来了什么表达能力？
- [ ] 为什么 decoder-only LLM 需要 causal mask？
- [ ] RoPE 为什么适合现代 LLM？
- [ ] RMSNorm 和 LayerNorm 的差异是什么？
- [ ] SwiGLU 为什么常见于 LLaMA 系模型？
- [ ] KV cache 如何降低自回归推理复杂度？
- [ ] GQA/MQA 如何降低推理显存？
- [ ] FlashAttention 为什么说是 IO-aware？

---

## T3：训练工程与分布式

- [ ] Experiment、Run、Config、Artifact 的关系是什么？
- [ ] 可重复、可复现、可复制有什么区别？
- [ ] 超参搜索为什么是带噪声、带预算的优化问题？
- [ ] 为什么训练必须记录代码版本、数据版本和环境版本？
- [ ] DDP 的梯度同步发生在什么时候？
- [ ] FSDP/ZeRO 把哪些状态切分了？
- [ ] TP、PP、DP、EP 分别切分了什么？
- [ ] 通信带宽如何影响大模型训练效率？
- [ ] activation checkpointing 省了什么、付出了什么？

---

## T4：微调与对齐

- [ ] LoRA 为什么能用低秩更新近似参数变化？
- [ ] LoRA 的 `r`、`alpha`、`target_modules` 如何影响结果？
- [ ] QLoRA 中 NF4、double quantization、paged optimizer 分别解决什么问题？
- [ ] SFT 数据质量为什么通常比数据量更关键？
- [ ] Chat template 错误会造成什么后果？
- [ ] Reward model 在 RLHF 中扮演什么角色？
- [ ] DPO 如何绕开显式 reward model 和 PPO？
- [ ] preference pair 中 chosen/rejected 的噪声会造成什么问题？
- [ ] 对齐后的模型为什么还需要安全评估？

---

## T5：推理系统

- [ ] Prefill 和 decode 的计算特征为什么不同？
- [ ] TTFT、TPOT、吞吐、QPS、尾延迟分别衡量什么？
- [ ] Continuous batching 为什么比静态 batching 更适合 LLM 服务？
- [ ] KV cache 的显存随哪些变量增长？
- [ ] PagedAttention 为什么能提升 KV cache 管理效率？
- [ ] 量化对显存、速度、质量分别有什么影响？
- [ ] speculative decoding 的核心思想是什么？
- [ ] OpenAI-compatible API 的兼容边界在哪里？
- [ ] 服务限流、队列、超时如何影响用户体验？

---

## T6：前沿模型架构

- [ ] GPT、LLaMA、Qwen、Mistral、DeepSeek 的关键架构差异是什么？
- [ ] MoE 为什么能提高总参数量但控制激活参数量？
- [ ] Load balancing loss 解决什么问题？
- [ ] MLA、GQA、MQA 都围绕什么瓶颈设计？
- [ ] DDPM 的前向和反向过程分别是什么？
- [ ] Latent Diffusion 为什么比 pixel diffusion 更高效？
- [ ] DiT 为什么用 Transformer 替代 U-Net？
- [ ] Flow Matching 和 Diffusion 的训练目标有什么差异？
- [ ] CLIP 如何实现图文对齐？
- [ ] LLaVA/Qwen-VL 如何把图像 token 接入 LLM？

---

## T7：GPU 与性能

- [ ] GPU 内存层级有哪些？
- [ ] warp、block、grid 分别是什么？
- [ ] 什么是 coalesced memory access？
- [ ] 什么是 bank conflict？
- [ ] Roofline 模型如何判断 compute-bound 和 memory-bound？
- [ ] operator fusion 为什么能减少显存读写？
- [ ] Triton 的 block-level 编程模型是什么？
- [ ] Nsight Systems 和 Nsight Compute 关注点有什么不同？
- [ ] 自定义算子接入 PyTorch 有哪些方式？

