# 实践路线验收清单

> 配合 [../routes/practice-track.md](../routes/practice-track.md) 使用。

---

## P0：环境与速查

- [ ] 能在 `aigc` 环境中运行 Python。
- [ ] 能确认 PyTorch 版本和 CUDA 是否可用。
- [ ] 能说出 `requirements.txt`、`CHEATSHEET.md` 的用途。

---

## P1：Python、数据、Tensor

- [ ] 能用 `dataclass` 或 Pydantic 写配置对象。
- [ ] 能用 generator 处理流式样本。
- [ ] 能解释 `list`、generator、iterator 的区别。
- [ ] 能解释 NumPy view/copy。
- [ ] 能用 einops 完成 CHW/HWC、patch flatten、QKV split。
- [ ] 能解释 PyTorch Tensor 的 shape、dtype、device。
- [ ] 能处理常见 shape mismatch 报错。

---

## P2：小模型训练

- [ ] 能写完整训练循环：forward、loss、backward、step、eval、save。
- [ ] 能解释 `optimizer.zero_grad(set_to_none=True)`。
- [ ] 能解释 train/eval mode 的区别。
- [ ] 能保存和加载 `state_dict`。
- [ ] 能记录 config、metrics、artifacts。
- [ ] 能复现实验并说明可复现边界。
- [ ] 能定位 loss NaN、loss 不下降、训练慢的常见原因。

---

## P3：Transformer 与 HuggingFace

- [ ] 能解释 tokenizer、token ids、attention mask。
- [ ] 能用 `AutoTokenizer` 和 `AutoModelForCausalLM` 做生成。
- [ ] 能解释 temperature、top-k、top-p、max_new_tokens。
- [ ] 能手推 attention 主要张量 shape。
- [ ] 能解释 causal mask。
- [ ] 能解释 RoPE、RMSNorm、SwiGLU、KV cache 的作用。
- [ ] 能跑通 toy Diffusers 示例。

---

## P4：微调与对齐

- [ ] 能构造 SFT JSONL 或 chat message 数据。
- [ ] 能解释 chat template 的作用。
- [ ] 能跑通 LoRA tiny train。
- [ ] 能计算 trainable parameter ratio。
- [ ] 能解释 LoRA、QLoRA、full fine-tuning 的差异。
- [ ] 能解释 INT8/INT4 量化的收益和风险。
- [ ] 能解释 chosen/rejected preference pair。
- [ ] 能解释 DPO loss 的基本目标。

---

## P5：推理服务

- [ ] 能启动 toy OpenAI-compatible server。
- [ ] 能构造 `/v1/chat/completions` 请求。
- [ ] 能解释 streaming 和 non-streaming。
- [ ] 能记录 request id、latency、输出长度。
- [ ] 能解释 TTFT、TPOT、吞吐、尾延迟。
- [ ] 能说明 vLLM/SGLang 与 toy server 的差距。
- [ ] 能写出服务健康检查和错误处理方案。

---

## P6：RAG 与 Agent

- [ ] 能读取 Markdown 文档并保留 metadata。
- [ ] 能实现 chunking，并解释 chunk size/overlap。
- [ ] 能生成或模拟 embedding。
- [ ] 能建立向量索引或轻量检索索引。
- [ ] 能返回 top-k chunk 和来源。
- [ ] 能把检索结果注入 prompt。
- [ ] 能准备测试问题并记录失败样例。
- [ ] 能说明 prompt injection 风险。

---

## P7：性能与规模

- [ ] 能估算模型参数显存。
- [ ] 能估算 KV cache 显存。
- [ ] 能解释 batch size、seq len、并发对显存和吞吐的影响。
- [ ] 能使用 profiler 或日志获得 baseline。
- [ ] 能判断瓶颈类型：CPU、I/O、GPU compute、GPU memory、通信。
- [ ] 能提出一个优化并测量效果。
- [ ] 能说明优化的副作用。

