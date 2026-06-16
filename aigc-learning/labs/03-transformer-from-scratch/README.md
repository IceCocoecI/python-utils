# Lab 03：从零理解 Transformer

> 目标：通过最小代码理解 attention、mask、RoPE、KV cache、采样，而不是把 Transformer 当黑箱。

---

## 前置知识

- [../../02-deep-learning-libraries/00-deep-learning-theory.md](../../02-deep-learning-libraries/00-deep-learning-theory.md)
- [../../02-deep-learning-libraries/06-transformer-principles-overview.md](../../02-deep-learning-libraries/06-transformer-principles-overview.md)
- [../../02-deep-learning-libraries/05-transformer-from-scratch.md](../../02-deep-learning-libraries/05-transformer-from-scratch.md)
- [../../09-frontier-models/01-llm-architectures.md](../../09-frontier-models/01-llm-architectures.md)

---

## 运行脚本

```bash
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/transformer_from_scratch.py
```

---

## 任务

1. 画出一条 token 序列经过 embedding、attention、FFN、LM head 的路径。
2. 在代码中找到 RMSNorm、RoPE、causal attention、SwiGLU、KV cache 的实现位置。
3. 记录每个关键张量的 shape：
   - token ids
   - embeddings
   - Q/K/V
   - attention scores
   - attention output
   - logits
4. 修改采样参数，观察输出变化。
5. 对比启用和不启用 KV cache 时生成过程的计算差异。
6. 写一页笔记解释 LLaMA 风格 block 的组成。

---

## 验收标准

- [ ] 能手推 `Q @ K.T / sqrt(d)` 的 shape。
- [ ] 能解释 causal mask 为什么不能省。
- [ ] 能解释 RoPE 作用在 Q/K 而不是 V 上的原因。
- [ ] 能估算 KV cache 显存随 layer、head、seq len 的变化。
- [ ] 能说明 top-k、top-p、temperature 的区别。

---

## 延伸挑战

- 增加 grouped-query attention 的 toy 实现。
- 加入一个简单 tokenizer，跑字符级生成。
- 对 attention 部分做 `torch.profiler` 分析。

