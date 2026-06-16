# Lab 04：LoRA、SFT 与 DPO 小闭环

> 目标：理解微调工程的最小链路：数据格式、LoRA 参数、量化影响、SFT 训练、偏好优化。

---

## 前置知识

- [../../06-finetuning-and-alignment/00-finetuning-and-alignment-theory.md](../../06-finetuning-and-alignment/00-finetuning-and-alignment-theory.md)
- [../../06-finetuning-and-alignment/01-peft-and-lora.md](../../06-finetuning-and-alignment/01-peft-and-lora.md)
- [../../06-finetuning-and-alignment/02-quantization.md](../../06-finetuning-and-alignment/02-quantization.md)
- [../../06-finetuning-and-alignment/03-sft-data-and-training.md](../../06-finetuning-and-alignment/03-sft-data-and-training.md)
- [../../06-finetuning-and-alignment/04-alignment-rlhf-dpo.md](../../06-finetuning-and-alignment/04-alignment-rlhf-dpo.md)

---

## 运行脚本

```bash
conda run -n aigc python aigc-learning/06-finetuning-and-alignment/examples/sft_data_pipeline.py
conda run -n aigc python aigc-learning/06-finetuning-and-alignment/examples/lora_tiny_train.py --epochs 1
conda run -n aigc python aigc-learning/06-finetuning-and-alignment/examples/quantization_sim.py
conda run -n aigc python aigc-learning/06-finetuning-and-alignment/examples/dpo_loss_demo.py
```

---

## 任务

1. 构造 20 条 instruction/chat 样例，字段至少包含 system、user、assistant。
2. 用 chat template 生成训练文本，检查特殊 token 和截断策略。
3. 跑通 tiny LoRA 训练，记录：
   - base 参数量
   - trainable 参数量
   - LoRA `r`、`alpha`、`dropout`
   - target modules
4. 运行量化模拟，比较 INT8/INT4 的误差直觉。
5. 运行 DPO loss demo，写出 chosen/rejected、beta、reference model 的作用。
6. 写一份微调实验卡片。

---

## 验收标准

- [ ] 能说明 SFT 数据为什么不能只保存纯文本答案。
- [ ] 能解释 LoRA 低秩更新的参数节省来自哪里。
- [ ] 能解释 QLoRA 为什么需要 4bit base + LoRA adapter。
- [ ] 能说明 DPO 和 PPO/RLHF 的工程差异。
- [ ] 能写出微调失败时的排查顺序：数据、模板、学习率、target modules、评估。

---

## 延伸挑战

- 增加一份 eval set，比较微调前后输出。
- 尝试不同 LoRA rank，记录效果和训练参数量。
- 把 adapter merge 回 base model，并记录文件变化。

