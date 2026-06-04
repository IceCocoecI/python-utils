# 03 · SFT 数据工程与训练

> 推荐等级：**★★★★★**
> "Data is all you need." —— SFT 的效果 80% 取决于数据质量，20% 取决于训练技巧。

---

## 0. 本地可运行示例

先用离线脚本验证 Alpaca/ShareGPT 转 messages、聊天模板、assistant-only loss mask 和 packing：

```bash
cd aigc-learning/06-finetuning-and-alignment/examples
conda run -n aigc python sft_data_pipeline.py --max-length 80
conda run -n aigc python sft_data_pipeline.py --max-length 80 --pack
```

脚本会打印每条样本的 token 数、被监督的 assistant token 数，以及被 mask 的 system/user token 数。真实训练时应使用目标模型 tokenizer 的 `apply_chat_template()`，本示例只保留机制。

---

## 1. SFT 在 LLM 流水线中的位置

```
Pretrain ──→  SFT  ──→ Alignment (RLHF/DPO)
 通用知识      做任务      做好任务

SFT = Supervised Fine-Tuning（有监督微调）

输入：预训练基座模型 + 指令-回复数据
输出：能按照指令完成任务的模型
```

**为什么不直接用预训练模型？**

预训练模型只会"续写"，不会"回答"：
- 输入 "What is Python?" → 输出 "What is Java? What is C++?..."（续写模式）
- SFT 后 → 输出 "Python is a programming language..."（回答模式）

---

## 2. 数据格式

### 2.1 Alpaca 格式（单轮）

```json
{
  "instruction": "将以下句子翻译成英文",
  "input": "今天天气很好",
  "output": "The weather is nice today."
}
```

### 2.2 ShareGPT 格式（多轮）

```json
{
  "conversations": [
    {"from": "human", "value": "帮我写一首关于秋天的诗"},
    {"from": "gpt", "value": "秋风起，落叶飘..."},
    {"from": "human", "value": "再写一首关于春天的"},
    {"from": "gpt", "value": "春风暖，花开满..."}
  ]
}
```

### 2.3 OpenAI 格式（messages）

```json
{
  "messages": [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "帮我写一首关于秋天的诗"},
    {"role": "assistant", "content": "秋风起，落叶飘..."},
    {"role": "user", "content": "再写一首关于春天的"},
    {"role": "assistant", "content": "春风暖，花开满..."}
  ]
}
```

### 2.4 格式转换

```python
def alpaca_to_messages(sample: dict) -> dict:
    """Alpaca 格式 → OpenAI messages 格式"""
    messages = []
    user_content = sample["instruction"]
    if sample.get("input"):
        user_content += f"\n\n{sample['input']}"
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": sample["output"]})
    return {"messages": messages}


def sharegpt_to_messages(sample: dict) -> dict:
    """ShareGPT 格式 → OpenAI messages 格式"""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = [
        {"role": role_map[turn["from"]], "content": turn["value"]}
        for turn in sample["conversations"]
    ]
    return {"messages": messages}
```

---

## 3. Chat Template（聊天模板）

不同模型要求不同的文本格式。**模板错误是 SFT 最常见的 bug。**

### 3.1 ChatML 格式（Qwen, Yi）

```
<|im_start|>system
你是一个有帮助的助手。<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么可以帮你的？<|im_end|>
```

### 3.2 Llama 3 格式

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

你是一个有帮助的助手。<|eot_id|><|start_header_id|>user<|end_header_id|>

你好<|eot_id|><|start_header_id|>assistant<|end_header_id|>

你好！有什么可以帮你的？<|eot_id|>
```

### 3.3 Mistral 格式

```
<s>[INST] 你好 [/INST]你好！有什么可以帮你的？</s>
```

### 3.4 使用 tokenizer 的 chat template

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

messages = [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "你好"},
]

# 自动应用模型的 chat template
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)

# 带 tokenize
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
```

**关键原则**：永远用 `tokenizer.apply_chat_template()`，不要手动拼字符串。

---

## 4. 数据质量工程

### 4.1 质量 > 数量

| 数据集 | 数据量 | 效果 |
|---|---|---|
| LIMA (Zhou et al.) | 1,000 条精选 | 媲美 65K Alpaca |
| Alpaca | 52K 条 GPT 生成 | 质量一般 |
| WizardLM | 250K 条渐进复杂 | 效果很好 |

**结论**：1000 条高质量 > 100K 条低质量。

### 4.2 数据质量检查清单

```python
def check_data_quality(dataset):
    issues = []
    for i, sample in enumerate(dataset):
        # 长度检查
        if len(sample["output"]) < 10:
            issues.append(f"[{i}] 回复太短")
        if len(sample["output"]) > 10000:
            issues.append(f"[{i}] 回复异常长")

        # 空字段检查
        if not sample["instruction"].strip():
            issues.append(f"[{i}] 指令为空")

        # 重复检查（用 instruction 去重）
        # 格式一致性检查
        # 语言一致性检查

    return issues
```

### 4.3 去重

```python
from datasets import Dataset
import hashlib

def dedup_by_instruction(dataset: Dataset) -> Dataset:
    seen = set()
    indices = []
    for i, sample in enumerate(dataset):
        h = hashlib.md5(sample["instruction"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            indices.append(i)
    return dataset.select(indices)
```

### 4.4 难度平衡

好的 SFT 数据集应包含不同难度级别：

```
简单（30%）：单步指令、事实问答
中等（50%）：多步推理、摘要、翻译
困难（20%）：代码生成、数学、创意写作
```

---

## 5. 开源数据集参考

| 数据集 | 规模 | 特点 | 适用 |
|---|---|---|---|
| tatsu-lab/alpaca | 52K | GPT-3.5 生成 | 入门 |
| WizardLM/WizardLM_evol_instruct | 250K | 渐进复杂化 | 通用 |
| teknium/OpenHermes-2.5 | 1M | 多源高质量 | 通用 |
| BAAI/Infinity-Instruct | 7M+ | 超大规模 | 通用 |
| m-a-p/Code-Feedback | 66K | 代码反馈 | 代码 |
| meta-math/MetaMathQA | 395K | 数学增强 | 数学 |

### 5.1 合成数据生成

```python
from openai import OpenAI

client = OpenAI()

def generate_sft_data(topic: str, n: int = 10) -> list[dict]:
    """用 GPT-4 生成 SFT 训练数据"""
    prompt = f"""为主题"{topic}"生成{n}条高质量的指令-回复训练数据。

要求：
1. 指令要多样化（问答、分析、创作、对比等）
2. 回复要详细、准确、有条理
3. 难度从简单到困难

输出 JSON 数组格式：
[{{"instruction": "...", "output": "..."}}]"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content
```

---

## 6. 使用 TRL SFTTrainer 训练

### 6.1 基础用法

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

model_id = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = load_dataset("json", data_files="train_data.jsonl", split="train")

# SFTConfig 继承自 TrainingArguments
training_args = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    max_seq_length=4096,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### 6.2 使用 messages 格式数据

```python
# 数据集格式：每条包含 "messages" 字段
# [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

dataset = load_dataset("json", data_files="chat_data.jsonl", split="train")

# TRL 会自动用 tokenizer.apply_chat_template 处理 messages 格式
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    # 不需要 formatting_func，TRL 自动检测 "messages" 字段
)
```

---

## 7. Loss Masking：只在 assistant 回复上计算 loss

### 7.1 为什么需要 loss masking？

```
输入: <user>天气怎么样？</user><assistant>今天天气晴朗，温度25度。</assistant>

不 mask:  loss 计算在整个序列上（包括 user 的问题）
有 mask:  loss 只计算 assistant 的回复部分
```

**不 mask 的问题**：模型会花精力"学习"如何生成用户的问题 → 浪费算力、可能学到不正确的模式。

### 7.2 TRL 自动 loss masking

```python
# TRL 支持 conversational dataset，具体是否自动只监督 assistant
# 取决于 TRL 版本、chat template 和 data collator 配置。
# 生产训练前应抽样检查 labels 中的 -100 mask 是否符合预期。

# 如果用 formatting_func，需要手动处理或显式配置：
training_args = SFTConfig(
    ...,
    dataset_text_field=None,  # 使用 messages 格式
)
```

### 7.3 手动实现 loss masking

```python
def create_labels_with_mask(input_ids, tokenizer, messages):
    """创建只在 assistant 回复处计算 loss 的 labels"""
    labels = input_ids.clone()

    # 找到 assistant 回复的起始和结束位置
    # 将非 assistant 部分的 label 设为 -100（CrossEntropyLoss 会忽略）
    assistant_start_token = tokenizer.encode("<|start_header_id|>assistant")

    in_assistant = False
    for i in range(len(input_ids)):
        if not in_assistant:
            labels[i] = -100  # 不计算 loss

    return labels
```

---

## 8. Packing：提高 GPU 利用率

### 8.1 问题

短文本 padding 到 max_length 会浪费大量计算：

```
样本1: [tok tok tok PAD PAD PAD PAD PAD]  ← 62% 浪费
样本2: [tok tok tok tok tok PAD PAD PAD]  ← 37% 浪费
```

### 8.2 Packing 策略

把多个短样本拼接到一个 max_length 序列中：

```
Packed: [样本1_tok tok tok SEP 样本2_tok tok tok tok tok]  ← 100% 利用
```

### 8.3 TRL 中启用 packing

```python
training_args = SFTConfig(
    ...,
    packing=True,          # 启用 packing
    max_seq_length=4096,   # pack 到这个长度
)
```

**注意**：Packing 会改变样本边界。生产训练前应确认所用 TRL 版本的数据整理逻辑是否满足你的 causal mask / loss mask 预期，尤其是多轮对话和自定义模板场景。

### 8.4 什么时候不该用 packing

- 样本长度本身接近 max_seq_length → packing 收益小
- 需要精确的 per-sample loss → packing 会混合多个样本的 loss
- 长上下文任务（如 RAG）→ 样本本身就很长

---

## 9. 超参数选择

### 9.1 推荐配置

| 超参数 | 推荐范围 | 说明 |
|---|---|---|
| learning_rate | 1e-5 ~ 5e-5 (full FT) / 1e-4 ~ 3e-4 (LoRA) | LoRA lr 要高 |
| num_epochs | 1–3 | **SFT 非常容易过拟合** |
| batch_size (effective) | 64–256 | = per_device × gradient_accum × num_gpus |
| warmup_ratio | 0.03–0.1 | 总步数的 3-10% |
| lr_scheduler | cosine | 比 linear 好 |
| weight_decay | 0.01–0.1 | 防止过拟合 |
| max_seq_length | 2048–8192 | 看数据分布和显存 |

### 9.2 数据量与 epoch 的关系

| 数据量 | 推荐 epoch | 说明 |
|---|---|---|
| < 1K | 5–10 | 数据少需要多看几遍 |
| 1K–10K | 3–5 | 正常范围 |
| 10K–100K | 1–3 | 大数据集 1-2 epoch 够了 |
| > 100K | 1 | 1 epoch 通常最佳 |

---

## 10. 评估策略

### 10.1 训练期间监控

```python
# 在 TrainingArguments 中设置
training_args = SFTConfig(
    ...,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)
```

### 10.2 Loss 曲线诊断

```
正常：train_loss 平稳下降，eval_loss 先降后平
过拟合：train_loss 持续降，eval_loss 开始上升
欠拟合：两者都很高且不下降 → lr 太小或数据太少
```

### 10.3 Benchmark 评估

```python
# 使用 lm-evaluation-harness
# pip install lm-eval

# 命令行评估
# lm_eval --model hf \
#     --model_args pretrained=./sft_output \
#     --tasks mmlu,hellaswag,arc_easy \
#     --batch_size 8
```

### 10.4 人工评估

对于聊天模型，自动指标不够，需要人工或 GPT-4 评估：

```python
def gpt4_judge(instruction: str, response_a: str, response_b: str) -> str:
    """用 GPT-4 做 A/B 对比评估"""
    prompt = f"""请评判以下两个回复的质量。

指令：{instruction}

回复A：{response_a}

回复B：{response_b}

请从有帮助性、准确性、详细程度三个维度评分（1-5），并给出总体判断（A好/B好/平局）。"""

    # 调用 GPT-4 API ...
```

---

## 11. 多轮对话微调

### 11.1 数据格式

```json
{
  "messages": [
    {"role": "system", "content": "你是一个 Python 编程助手。"},
    {"role": "user", "content": "如何排序一个列表？"},
    {"role": "assistant", "content": "可以用 sorted() 或 list.sort()..."},
    {"role": "user", "content": "如果要按第二个元素排序呢？"},
    {"role": "assistant", "content": "使用 key 参数: sorted(lst, key=lambda x: x[1])"}
  ]
}
```

### 11.2 Loss masking 策略

多轮对话中，**所有 assistant 回复**都计算 loss：

```
[system]...[user]轮1...[assistant]轮1...[user]轮2...[assistant]轮2...
     ↑ mask         ↑ mask        ↑ LOSS      ↑ mask        ↑ LOSS
```

---

## 12. 完整 SFT 训练脚本

```python
"""完整的 SFT 训练脚本（QLoRA + TRL）"""
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ============ 配置 ============
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH = "./sft_data.jsonl"
OUTPUT_DIR = "./sft_output"

# ============ 1. 加载模型 ============
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============ 2. LoRA 配置 ============
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ============ 3. 加载数据 ============
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.05, seed=42)

# ============ 4. 训练配置 ============
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    max_seq_length=4096,
    packing=False,
    report_to="wandb",
)

# ============ 5. 训练 ============
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    peft_config=lora_config,
)

trainer.train()

# ============ 6. 保存 ============
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
```

---

## 13. 常见坑

### 13.1 Catastrophic Forgetting（灾难性遗忘）

```python
# 症状：微调后模型忘记了通用能力
# 原因：数据太窄、epoch 太多、lr 太大

# 缓解方案：
# 1. 混合通用数据（10-20%）
# 2. 减少 epoch（1-3）
# 3. 用 LoRA 而非 full FT
# 4. 降低 learning rate
```

### 13.2 过拟合

```python
# 症状：train_loss 极低但 eval_loss / 实际效果变差
# 判断：如果 train_loss < 0.5 且 eval_loss > 1.5 → 严重过拟合

# 缓解：
# 1. 减少 epoch
# 2. 增加 dropout（lora_dropout=0.1）
# 3. 增加数据量
# 4. 减小 max_seq_length 避免 padding 过多
```

### 13.3 Chat Template 不匹配

```python
# ❌ 手动拼格式字符串
text = f"[INST] {instruction} [/INST] {response}"

# ✓ 用 tokenizer 的 chat template
text = tokenizer.apply_chat_template(messages, tokenize=False)
```

### 13.4 数据泄露

```python
# ❌ eval 数据和 train 数据有重叠
# ✓ 先去重，再 split

dataset = dataset.shuffle(seed=42)
dataset = dedup_by_instruction(dataset)
split = dataset.train_test_split(test_size=0.05, seed=42)
```

### 13.5 padding_side 设置错误

```python
# ❌ Causal LM 用了 left padding 训练
tokenizer.padding_side = "left"  # 训练时这是错的

# ✓ 训练时用 right padding
tokenizer.padding_side = "right"

# 注意：推理时 batch decode 需要 left padding
# 训练用 right，推理用 left
```

### 13.6 gradient_checkpointing 与 use_reentrant

```python
# ❌ PyTorch 2.0+ 默认 use_reentrant=True，与 LoRA 有冲突
gradient_checkpointing=True

# ✓ 显式设置
gradient_checkpointing=True,
gradient_checkpointing_kwargs={"use_reentrant": False},
```

---

## 小结

| 概念 | 记忆要点 |
|---|---|
| 数据格式 | 统一用 messages 格式（OpenAI 风格） |
| Chat Template | **永远用 tokenizer.apply_chat_template()** |
| 数据质量 | 1K 精选 > 100K 噪声 |
| Loss Mask | 只在 assistant 回复上算 loss |
| Packing | 短文本拼起来，提高 GPU 利用率 |
| Epochs | 大数据 1 epoch，小数据 3-5 epochs |
| 过拟合 | SFT 最大的敌人，时刻监控 eval_loss |
| LR | Full FT 用 2e-5，LoRA 用 2e-4 |

**一句话**：**数据决定上限，训练技巧决定逼近上限的速度。花 80% 时间在数据上。**
