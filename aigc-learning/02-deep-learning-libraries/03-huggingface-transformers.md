# 03 · HuggingFace Transformers

> `transformers` 是现代 NLP/LLM 事实标准，覆盖 10 万+ 预训练模型。
> 本节带你吃透它的四个核心抽象：**Tokenizer / Model / Pipeline / Trainer**，
> 并且动手用 PEFT 给 LLM 加 LoRA。

---

## 1. 快速上手

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "你是一个乐于助人的助手"},
    {"role": "user", "content": "用一句话介绍扩散模型"},
]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

短短 10 行代码，调用了一个完整的 LLM。

---

## 2. Tokenizer：文本 ↔ Token IDs

### 2.1 基本用法

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")

ids = tok.encode("Hello, world!")
text = tok.decode(ids)

tokens = tok.tokenize("Hello, world!")

out = tok(
    ["hello", "how are you?"],
    padding=True,
    truncation=True,
    max_length=32,
    return_tensors="pt",
)
print(out["input_ids"].shape, out["attention_mask"].shape)
```

### 2.2 Chat Template（LLM 必备）

现代 LLM 的"对话格式"是由模型自己定义的特殊模板：

```python
messages = [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "what's your name?"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)
print(prompt)
```

**不要自己拼接对话字符串**——每个模型的模板都不同（Qwen / Llama / ChatGLM / Mistral 各不一样）。

### 2.3 特殊 token

```python
print(tok.bos_token, tok.eos_token, tok.pad_token)
print(tok.bos_token_id, tok.eos_token_id)

tok.pad_token = tok.eos_token
```

---

## 3. Model：模型加载与使用

### 3.1 AutoModel 家族

| 类 | 用途 |
|---|---|
| `AutoModel` | 只输出 hidden states |
| `AutoModelForCausalLM` | 因果 LM（GPT / Llama / Qwen） |
| `AutoModelForSeq2SeqLM` | 编码器-解码器（T5 / BART） |
| `AutoModelForSequenceClassification` | 文本分类 |
| `AutoModelForTokenClassification` | NER |
| `AutoModelForMaskedLM` | BERT 类 |

### 3.2 生成参数详解

```python
out = model.generate(
    input_ids=inputs["input_ids"],
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.05,
    pad_token_id=tokenizer.eos_token_id,
)
```

- `temperature`：0 → 贪心，越大越随机。
- `top_p`：nucleus sampling（保留累积概率 p 的 token）。
- `top_k`：只从前 k 个候选采样。
- 做聊天 bot → `do_sample=True`；做抽取/分类 → `do_sample=False`。

### 3.3 流式生成（服务化常用）

```python
from transformers import TextIteratorStreamer
from threading import Thread

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
thread = Thread(target=model.generate, kwargs={**inputs, "streamer": streamer, "max_new_tokens": 256})
thread.start()

for token in streamer:
    print(token, end="", flush=True)
```

---

## 4. Pipeline：一行代码搞定推理

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2", device=0)
print(generator("Once upon a time", max_new_tokens=50))

sentiment = pipeline("sentiment-analysis")
print(sentiment(["I love this", "This is terrible"]))

ner = pipeline("ner", aggregation_strategy="simple")
print(ner("My name is Alice and I work at OpenAI."))

emb = pipeline("feature-extraction", model="BAAI/bge-small-en-v1.5")
vec = emb("hello world")
```

适合快速 demo；**生产环境建议用 vLLM / SGLang**。

---

## 5. 微调：Trainer API

### 5.1 完整最小示例

```python
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments,
)
from datasets import load_dataset

model_name = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

ds = load_dataset("imdb")
def preprocess(batch):
    return tok(batch["text"], truncation=True, max_length=256)
ds = ds.map(preprocess, batched=True)

args = TrainingArguments(
    output_dir="./ckpt",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"].shuffle(seed=42).select(range(5000)),
    eval_dataset=ds["test"].select(range(1000)),
    tokenizer=tok,
)
trainer.train()
```

`Trainer` 已经帮你处理了：DDP、混合精度、梯度累积、日志、checkpoint、eval。

---

## 6. PEFT：参数高效微调（LoRA）

常见设置下，全参微调 8B 模型往往需要几十 GB 甚至更高显存；LoRA 只训练少量低秩适配器，显存压力会明显下降。实际占用还取决于序列长度、batch size、优化器状态、梯度检查点、量化方式和是否使用 ZeRO/FSDP。

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
```

配合 `Trainer` 正常训练即可，只会训练 LoRA 参数。

### 6.1 LoRA 关键参数

- `r`：低秩矩阵的秩，常见 8 / 16 / 32 / 64。越大效果越好，显存也越大。
- `lora_alpha`：缩放因子，一般 `alpha = 2 * r`。
- `target_modules`：对哪些层加 LoRA（通常是注意力的 QKV + 输出）。

### 6.2 QLoRA：4bit 量化 + LoRA

显存再砍一半：

```python
from transformers import BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb,
    device_map="auto",
)
```

---

## 7. `TRL`：SFT / DPO / GRPO 对齐训练

`trl` 是 HuggingFace 专门做 LLM 对齐的库：

> 本节需要额外安装 `trl`。如果只是运行本模块的离线 smoke test，可以先跳过。

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(output_dir="./sft-ckpt", max_seq_length=2048),
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_cfg,
)
trainer.train()
```

支持的算法：SFT / DPO / IPO / KTO / GRPO / PPO / RewardTrainer。
做大模型对齐几乎绕不开它。

---

## 8. `datasets`：数据加载与处理

```python
from datasets import load_dataset, Dataset

ds = load_dataset("wikipedia", "20220301.en", split="train[:1%]")

ds = Dataset.from_dict({"text": ["hello", "world"], "label": [0, 1]})

ds = ds.map(lambda x: {"tokens": tok(x["text"])["input_ids"]}, batched=True)

ds = ds.filter(lambda x: len(x["text"]) > 10)

ds = ds.shuffle(seed=42).train_test_split(test_size=0.1)
```

大数据集（TB 级）用 `load_dataset(..., streaming=True)` 流式读取。

---

## 9. 常见坑

### 9.1 `pad_token` 未设置

很多因果 LM 没 pad token，需要：

```python
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
```

### 9.2 `device_map="auto"` 与手动 `.to(device)` 冲突

用了 `device_map="auto"` 就不要再 `.to(device)`，会报错。

### 9.3 推理时忘了 `model.eval()`

有 Dropout/BatchNorm 的模型推理时会不稳定。
`pipeline` 和 `generate` 内部已经处理好了；手写循环时要注意。

### 9.4 内存爆炸

- 长文本场景：`max_length` 开得太大。
- 推理时：没开 `torch.no_grad()`。
- 训练时：没用梯度检查点 `model.gradient_checkpointing_enable()`。

---

## 小结

| 抽象 | 一句话记忆 |
|---|---|
| `AutoTokenizer` | 文本 ↔ token ids，记得用 chat template |
| `AutoModelFor*` | 模型加载；根据任务选对 `For*` 后缀 |
| `pipeline` | 一行代码 demo 神器 |
| `Trainer` | 标准化训练循环 |
| `peft` + `LoraConfig` | LoRA 微调 |
| `trl` | SFT / DPO / GRPO 对齐 |
| `datasets` | 统一数据加载 |

配套示例：

```bash
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/transformers_quickstart.py
```

该命令默认使用本地随机初始化的极小 GPT-2 和自定义 tokenizer，用于离线验证 `tokenizer`、`generate`、`TextIteratorStreamer`、`pipeline` 的 API 链路。要加载真实 HuggingFace 模型时再运行：

```bash
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/transformers_quickstart.py --real-model
```

下一节进入 **HuggingFace Diffusers**，AIGC 图像/视频生成的基石。
