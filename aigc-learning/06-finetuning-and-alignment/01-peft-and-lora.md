# 01 · PEFT 与 LoRA：低成本微调大模型

> 推荐等级：**★★★★★**
> LoRA 让单卡 24GB 就能微调 7B 模型——这是 AIGC 工程师的基本功。

---

## 0. 本地可运行示例

本节的 LoRA 低秩更新、冻结基座参数和合并权重，可以先用 CPU 小实验验证：

```bash
cd aigc-learning/06-finetuning-and-alignment/examples
conda run -n aigc python lora_tiny_train.py --epochs 6 --rank 2
```

脚本用一个线性层模拟 `W' = W + BA`，会输出 full fine-tuning 和 LoRA 的可训练参数量、验证集 MSE，以及 LoRA adapter 合并后的权重误差。

---

## 1. 问题：全量微调太贵

全量微调（Full Fine-Tuning）一个 7B 模型需要什么？

| 项目 | 估算 |
|---|---|
| 模型参数 | 7B × 4 bytes (FP32) = 28 GB |
| 梯度 | 28 GB |
| 优化器状态（Adam） | 56 GB（2× 动量） |
| **总计** | ~112 GB，至少 2× A100-80GB |

而且：每个下游任务保存一份完整模型，存储和发布成本都会迅速膨胀。

**PEFT 的核心思想**：冻结大部分参数，只训一小部分 → 省显存、省计算、省存储。

---

## 2. PEFT 方法全景

```
┌───────────────────────────────────────────────────┐
│              Parameter-Efficient Fine-Tuning        │
├───────────────────────────────────────────────────┤
│                                                   │
│  加法类（Addition）        重参数化（Reparametrization）│
│  ┌──────────────┐        ┌──────────────────┐    │
│  │ Prefix Tuning│        │ LoRA             │    │
│  │ Prompt Tuning│        │ QLoRA            │    │
│  │ Adapter      │        │ DoRA             │    │
│  │ (IA)³        │        │ AdaLoRA          │    │
│  └──────────────┘        └──────────────────┘    │
│                                                   │
│  选择类（Selection）                               │
│  ┌──────────────┐                                │
│  │ BitFit       │                                │
│  │ Partial FT   │                                │
│  └──────────────┘                                │
└───────────────────────────────────────────────────┘
```

**实战中 90% 的场景用 LoRA / QLoRA 就够了。**

---

## 3. LoRA 数学原理

### 3.1 核心公式

对于预训练权重矩阵 `W ∈ ℝ^(d×k)`：

```
W' = W + ΔW = W + B·A

其中：
  W: 原始权重（冻结）        d × k
  A: 下投影矩阵（可训练）     r × k    （初始化为高斯随机）
  B: 上投影矩阵（可训练）     d × r    （初始化为零）
  r: 秩（rank），通常 r << min(d, k)

前向传播：
  h = W·x + (α/r) · B·A·x

  α: scaling factor，控制 LoRA 的"修改强度"
```

### 3.2 为什么有效？

论文的关键 insight：**预训练模型的权重更新具有低秩结构**。

- 7B 模型某一层 `W` 是 `4096 × 4096`（~16M 参数）
- 用 rank=8 的 LoRA：`A` 是 `8×4096`，`B` 是 `4096×8`（~65K 参数）
- **参数量压缩比：16M / 65K ≈ 250×**

### 3.3 PyTorch 手动实现

```python
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 原始权重（冻结）
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features), requires_grad=False
        )

        # LoRA 低秩矩阵
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始前向 + LoRA 修正
        base_out = x @ self.weight.T
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out
```

### 3.4 rank 和 alpha 怎么选？

| 场景 | rank | alpha | 说明 |
|---|---|---|---|
| 轻量适配（风格迁移） | 4–8 | 16 | 任务简单，少量参数即可 |
| 通用微调 | 16–32 | 32 | 大部分 SFT 任务 |
| 复杂任务（代码、数学） | 64–128 | 128 | 需要较大表达能力 |
| 全量替代 | 256+ | 256 | 接近 full fine-tuning |

**经验法则**：`alpha = 2 × rank` 是常见默认配置。

---

## 4. QLoRA：单卡微调 70B 的方案

QLoRA = **4-bit 量化基座** + **LoRA 适配器**（全精度训练）

三大核心技术：

| 技术 | 作用 |
|---|---|
| NF4 (NormalFloat4) | 比均匀 INT4 更适合正态分布权重 |
| Double Quantization | 对量化常数再量化，省 ~0.5 GB |
| Paged Optimizers | 显存不足时把优化器状态卸载到 CPU |

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 4-bit 量化加载
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

# 准备 k-bit 训练
model = prepare_model_for_kbit_training(model)

# 添加 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 13,631,488 || all params: 8,043,847,680 || trainable%: 0.17%
```

**显存占用对比（Llama-3.1-8B）**：

| 方法 | 显存 |
|---|---|
| Full FT (FP16) | ~60 GB |
| LoRA (FP16 base) | ~35 GB |
| QLoRA (4-bit base) | ~12 GB ✓ 单卡 4090 |

---

## 5. DoRA：权重分解低秩适配

DoRA (Weight-Decomposed Low-Rank Adaptation) 将权重分解为**方向**和**大小**：

```
W = m · (V / ||V||)

其中：
  m: magnitude（标量/向量）—— 可训练
  V: direction matrix —— 用 LoRA 更新

W' = (m + Δm) · (V + ΔV) / ||V + ΔV||
```

DoRA 在多数 benchmark 上优于 LoRA，且参数量增加极少。

```python
from peft import LoraConfig

# DoRA 只需设置 use_dora=True
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    use_dora=True,  # ← 启用 DoRA
)
```

---

## 6. 其他 PEFT 方法速览

### 6.1 Prefix Tuning

在每一层的 KV cache 前面拼接可学习的"虚拟 token"：

```python
from peft import PrefixTuningConfig

config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
)
```

### 6.2 Prompt Tuning

只在输入 embedding 前拼接可学习 token（比 Prefix Tuning 更轻量）：

```python
from peft import PromptTuningConfig, PromptTuningInit

config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=8,
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Classify the following text:",
    tokenizer_name_or_path="meta-llama/Llama-3.1-8B",
)
```

### 6.3 (IA)³

只学习三个缩放向量（key, value, FFN），极轻量：

```python
from peft import IA3Config

config = IA3Config(
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],
)
```

### 6.4 方法对比

| 方法 | 可训练参数 | 推理开销 | 适用场景 |
|---|---|---|---|
| LoRA | 0.1–1% | 可合并无开销 | **通用首选** |
| QLoRA | 0.1–1% | 量化推理 | 显存受限 |
| DoRA | ~0.1–1% | 略高于 LoRA | 追求更好效果 |
| Prefix Tuning | <0.1% | 有额外 KV | 多任务切换 |
| Prompt Tuning | <0.01% | 有额外 token | 极大模型/极多任务 |
| (IA)³ | <0.01% | 几乎无 | Few-shot 场景 |
| AdaLoRA | 0.1–1% | 可合并 | 自动分配 rank |

---

## 7. 实战：目标层选择

### 7.1 哪些层加 LoRA？

```python
# 查看模型结构
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name, module.in_features, module.out_features)

# 常见选择（Llama 系列）
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",       # MLP (FFN)
]
```

**经验**：

| 策略 | target_modules | 效果 |
|---|---|---|
| 保守（省显存） | `["q_proj", "v_proj"]` | 够用但不充分 |
| 推荐 | Attention 全部 + MLP 全部 | 效果好，显存可接受 |
| all-linear | 所有 Linear 层 | 接近 full FT |

### 7.2 learning rate 选择

LoRA 的最佳 lr 通常比全量微调**高 5-10 倍**：

| 方法 | 推荐 lr |
|---|---|
| Full Fine-Tuning | 1e-5 ~ 5e-5 |
| LoRA | 1e-4 ~ 3e-4 |
| QLoRA | 1e-4 ~ 2e-4 |

---

## 8. LoRA 合并与部署

### 8.1 合并回基座

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
peft_model = PeftModel.from_pretrained(base_model, "./lora_adapter")

# 合并权重
merged_model = peft_model.merge_and_unload()

# 保存合并后的模型
merged_model.save_pretrained("./merged_model")
```

合并后的好处：
- 推理时没有额外开销
- 可以进一步做 GPTQ/AWQ 量化
- 兼容所有推理框架（vLLM, TGI, llama.cpp）

### 8.2 多 LoRA 服务

不合并，保留 adapter，推理时动态切换：

```python
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model = PeftModel.from_pretrained(model, "./lora_chinese")

# 加载另一个 adapter
model.load_adapter("./lora_code", adapter_name="code")

# 切换
model.set_adapter("code")
output = model.generate(...)

# 切回
model.set_adapter("default")
```

vLLM 原生支持多 LoRA 并发服务：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B \
    --enable-lora \
    --lora-modules chinese=./lora_chinese code=./lora_code
```

---

## 9. peft 库核心 API

```python
from peft import (
    get_peft_model,           # 给模型添加 PEFT 层
    PeftModel,                # 加载已保存的 PEFT 模型
    LoraConfig,               # LoRA 配置
    TaskType,                 # CAUSAL_LM, SEQ_CLS, ...
    prepare_model_for_kbit_training,  # QLoRA 必须
)

# === 训练流程 ===
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",  # 自动匹配所有 Linear
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, config)

# 训练完保存（只保存 adapter，几十 MB）
model.save_pretrained("./my_adapter")

# === 加载流程 ===
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model = PeftModel.from_pretrained(base_model, "./my_adapter")
```

---

## 10. 完整示例：QLoRA 微调 Llama-3.1-8B

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 1. 量化加载模型
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model_id = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 2. 准备 QLoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. 加载数据
dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

def format_instruction(sample):
    if sample["input"]:
        return f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n{sample['output']}"
    return f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"

# 4. 训练
training_config = SFTConfig(
    output_dir="./qlora_llama3_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    max_seq_length=2048,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=format_instruction,
    args=training_config,
    tokenizer=tokenizer,
)

trainer.train()

# 5. 保存 adapter
model.save_pretrained("./qlora_llama3_adapter")
tokenizer.save_pretrained("./qlora_llama3_adapter")
```

---

## 11. 常见坑

### 11.1 忘记冻结基座

```python
# ❌ 错误：没用 get_peft_model 就手动加了 LoRA 层，但没冻结原参数
# ✓ 正确：用 peft 库自动管理冻结

model = get_peft_model(model, lora_config)  # 自动冻结非 LoRA 参数
```

### 11.2 QLoRA 忘记 prepare_model_for_kbit_training

```python
# ❌ 量化加载后直接加 LoRA
model = get_peft_model(model, config)  # 梯度计算会出问题

# ✓ 先 prepare
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, config)
```

### 11.3 target_modules 写错名字

```python
# ❌ 用了 GPT-2 的命名去微调 Llama
target_modules = ["c_attn", "c_proj"]  # 这是 GPT-2 的

# ✓ 查看模型的实际 module name
for name, _ in model.named_modules():
    print(name)
```

### 11.4 合并后再量化精度下降

```python
# ❌ QLoRA adapter + 4-bit base → merge → 再 4-bit 量化 = 双重量化损失
# ✓ 先把 base 加载为 FP16 → merge adapter → 再做量化
```

### 11.5 学习率太低

```python
# ❌ 用 full FT 的 lr
lr = 2e-5  # LoRA 参数太少，这个 lr 学不动

# ✓ LoRA 推荐
lr = 2e-4  # 高 10 倍
```

---

## 小结

| 概念 | 记忆要点 |
|---|---|
| LoRA | `W' = W + (α/r)·B·A`，训 A 和 B |
| QLoRA | 4-bit base + FP16 LoRA + Double Quant + Paged Opt |
| DoRA | 分方向（LoRA 更新）和大小（单独学） |
| rank | 越大越接近 full FT，越小越省 |
| alpha | 通常 = 2×rank |
| target_modules | 推荐 all-linear 或至少 qkvo + mlp |
| 合并 | `merge_and_unload()` → 零推理开销 |
| 多 LoRA | 不合并，动态 `set_adapter()` |

**一句话**：**先 QLoRA 跑通，效果不够再提 rank，还不够就上 full FT。**
