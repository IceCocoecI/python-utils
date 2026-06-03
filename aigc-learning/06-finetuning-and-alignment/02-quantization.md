# 02 · 量化：让大模型跑在消费级硬件上

> 推荐等级：**★★★★★**
> 70B 模型用 FP16 需要 140GB 显存，4-bit 量化后只需 ~35GB。
> 量化是大模型从实验室走向生产环境的关键桥梁。

---

## 1. 为什么需要量化？

| 精度 | 每参数字节 | 7B 模型显存 | 70B 模型显存 |
|---|---|---|---|
| FP32 | 4 | 28 GB | 280 GB |
| FP16 / BF16 | 2 | 14 GB | 140 GB |
| INT8 | 1 | 7 GB | 70 GB |
| INT4 | 0.5 | 3.5 GB | 35 GB |

量化的目标：**用更低精度存储和计算，在几乎不损失质量的前提下，大幅降低显存和提升速度。**

---

## 2. 数字表示基础

### 2.1 浮点格式对比

```
FP32:  1 bit sign | 8 bits exponent  | 23 bits mantissa   (精度高，慢)
FP16:  1 bit sign | 5 bits exponent  | 10 bits mantissa   (易溢出)
BF16:  1 bit sign | 8 bits exponent  | 7 bits mantissa    (范围大，精度低)
FP8:   1 bit sign | 4/5 bits exp     | 2/3 bits mantissa  (Hopper GPU)
```

### 2.2 整数量化原理

**线性量化（对称）**：

```
x_quant = round(x / scale)
x_dequant = x_quant × scale

scale = max(|x|) / (2^(n-1) - 1)
```

**线性量化（非对称）**：

```
x_quant = round(x / scale) + zero_point
x_dequant = (x_quant - zero_point) × scale

scale = (max(x) - min(x)) / (2^n - 1)
zero_point = round(-min(x) / scale)
```

### 2.3 分组量化（Group Quantization）

不对整个张量用一个 scale，而是每 `group_size`（通常 128）个元素用一个 scale：

```
W: [4096, 4096]
group_size = 128
→ 每行 4096 / 128 = 32 组
→ 需要 4096 × 32 个 scale 值
```

分组量化是现代 INT4 方法的标配。

---

## 3. PTQ vs QAT

| 方法 | 全称 | 是否需要训练 | 精度 | 速度 |
|---|---|---|---|---|
| PTQ | Post-Training Quantization | 不需要（可能需要校准数据） | 略低 | 快 |
| QAT | Quantization-Aware Training | 需要反向传播 | 高 | 很慢 |

**LLM 场景几乎都用 PTQ**（因为模型太大，QAT 成本极高）。

---

## 4. bitsandbytes：最简单的量化方案

### 4.1 8-bit 量化

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    load_in_8bit=True,
    device_map="auto",
)
```

原理：LLM.int8() 论文——对离群特征（outlier features）保留 FP16，其余 INT8。

### 4.2 4-bit 量化（NF4）

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算用 BF16
    bnb_4bit_use_double_quant=True,      # 二次量化（省显存）
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)
```

### 4.3 NF4 vs FP4

| 量化类型 | 原理 | 效果 |
|---|---|---|
| FP4 | 均匀 4-bit 浮点 | 一般 |
| NF4 | 假设权重服从正态分布，非均匀量化 | **更好** |

NF4 的量化点分布：按正态分布的分位数放置 16 个量化值，使得量化误差在正态分布权重上最小。

### 4.4 Double Quantization

对量化常数（scale）本身再量化：

```
原始：每 64 个权重一个 FP32 scale → 每参数额外 0.5 bit
Double Quant：scale 再量化为 FP8 → 每参数额外 ~0.127 bit
```

省约 0.37 bit/参数 → 对 70B 模型省 ~3.3 GB。

---

## 5. GPTQ：基于二阶信息的精确量化

### 5.1 算法原理

GPTQ 基于 Optimal Brain Quantization 框架：

1. 收集少量**校准数据**（128 条文本）
2. 逐层量化：用 Hessian 信息（二阶导数）决定最优量化顺序
3. 量化某列后，对**未量化列**做补偿（减小累积误差）

### 5.2 使用 AutoGPTQ

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 量化配置
gptq_config = GPTQConfig(
    bits=4,
    group_size=128,
    dataset="c4",             # 校准数据集
    desc_act=True,            # 按激活降序量化（更准但更慢）
)

# 量化
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=gptq_config,
    device_map="auto",
)

# 保存量化模型
model.save_pretrained("./llama3-8b-gptq-4bit")
tokenizer.save_pretrained("./llama3-8b-gptq-4bit")
```

### 5.3 加载已量化模型

```python
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    device_map="auto",
)
```

---

## 6. AWQ：激活感知权重量化

### 6.1 核心思想

不是所有权重同等重要。AWQ 发现：**1% 的"显著权重"对模型质量影响巨大**。

策略：根据激活值的分布，对显著权重通道做缩放（保护），然后再量化。

```
AWQ vs GPTQ:
  GPTQ: 逐列量化 + Hessian 补偿（更精确，更慢）
  AWQ:  通道缩放 + 分组量化（更快，推理也更快）
```

### 6.2 使用 AutoAWQ

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-3.1-8B"
quant_path = "./llama3-8b-awq-4bit"

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 量化
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
}
model.quantize(tokenizer, quant_config=quant_config)

# 保存
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### 6.3 用 transformers 加载 AWQ 模型

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "casperhansen/llama-3-8b-instruct-awq",
    device_map="auto",
)
```

---

## 7. GGUF 与 llama.cpp

### 7.1 GGUF 格式

GGUF 是 llama.cpp 使用的模型格式，支持 CPU + GPU 混合推理。

### 7.2 量化方法命名

| 方法 | 说明 | 推荐度 |
|---|---|---|
| Q4_0 | 4-bit，无分组 | 质量差 |
| Q4_K_M | 4-bit，K-quant，medium | **推荐** |
| Q5_K_M | 5-bit，K-quant，medium | 质量好 |
| Q6_K | 6-bit，K-quant | 几乎无损 |
| Q8_0 | 8-bit，简单 | 最高质量 |
| IQ4_XS | 4-bit，importance-based | 同 bit 最好 |

### 7.3 转换与量化

```bash
# 从 HuggingFace 格式转换
python convert_hf_to_gguf.py ./llama3-8b/ --outtype f16 --outfile llama3-8b-f16.gguf

# 量化
./llama-quantize llama3-8b-f16.gguf llama3-8b-Q4_K_M.gguf Q4_K_M

# 推理
./llama-cli -m llama3-8b-Q4_K_M.gguf -p "Hello, world!" -n 100
```

### 7.4 选择建议

| 显存/内存 | 7B 推荐 | 70B 推荐 |
|---|---|---|
| 8 GB | Q4_K_M | 不可行 |
| 16 GB | Q5_K_M 或 Q6_K | Q4_K_M (需 offload) |
| 32 GB | Q8_0 或 FP16 | Q4_K_M |
| 64 GB+ | FP16 | Q5_K_M |

---

## 8. FP8 量化（Hopper / Ada GPU）

NVIDIA H100/H200 和 RTX 4090 支持 FP8 硬件加速：

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# FP8 量化（需要 bitsandbytes >= 0.43 或 transformers FP8 支持）
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# 或者使用 vLLM 的 FP8 支持
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-3.1-8B \
#     --quantization fp8
```

FP8 的优势：精度损失极小（接近 FP16），速度提升 ~2×。

---

## 9. 量化方法全面对比

| 方法 | Bits | 需要校准 | 推理速度 | 质量 | GPU 推理 | CPU 推理 | 可 LoRA |
|---|---|---|---|---|---|---|---|
| bitsandbytes 8-bit | 8 | 否 | 慢 | 高 | ✓ | ✗ | ✓ |
| bitsandbytes 4-bit | 4 | 否 | 中 | 中 | ✓ | ✗ | ✓ (QLoRA) |
| GPTQ | 4/3/2 | 是 | **快** | 中高 | ✓ | ✗ | ✓ |
| AWQ | 4 | 是 | **快** | 中高 | ✓ | ✗ | ✓ |
| GGUF (llama.cpp) | 2–8 | 否 | 快(CPU) | 可选 | ✓ | **✓** | ✗ |
| FP8 | 8 | 否 | **最快** | 高 | ✓(H100) | ✗ | ✓ |

**选择流程**：

```
需要 QLoRA 训练？ → bitsandbytes 4-bit (NF4)
GPU 推理部署？   → AWQ 或 GPTQ（vLLM 都支持）
CPU / 边缘部署？ → GGUF (llama.cpp)
有 H100？       → FP8
```

---

## 10. BitsAndBytesConfig 完整参考

```python
from transformers import BitsAndBytesConfig
import torch

config = BitsAndBytesConfig(
    # === 8-bit ===
    load_in_8bit=False,
    llm_int8_threshold=6.0,          # 离群特征阈值
    llm_int8_has_fp16_weight=False,

    # === 4-bit ===
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # "nf4" 或 "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16,  # 矩阵乘法精度
    bnb_4bit_use_double_quant=True,  # 二次量化
    bnb_4bit_quant_storage=torch.uint8,     # 存储类型
)
```

---

## 11. 量化质量评估

### 11.1 Perplexity 对比

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def compute_perplexity(model, tokenizer, dataset, max_length=2048):
    model.eval()
    total_loss = 0
    total_tokens = 0

    for text in dataset["text"][:100]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=max_length).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].numel()
        total_tokens += inputs["input_ids"].numel()

    return torch.exp(torch.tensor(total_loss / total_tokens)).item()
```

### 11.2 典型 Perplexity 变化（Llama-2-7B, WikiText-2）

| 方法 | Perplexity | 相对增加 |
|---|---|---|
| FP16（基准） | 5.47 | — |
| GPTQ 4-bit | 5.63 | +2.9% |
| AWQ 4-bit | 5.60 | +2.4% |
| bitsandbytes NF4 | 5.72 | +4.6% |
| GGUF Q4_K_M | 5.65 | +3.3% |
| GPTQ 3-bit | 6.12 | +11.9% |

**结论**：4-bit 量化的质量损失通常 < 5%，完全可接受。

---

## 12. 常见坑

### 12.1 量化模型不能直接训练

```python
# ❌ 量化后直接 backward
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)
output = model(input_ids)
output.loss.backward()  # RuntimeError: 量化层不支持梯度

# ✓ 必须配合 LoRA 使用
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
```

### 12.2 GPTQ 校准数据选择

```python
# ❌ 用目标域数据做校准（可能导致过拟合特定分布）
# ✓ 用通用文本（C4, WikiText）做校准

# GPTQ 校准数据只需要 128-256 条，不要太多
```

### 12.3 compute_dtype 设置错误

```python
# ❌ 4-bit 加载但用 FP32 计算 → 速度很慢
bnb_4bit_compute_dtype=torch.float32

# ✓ 用 BF16（A100/H100）或 FP16（其他 GPU）
bnb_4bit_compute_dtype=torch.bfloat16
```

### 12.4 量化 + Flash Attention 兼容性

```python
# 某些量化配置与 flash_attention_2 不兼容
# ✓ 确保 compute_dtype 与 Flash Attention 要求一致
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",  # 需要 compute_dtype 为 fp16/bf16
)
```

### 12.5 device_map 与多 GPU

```python
# ❌ 量化模型手动 .to(device) 会报错
model = model.to("cuda:0")  # Error

# ✓ 必须使用 device_map
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",  # 自动分配到可用 GPU
)
```

---

## 小结

| 概念 | 记忆要点 |
|---|---|
| FP16/BF16 | 半精度，2 bytes/param，训练推理标配 |
| NF4 | 正态分布优化的 4-bit，QLoRA 用 |
| GPTQ | 有校准、Hessian 补偿、推理快 |
| AWQ | 激活感知缩放、推理最快 |
| GGUF | llama.cpp 格式，CPU 推理首选 |
| FP8 | H100 硬件加速，几乎无损 |
| Double Quant | 省额外 ~3GB/70B |
| 选择标准 | 训练→bnb / GPU推理→AWQ / CPU→GGUF |

**一句话**：**部署选 AWQ/GPTQ + vLLM，训练选 bitsandbytes NF4 + LoRA，边缘设备选 GGUF。**
