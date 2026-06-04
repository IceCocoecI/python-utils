# 04 · 对齐：RLHF、DPO 与 GRPO

> 推荐等级：**★★★★★**
> SFT 让模型"能做事"，对齐让模型"做好事"——更安全、更有帮助、更诚实。

---

## 0. 本地可运行示例

先用小张量验证 Reward Model pair loss、DPO margin、IPO loss 和 GRPO 组内 advantage：

```bash
cd aigc-learning/06-finetuning-and-alignment/examples
conda run -n aigc python dpo_loss_demo.py --beta 0.1
```

脚本不会训练大模型，而是直接计算偏好优化的核心量，便于把公式和数值对应起来。

---

## 1. 为什么需要对齐？

SFT 模型的问题：
- **有害输出**：会回答"如何制造炸弹"
- **幻觉**：编造不存在的论文和事实
- **谄媚**：只说用户爱听的话
- **冗长**：废话连篇

对齐的目标（HHH 原则）：

| 原则 | 含义 | 例子 |
|---|---|---|
| Helpful | 有帮助 | 回答准确、完整、有条理 |
| Harmless | 无害 | 拒绝危险请求 |
| Honest | 诚实 | 承认不确定，不编造 |

---

## 2. 对齐流水线全景

```
┌────────────────────────────────────────────────────────────────┐
│                       经典 RLHF 流水线                           │
│                                                                │
│  ┌─────┐    ┌───────────┐    ┌──────────────┐    ┌─────────┐ │
│  │ SFT │ ──→│ Reward    │ ──→│ PPO Training │ ──→│ Aligned │ │
│  │Model│    │ Model     │    │ (RL)         │    │ Model   │ │
│  └─────┘    └───────────┘    └──────────────┘    └─────────┘ │
│                  ↑                   ↑                         │
│            偏好数据             SFT Model (ref)                 │
│         (chosen/rejected)       + KL 约束                      │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                       DPO 简化流水线                             │
│                                                                │
│  ┌─────┐    ┌─────────────────────────┐    ┌─────────┐        │
│  │ SFT │ ──→│ DPO Training            │ ──→│ Aligned │        │
│  │Model│    │ (直接用偏好数据优化策略)    │    │ Model   │        │
│  └─────┘    └─────────────────────────┘    └─────────┘        │
│                        ↑                                       │
│                  偏好数据                                        │
│              (chosen/rejected)                                  │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Reward Model（奖励模型）

### 3.1 数据格式

```json
{
  "prompt": "解释量子力学的基本原理",
  "chosen": "量子力学的核心原理包括：1. 波粒二象性...(详细准确的回答)",
  "rejected": "量子力学就是说东西可以同时在两个地方...(不准确的简化)"
}
```

### 3.2 训练原理

Reward Model 的目标：给 chosen 比 rejected 更高的分数。

```
Loss = -log(σ(r(chosen) - r(rejected)))

其中：
  r(x) = RewardModel(prompt, response_x)  →  标量分数
  σ = sigmoid
```

### 3.3 训练代码

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    num_labels=1,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train[:10000]")

training_args = RewardConfig(
    output_dir="./reward_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    bf16=True,
    max_length=2048,
)

trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

---

## 4. RLHF with PPO

### 4.1 PPO 在 LLM 中的应用

```
目标函数：
  maximize E[r(x, y)] - β · KL(π_θ || π_ref)

其中：
  π_θ:    当前策略（正在训练的模型）
  π_ref:  参考策略（SFT 模型，冻结）
  r(x,y): Reward Model 给的分数
  β:      KL 惩罚系数（防止偏离太远）
  KL:     两个策略分布的 KL 散度
```

### 4.2 PPO 训练流程

```
For each batch:
  1. 生成：π_θ 对 prompt 生成回复
  2. 打分：Reward Model 对回复评分
  3. 计算优势：A(t) = r(t) - V(t)  (用 Value Model 估计 baseline)
  4. 更新策略：用 PPO clip 目标更新 π_θ
  5. 更新 Value Model
```

### 4.3 TRL PPOTrainer

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

# 模型需要额外的 value head
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "path/to/sft_model"
)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "path/to/sft_model"
)
tokenizer = AutoTokenizer.from_pretrained("path/to/sft_model")

ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=64,
    mini_batch_size=8,
    gradient_accumulation_steps=8,
    ppo_epochs=4,
    kl_penalty="kl",
    init_kl_coef=0.2,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# 训练循环
for batch in dataloader:
    query_tensors = batch["input_ids"]

    # 1. 生成回复
    response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=256)

    # 2. 用 Reward Model 打分
    rewards = reward_model.score(query_tensors, response_tensors)

    # 3. PPO 更新
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

### 4.4 为什么 PPO 很难？

| 问题 | 原因 |
|---|---|
| 不稳定 | 4 个模型同时交互（policy, ref, reward, value） |
| 内存大 | 需要同时加载 4 个模型 |
| Reward Hacking | 模型学会"欺骗" reward model 而非真正变好 |
| 超参敏感 | KL coef, clip range, lr 都很敏感 |
| 慢 | 每步需要 generate + score + update |

---

## 5. DPO：Direct Preference Optimization

### 5.1 核心思想

DPO 的关键 insight：**Reward Model 可以被隐式表示为策略的函数**。

```
经典 RLHF: 先训 Reward Model → 再用 RL 优化策略
DPO:       直接用偏好数据优化策略（跳过 RM + RL）
```

### 5.2 数学推导

从 RLHF 目标出发：

```
RLHF 最优策略：
  π*(y|x) = π_ref(y|x) · exp(r(x,y) / β) / Z(x)

反解 reward：
  r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)

代入 Bradley-Terry 偏好模型：
  P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
                    = σ(β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x)))

DPO Loss：
  L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

### 5.3 直觉理解

```
DPO 做的事：
  - 让 chosen 回复的概率上升
  - 让 rejected 回复的概率下降
  - 同时不要偏离参考模型太远（隐式 KL 约束）
```

### 5.4 DPO 数据格式

```json
{
  "prompt": "如何提高代码质量？",
  "chosen": "提高代码质量可以从以下几个方面入手：\n1. 代码审查...\n2. 单元测试...\n3. 静态分析...",
  "rejected": "写好代码就行了。"
}
```

或使用 messages 格式：

```json
{
  "chosen": [
    {"role": "user", "content": "如何提高代码质量？"},
    {"role": "assistant", "content": "提高代码质量可以从以下几个方面入手..."}
  ],
  "rejected": [
    {"role": "user", "content": "如何提高代码质量？"},
    {"role": "assistant", "content": "写好代码就行了。"}
  ]
}
```

### 5.5 DPO vs PPO

| 维度 | PPO | DPO |
|---|---|---|
| 需要 Reward Model | ✓ | ✗（隐式） |
| 需要 Value Model | ✓ | ✗ |
| 需要在线生成 | ✓ | ✗（离线） |
| 训练稳定性 | 低 | **高** |
| 显存需求 | 4 个模型 | 2 个模型（policy + ref） |
| 实现复杂度 | 高 | **低**（类似 SFT） |
| 理论最优 | ✓ | 接近 |
| 适合场景 | 追求极致效果 | **大部分场景首选** |

---

## 6. GRPO：Group Relative Policy Optimization

### 6.1 DeepSeek 的方案

GRPO 是 DeepSeek 提出的对齐方法，**不需要 Critic/Value Model**。

核心思想：

```
传统 PPO：用 Value Model 估计 baseline → A(t) = r(t) - V(t)
GRPO：    对同一 prompt 采样 G 个回复 → 用组内相对排名作为 advantage
```

### 6.2 算法流程

```
For each prompt x:
  1. 从 π_θ 采样 G 个回复: {y_1, y_2, ..., y_G}
  2. 用 Reward Model 打分: {r_1, r_2, ..., r_G}
  3. 组内标准化得到 advantage:
     A_i = (r_i - mean(r)) / std(r)
  4. 用 PPO-style clipped objective 更新策略（但不需要 Value Model）

目标函数：
  L_GRPO = E[min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)] - β · KL(π_θ || π_ref)
```

### 6.3 GRPO 的优势

| 维度 | PPO | GRPO |
|---|---|---|
| Value Model | 需要（大模型） | **不需要** |
| 显存 | 4 模型 | 3 模型（policy + ref + reward） |
| Baseline 质量 | 取决于 V 训练质量 | 组内统计，稳定 |
| 样本效率 | 每 prompt 1 回复 | 每 prompt G 回复 |

### 6.4 TRL GRPOTrainer

```python
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/sft_model")
tokenizer = AutoTokenizer.from_pretrained("path/to/sft_model")

# 定义 reward function（可以是规则、模型、或 API）
def reward_fn(completions: list[str], prompts: list[str]) -> list[float]:
    """计算每个回复的 reward"""
    rewards = []
    for completion in completions:
        score = len(completion.split()) / 100  # 示例：长度奖励
        rewards.append(score)
    return rewards

grpo_config = GRPOConfig(
    output_dir="./grpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    num_generations=8,         # 每个 prompt 采样 G 个回复
    max_completion_length=256,
    learning_rate=5e-6,
    beta=0.1,                  # KL 系数
    bf16=True,
)

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    tokenizer=tokenizer,
    train_dataset=dataset,
    reward_funcs=reward_fn,
)

trainer.train()
```

### 6.5 GRPO + 可验证奖励（RLVR）

DeepSeek-R1 的核心：**数学/代码题有确定答案 → 奖励可自动验证**。

```python
import re

def math_reward_fn(completions: list[str], answers: list[str]) -> list[float]:
    """数学题：答案正确 +1，错误 -1"""
    rewards = []
    for completion, answer in zip(completions, answers):
        # 提取模型输出中的最终答案
        match = re.search(r"\\boxed\{(.+?)\}", completion)
        if match and match.group(1).strip() == answer.strip():
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards
```

---

## 7. 其他对齐方法

### 7.1 KTO（Kahneman-Tversky Optimization）

不需要 paired 偏好数据，只需要"好/坏"标签：

```json
{"prompt": "...", "completion": "...", "label": true}
{"prompt": "...", "completion": "...", "label": false}
```

```python
from trl import KTOTrainer, KTOConfig

config = KTOConfig(
    output_dir="./kto_output",
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0,
)
```

### 7.2 IPO（Identity Preference Optimization）

DPO 的改进——避免 DPO 在概率趋近 0/1 时的数值问题：

```
L_IPO = E[(log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l)) - 1/(2β))²]
```

### 7.3 ORPO（Odds Ratio Preference Optimization）

不需要参考模型，直接用 odds ratio：

```python
from trl import ORPOTrainer, ORPOConfig

config = ORPOConfig(
    output_dir="./orpo_output",
    beta=0.1,  # odds ratio weight
    num_train_epochs=1,
)
```

### 7.4 SimPO（Simple Preference Optimization）

用序列平均 log-probability 替代 DPO 中的显式 ref model：

```
r(x,y) = (1/|y|) · log π_θ(y|x) - γ

L_SimPO = -log σ(β · (r(x, y_w) - r(x, y_l)))
```

不需要 reference model → 省一半显存。

### 7.5 方法对比总结

| 方法 | 需要 Ref Model | 需要 Reward Model | 数据格式 | 复杂度 |
|---|---|---|---|---|
| PPO | ✓ | ✓ | prompt only | 最高 |
| DPO | ✓ | ✗ | chosen/rejected | 中 |
| GRPO | ✓ | ✓ | prompt only | 中高 |
| KTO | ✓ | ✗ | 好/坏标签 | 低 |
| IPO | ✓ | ✗ | chosen/rejected | 中 |
| ORPO | ✗ | ✗ | chosen/rejected | **最低** |
| SimPO | ✗ | ✗ | chosen/rejected | 低 |

---

## 8. 偏好数据构建

### 8.1 人工标注

```
流程：
  1. 模型生成多个回复
  2. 标注员选择更好的那个
  3. 得到 (prompt, chosen, rejected) 三元组

成本：~$1-5 per comparison
质量：最高
```

### 8.2 AI 反馈（RLAIF）

```python
def generate_preference_data(prompt: str, model_responses: list[str]) -> dict:
    """用 GPT-4 判断哪个回复更好"""
    judge_prompt = f"""请比较以下两个回复，选择更好的一个。

问题：{prompt}

回复A：{model_responses[0]}
回复B：{model_responses[1]}

请只输出 "A" 或 "B"。"""

    # 调用 GPT-4 ...
    winner = call_gpt4(judge_prompt)

    if winner == "A":
        return {"prompt": prompt, "chosen": model_responses[0], "rejected": model_responses[1]}
    else:
        return {"prompt": prompt, "chosen": model_responses[1], "rejected": model_responses[0]}
```

### 8.3 自动构建策略

| 策略 | 方法 | 质量 |
|---|---|---|
| Best-of-N | 生成 N 个，用 RM 选最好和最差 | 高 |
| On-policy | 用当前模型生成 chosen/rejected | 中高 |
| Rejection Sampling | 生成直到得到好/坏配对 | 高 |
| 规则过滤 | 按长度/格式/关键词分好坏 | 中 |

---

## 9. 评估

### 9.1 自动评估

| Benchmark | 评估内容 | 方法 |
|---|---|---|
| MT-Bench | 多轮对话质量 | GPT-4 评分 (1-10) |
| AlpacaEval | 单轮指令跟随 | GPT-4 对比 win rate |
| Arena (Chatbot Arena) | 综合对话能力 | 人类投票 ELO |
| HumanEval / MBPP | 代码能力 | 执行通过率 |
| GSM8K / MATH | 数学推理 | 答案准确率 |

### 9.2 Reward Model 评估

```python
# 在 held-out 偏好数据上计算准确率
def evaluate_reward_model(rm, eval_dataset):
    correct = 0
    total = 0
    for sample in eval_dataset:
        r_chosen = rm.score(sample["prompt"], sample["chosen"])
        r_rejected = rm.score(sample["prompt"], sample["rejected"])
        if r_chosen > r_rejected:
            correct += 1
        total += 1
    return correct / total

# 好的 RM 准确率应该 > 70%
```

---

## 10. 完整 DPO 训练示例

```python
"""完整的 DPO 训练脚本"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

# ============ 配置 ============
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./dpo_output"

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
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# ============ 2. LoRA 配置 ============
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ============ 3. 加载偏好数据 ============
dataset = load_dataset(
    "HuggingFaceH4/ultrafeedback_binarized",
    split="train_prefs[:5000]",
)

# 数据格式：包含 "chosen" 和 "rejected" 字段（messages 格式）
# TRL DPOTrainer 会自动处理

# ============ 4. DPO 训练配置 ============
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    beta=0.1,                  # DPO 温度参数（越大 → 越保守）
    loss_type="sigmoid",       # "sigmoid" (原始 DPO) 或 "ipo"
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=10,
    save_strategy="epoch",
    max_length=2048,
    max_prompt_length=1024,
    report_to="wandb",
)

# ============ 5. 训练 ============
trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

trainer.train()

# ============ 6. 保存 ============
trainer.save_model(OUTPUT_DIR)
print(f"DPO model saved to {OUTPUT_DIR}")
```

---

## 11. 前沿方向

### 11.1 Constitutional AI（宪法 AI）

```
流程：
  1. 定义一组原则（"宪法"）
  2. 让模型自己生成回复
  3. 让模型根据原则自我批评和修改
  4. 用修改后的回复做 RLHF

优势：减少人工标注需求
```

### 11.2 RLVR（RL from Verifiable Rewards）

对于有确定答案的任务（数学、代码），用自动验证替代 RM：

```python
def verifiable_reward(completion: str, test_cases: list) -> float:
    """代码题：跑测试用例"""
    try:
        exec(completion)
        passed = sum(1 for tc in test_cases if run_test(completion, tc))
        return passed / len(test_cases)
    except Exception:
        return 0.0
```

DeepSeek-R1 就是用 RLVR（数学/代码验证） + GRPO 训练的。

### 11.3 Process Reward Model (PRM)

传统 RM 给最终回复一个总分（Outcome Reward Model, ORM）。
PRM 对推理的**每一步**打分：

```
问题：5 + 3 × 2 = ?

Step 1: 先算乘法 3 × 2 = 6    ← PRM: ✓ (+1)
Step 2: 再算加法 5 + 6 = 11   ← PRM: ✓ (+1)
答案：11                       ← ORM: ✓ (+1)
```

PRM 能更精确地定位错误步骤，训练出更好的推理模型。

### 11.4 Online DPO / Iterative DPO

```
标准 DPO：用固定的离线数据训练一轮
Online DPO：每轮用当前模型生成新的 chosen/rejected → 迭代训练

优势：数据分布与当前策略匹配（on-policy），效果更好
```

---

## 12. 常见坑

### 12.1 DPO 的 beta 选择

```python
# β 太大（如 0.5）→ 太保守，几乎不学习
# β 太小（如 0.01）→ 偏离 ref model 太远，输出质量崩塌

# ✓ 推荐范围
beta = 0.1  # 大部分场景的默认值
# 如果 chosen/rejected 差距大 → beta 可以大一些
# 如果差距小 → beta 小一些
```

### 12.2 偏好数据质量

```python
# ❌ chosen 和 rejected 质量差距太小 → 模型学不到有意义的信号
# ❌ chosen 和 rejected 完全不相关 → 模型学到错误的偏好
# ✓ 好的偏好数据：同一 prompt，chosen 明显优于 rejected
```

### 12.3 Reference Model 不匹配

```python
# ❌ DPO 的 ref_model 和 train_model 来自不同 checkpoint
# ✓ ref_model 应该就是 DPO 训练的起点模型（通常是 SFT 后的模型）

# TRL DPOTrainer 默认用训练模型本身的初始状态作为 ref
# 如果用 LoRA，ref 就是冻结的 base model（自动处理）
```

### 12.4 Reward Hacking

```python
# 症状：Reward Model 分数上升，但实际输出质量下降
# 原因：模型学会了"欺骗" RM 的模式（如特定格式、关键词）

# 缓解：
# 1. 加大 KL 惩罚（beta）
# 2. 定期更新 Reward Model
# 3. 用多个 RM 集成
# 4. 加入人工评估检查点
```

### 12.5 DPO 过拟合

```python
# DPO 比 SFT 更容易过拟合（因为数据通常更少）

# 缓解：
# 1. 只训 1 epoch
# 2. 用 LoRA 限制参数量
# 3. 增大 beta（更保守）
# 4. 监控 train/eval reward margin
```

### 12.6 多轮对话的 DPO

```python
# ❌ 只在最后一轮做 DPO → 忽略了中间轮的偏好
# ✓ 每轮都可以有 chosen/rejected，或者只在关键轮次做

# TRL 支持多轮 DPO，数据用 messages 格式即可
```

---

## 小结

| 概念 | 记忆要点 |
|---|---|
| RLHF (PPO) | SFT → RM → RL，经典但复杂 |
| DPO | 跳过 RM + RL，直接用偏好数据优化 |
| GRPO | 不要 Value Model，组内相对排名 |
| KTO | 只需好/坏标签，不需 paired data |
| ORPO/SimPO | 不需 ref model，最轻量 |
| Beta (β) | 控制与 ref model 的偏离程度 |
| Reward Hacking | 模型学会欺骗 RM，需要 KL 约束 |
| RLVR | 有确定答案时用自动验证当 reward |
| 数据 | chosen/rejected 差距要明显且合理 |

**一句话**：**2024+ 实战首选 DPO（简单稳定），追求极致用 GRPO/Online DPO，有确定答案用 RLVR。**
