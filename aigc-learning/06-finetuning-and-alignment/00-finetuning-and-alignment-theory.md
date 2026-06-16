# 00 · 微调与对齐理论总览

> 微调不是“继续训练一下”，对齐也不是“让模型更礼貌一点”。
> 它们分别改变模型的任务条件分布和偏好分布，是把基座模型变成可用助手的两类核心技术。

---

## 1. 从预训练到可用助手

大语言模型的训练通常分为三段：

```
Pretraining ──→ Supervised Fine-Tuning ──→ Preference Alignment
  学语言和知识       学指令到回复的映射          学什么回复更好
```

### 1.1 预训练学到什么？

预训练目标通常是 next-token prediction：

```
max log p_theta(x_t | x_<t)
```

模型在大规模语料上学习“给定前文，下一个 token 应该是什么”。这让模型获得语言结构、事实知识、代码模式和推理雏形，但它默认做的是续写，不一定会遵循用户指令。

### 1.2 SFT 学到什么？

SFT 用 `(instruction, response)` 或多轮 `messages` 数据训练：

```
max log p_theta(y | x)
```

其中 `x` 是用户指令或对话上下文，`y` 是期望回复。SFT 的核心变化是把模型从“续写网页文本”推向“按特定模板回答用户”。

### 1.3 对齐学到什么？

对齐使用偏好、奖励或规则信号，使模型倾向于输出更有帮助、更安全、更诚实的回复。它不是单纯提高准确率，而是在多个合理回复之间学习排序：

```
chosen response > rejected response
```

### 1.4 三阶段分别改变了什么分布？

很多人把预训练、SFT、RLHF/DPO 都理解成“继续训练”，这样会漏掉关键差异。
它们都在更新参数，但优化的问题不同。

| 阶段 | 主要优化对象 | 直觉 | 常见失败模式 |
|---|---|---|---|
| Pretraining | `p(text)` 或 `p(next_token | prefix)` | 学世界文本分布、语言规律和知识关联 | 会续写但不一定听指令；可能学到噪声和偏见 |
| SFT | `p(answer | instruction, context)` | 学用户输入到助手回复的条件映射 | 数据模板污染、过拟合风格、灾难性遗忘 |
| Preference Alignment | `chosen > rejected` 的排序偏好 | 在多个可行回复中偏向更符合人类偏好的回复 | 过度拒答、奖励投机、能力退化、回复模板化 |

一个有用的心智模型：

```text
预训练：扩大能力边界
SFT：指定交互方式
偏好对齐：调整回答排序
```

因此，领域微调前要先判断问题属于哪一类：

- 模型缺少领域知识：优先检查 RAG、继续预训练或领域数据混合，而不是只做偏好对齐。
- 模型不会按格式回答：优先做高质量 SFT 和模板控制。
- 模型能回答但偏啰嗦、不安全或不符合偏好：再考虑 DPO/RLHF/规则偏好优化。

对齐不是给模型“灌知识”的主要方式。偏好数据通常只告诉模型两个回答哪个更好，并不可靠地提供新的事实知识。

#### 深度解读：SFT、RLHF、DPO 最容易混淆的边界

SFT、RLHF、DPO 都会改变模型输出，但它们不是同一种“能力增强”。

SFT 主要解决“模型如何回应用户”的问题。它通过高质量指令和回复样本，让模型学会对话模板、回答格式、领域任务流程和工具调用格式。
如果模型不会按 JSON 输出、不会遵循多轮对话格式、不会用某种语气写报告，SFT 往往是直接有效的。
但如果问题是“模型不知道某个私有事实”，单靠少量 SFT 很容易变成记忆样本，而不是可靠知识系统。

RLHF 和 DPO 主要解决“多个可行回答里哪个更好”的问题。
它们会提高 chosen 回复相对 rejected 回复的概率，但这不等于模型学会了新的事实数据库，也不等于数学、代码、检索能力自然增强。
偏好优化过强时，模型可能更会迎合评审偏好，却出现过度拒答、套话变多、长答案虚高、客观能力回退等问题。

做领域模型时，可以按下面顺序判断：

| 问题类型 | 优先手段 |
|---|---|
| 缺少最新或私有知识 | RAG、工具查询、继续预训练或领域数据混合 |
| 输出格式和任务流程不对 | SFT、模板约束、assistant-only loss |
| 风格、礼貌、安全边界不合适 | DPO/RLHF/规则偏好数据 |
| 答案需要可验证计算或外部动作 | 工具调用、校验器、工作流编排 |

一句话：SFT 教“怎么答”，对齐调“更偏好哪种答”，RAG/工具解决“依据什么事实和动作来答”。

---

## 2. 微调的优化对象

### 2.1 全量微调

全量微调直接更新模型全部参数：

```
theta' = theta - eta * grad L(theta)
```

优点是表达能力强，缺点是显存、算力和存储成本高。对 7B 以上模型，Adam 优化器状态、梯度和激活会让单卡训练变得困难。

### 2.2 PEFT 的理论动机

PEFT 假设下游任务不需要改变模型的全部自由度。预训练模型已经有大部分能力，微调只需要在参数空间里移动一小段距离。

LoRA 进一步假设权重更新 `Delta W` 近似低秩：

```
W' = W + Delta W
Delta W = B A

W: d x k
A: r x k
B: d x r
r << min(d, k)
```

如果 `d = k = 4096`，全量更新需要 `4096 * 4096 = 16,777,216` 个参数。rank=8 的 LoRA 只需要：

```
8 * 4096 + 4096 * 8 = 65,536
```

这不是魔法，而是用低秩子空间限制更新方向。它降低成本，也降低过拟合风险。

### 2.3 Rank 的含义

`rank` 控制 LoRA 更新矩阵的最大自由度：

| Rank | 表达能力 | 适用场景 |
|---|---|---|
| 4-8 | 低 | 风格、格式、小任务适配 |
| 16-32 | 中 | 常规指令微调 |
| 64+ | 高 | 代码、数学、复杂领域迁移 |

rank 不是越大越好。rank 过大时，显存、训练时间和过拟合风险都会增加。工程上通常先用小 rank 建立基线，再根据验证集和人工评估上调。

---

## 3. 量化的理论基础

### 3.1 量化在做什么？

量化把连续的高精度数映射到少量离散值：

```
q = round(x / scale)
x_hat = q * scale
```

误差来自 `x_hat` 和 `x` 的差距。LLM 量化的目标不是让每个权重都完全精确，而是让整体输出分布尽量接近原模型。

### 3.2 为什么分组量化更准？

如果整层权重共用一个 scale，少数 outlier 会拉大 scale，导致普通权重的有效精度变低。分组量化让每组权重拥有自己的 scale：

```
每 128 个权重一个 scale
```

这样可以在存储额外 scale 的代价下显著降低误差。GPTQ、AWQ、GGUF 的很多 4-bit 方法都依赖分组量化。

### 3.3 PTQ 与 QAT

| 方法 | 核心思路 | LLM 中的常见程度 |
|---|---|---|
| PTQ | 训练后量化，可能需要校准数据 | 主流 |
| QAT | 训练时模拟量化误差 | 成本高，较少用于大 LLM |

QLoRA 的关键点是“量化基座 + 训练 LoRA”。量化权重本身通常不直接反向更新，梯度主要进入 LoRA adapter。

---

## 4. SFT 的数据理论

### 4.1 Chat Template 是训练分布的一部分

同一段对话，用不同模板序列化后会变成不同 token 序列。模型学到的是 token 分布，因此模板错误会直接改变训练目标。

```
messages ──chat template──→ token sequence ──loss──→ parameter update
```

实战原则：

- 使用目标模型 tokenizer 自带的 `apply_chat_template()`。
- 训练和推理使用同一套模板。
- 不要把多个模型家族的模板混在同一个训练集中。

### 4.2 Loss Masking 的意义

SFT 不应该让模型学习生成用户问题本身。常见做法是只在 assistant 回复上计算 loss：

```
system/user tokens      -> label = -100
assistant answer tokens -> label = token_id
```

这样优化目标更接近“给定用户输入生成助手回复”，而不是“复现整段对话文本”。

### 4.3 数据质量为什么比数量重要？

SFT 数据直接定义了模型的行为边界。低质量样本会引入几类问题：

| 问题 | 影响 |
|---|---|
| 错误事实 | 强化幻觉 |
| 冗长模板化回复 | 模型变啰嗦 |
| 指令和回复不匹配 | 降低指令跟随 |
| 领域过窄 | 灾难性遗忘 |
| 重复样本过多 | 过拟合、评估虚高 |

好的 SFT 数据通常是高覆盖、低重复、明确可执行、回复风格一致且事实可靠的数据。

---

## 5. 对齐的概率视角

### 5.1 Reward Model

奖励模型用偏好数据训练一个标量函数：

```
r(prompt, response) -> score
```

训练目标通常是 Bradley-Terry pairwise loss：

```
L = -log sigmoid(r_chosen - r_rejected)
```

它只要求 chosen 分数高于 rejected，不要求 reward 的绝对值有真实物理意义。

### 5.2 RLHF 为什么需要 KL？

如果只最大化 reward，模型可能偏离原始语言模型太远，甚至学会欺骗 Reward Model。RLHF 目标通常加入 KL 约束：

```
maximize E[r(x, y)] - beta * KL(pi_theta || pi_ref)
```

`pi_ref` 通常是 SFT 模型。KL 项的作用是保留原模型的语言能力和行为边界。

### 5.3 DPO 的核心简化

DPO 跳过显式 Reward Model 和 PPO，把偏好数据直接转成分类式目标：

```
L_DPO = -log sigmoid(
  beta * [
    log pi_theta(y_w|x) - log pi_ref(y_w|x)
    - log pi_theta(y_l|x) + log pi_ref(y_l|x)
  ]
)
```

直觉上，DPO 同时做三件事：

- 提高 chosen 回复概率。
- 降低 rejected 回复概率。
- 用 reference model 约束偏离程度。

### 5.4 Beta 的含义

`beta` 控制偏好优化的强度和保守程度：

| Beta | 行为 |
|---|---|
| 太小 | 更新激进，容易偏离参考模型 |
| 适中 | 能学习偏好，同时保留原能力 |
| 太大 | 更新保守，学习信号弱 |

工程上常从 `0.1` 附近开始，结合验证集 win rate、人工评估和生成质量调整。

---

## 6. 过拟合与灾难性遗忘

SFT 和 DPO 都容易过拟合，原因是数据通常比预训练小几个数量级。典型症状：

- 训练 loss 下降，验证 loss 上升。
- 模型过度模仿训练集格式。
- 通用问答、代码、数学能力下降。
- 安全拒答或格式偏好变得僵硬。

缓解策略：

| 策略 | 说明 |
|---|---|
| 控制 epoch | 大数据 1-3 epoch，小数据谨慎多轮 |
| 降低学习率 | LoRA 可比 full FT 高，但不能无限高 |
| 混入通用数据 | 保留原模型通用能力 |
| 使用 PEFT | 限制可训练自由度 |
| 设置 eval split | 监控真实泛化而非训练记忆 |
| 做人工评估 | 发现自动指标看不到的问题 |

---

## 7. 工程决策地图

| 目标 | 推荐起点 |
|---|---|
| 单卡做领域 SFT | QLoRA + TRL SFTTrainer |
| 数据少、任务简单 | LoRA rank 4-8 |
| 通用指令微调 | LoRA rank 16-32，assistant-only loss |
| GPU 推理部署 | 合并 LoRA 后做 AWQ/GPTQ，或直接 vLLM 多 LoRA |
| CPU/边缘部署 | GGUF + llama.cpp |
| 简单偏好对齐 | DPO |
| 有可验证答案 | GRPO/RLVR |
| 追求复杂 RLHF | Reward Model + PPO，但成本和风险更高 |

---

## 8. 本地可运行示例

本模块的 `examples/` 目录提供了可离线运行的小实验：

```bash
cd aigc-learning/06-finetuning-and-alignment/examples
conda run -n aigc python lora_tiny_train.py --epochs 6 --rank 2
conda run -n aigc python quantization_sim.py --bits 4 --group-size 32
conda run -n aigc python sft_data_pipeline.py --max-length 80 --pack
conda run -n aigc python dpo_loss_demo.py --beta 0.1
```

这些脚本不替代真实大模型训练，但能验证核心机制：低秩更新、分组量化误差、assistant-only loss mask、DPO margin 与 GRPO advantage。
