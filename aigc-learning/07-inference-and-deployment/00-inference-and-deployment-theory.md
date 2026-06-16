# 00 · 推理与部署理论总览

> 推理部署不是把 `model.generate()` 包一层 HTTP。
> 它是在质量、延迟、吞吐、显存和成本之间做系统性权衡。
> 工程优化背后真正起作用的是性能模型、调度模型和容量模型。

---

## 1. 推理系统的目标函数

训练阶段优化的是 loss，推理阶段优化的是线上效用：

```
maximize   Quality - Cost - LatencyPenalty - ErrorPenalty
subject to GPU memory, SLA, concurrency, safety constraints
```

在真实服务里，没有单一指标能代表好坏。常见指标分三层：

| 层级 | 指标 | 含义 |
|---|---|---|
| 用户体验 | TTFT / E2E latency / 流式稳定性 | 用户多久看到第一个 token，多久完成 |
| 系统效率 | throughput / QPS / GPU utilization | 单位 GPU 时间能服务多少请求 |
| 成本可靠性 | cost per 1K tokens / error rate / queue length | 是否可持续、是否稳定 |

核心结论：**低延迟和高吞吐经常冲突**。为了提高吞吐，系统会做 batching；为了降低单请求延迟，系统又要减少排队和批处理等待。因此推理部署的关键不是“开某个优化开关”，而是根据业务目标选择合理的 Pareto 点。

### 1.1 推理优化为什么是大模型落地的关键思想？

训练阶段的突破让模型“有能力”，推理系统决定这种能力能否以可接受成本服务真实用户。
很多推理优化不改变模型权重，也不改变输出数学分布，却能让同一张 GPU 服务更多请求或支持更长上下文。

关键转向是：**从单次 forward 的代码视角，转向在线系统的资源调度视角**。

| 思想 | 改变了什么 | 为什么重要 |
|---|---|---|
| Prefill / Decode 分离 | 把一次生成拆成计算密集阶段和带宽密集阶段 | 不同阶段瓶颈不同，优化手段也不同 |
| KV Cache | 历史 token 的 K/V 只算一次 | 让自回归生成从重复算前缀变成增量生成 |
| Continuous Batching | 每个 decode step 动态重组 batch | 提高吞吐，同时减少短请求被长请求拖住 |
| PagedAttention | 用 block/page 管理 KV cache | 降低碎片，提高长上下文和高并发容量 |
| Speculative Decoding | 小模型草拟，大模型验证 | 标准接受-拒绝算法在特定采样设置下可保持目标分布，同时降低 decode 延迟 |

这些技术的共同点是：模型公式基本没变，改变的是计算顺序、缓存方式、内存布局和调度策略。
这也是为什么推理工程必须同时懂 Transformer、显存模型、队列调度和 GPU 性能。

---

## 2. LLM 推理的两阶段性能模型

Decoder-only LLM 自回归生成分为 prefill 和 decode。

### 2.1 Prefill：计算密集

Prefill 一次性处理整个 prompt：

```
input:  B x S_prompt
output: first token logits + KV Cache
```

这一步的矩阵乘法规模大，GPU tensor core 利用率较高。瓶颈通常是 FLOPS、attention 的序列长度，以及 prompt 长度带来的 quadratic attention 成本。

可以粗略理解为：

```
prefill_cost ≈ O(num_layers * S_prompt * hidden^2)
             + O(num_layers * S_prompt^2 * hidden)
```

长 prompt、RAG 大上下文、长对话历史都会主要拉高 prefill latency，也就是 TTFT。

### 2.2 Decode：内存带宽密集

Decode 每次只生成 1 个新 token：

```
input:  B x 1 + historical KV Cache
output: next token + appended KV Cache
```

每生成一个 token 都需要读模型权重和历史 KV。单请求 decode 的矩阵规模很小，GPU 算力常常吃不满，瓶颈变成 HBM 显存带宽。

粗略上限可以这样估算：

```
max_tokens_per_sec ≈ memory_bandwidth_bytes_per_sec / model_weight_bytes
```

例如 7B FP16 权重约 14GB，如果 GPU 显存带宽是 2TB/s，那么单请求 decode 的理想上限约：

```
2000GB/s / 14GB ≈ 143 tokens/s
```

实际值会更低，因为还有 KV Cache、kernel launch、采样、通信和调度开销。

### 2.3 为什么 batching 对 decode 特别重要？

单请求 decode 读取一次权重只服务一个 token。batch decode 读取同一份权重，可以同时服务多个请求的下一个 token：

```
单请求:  read W -> 1 token
batch:   read W -> B tokens
```

这就是连续批处理能显著提升吞吐的根本原因：它把“读权重”的成本摊到更多 token 上。

---

## 3. KV Cache 的显存模型

Attention 需要历史 token 的 Key 和 Value。KV Cache 保存每层的 K/V，避免每一步重新计算历史 token。

### 3.1 基本公式

```
KV Cache bytes = 2 * L * H_kv * D_head * S * B * bytes_per_value
```

含义：

| 符号 | 含义 |
|---|---|
| `2` | Key 和 Value |
| `L` | Transformer 层数 |
| `H_kv` | KV heads 数量 |
| `D_head` | 每个 head 的维度 |
| `S` | 已缓存序列长度 |
| `B` | batch / 并发序列数 |
| `bytes_per_value` | FP16/BF16 为 2，FP8/INT8 为 1 |

示例代码：[`examples/kv_cache_and_batching.py`](./examples/kv_cache_and_batching.py)

```bash
cd aigc-learning/07-inference-and-deployment/examples
conda run -n aigc python kv_cache_and_batching.py --model llama2-7b --batch-size 4
```

### 3.2 GQA/MQA 为什么省显存？

传统 MHA 每个 query head 都有自己的 K/V head。GQA/MQA 让多个 query head 共享更少的 KV heads：

| 架构 | KV heads | KV Cache |
|---|---:|---:|
| MHA | 等于 attention heads | 最大 |
| GQA | 小于 attention heads | 按比例减少 |
| MQA | 1 | 最小 |

如果一个模型从 32 个 KV heads 改为 8 个 KV heads，KV Cache 显存理论上降低 4 倍。长上下文和高并发场景下，这个收益非常直接。

### 3.3 显存瓶颈来自哪里？

LLM 推理显存通常由四部分组成：

```
Total memory = model weights + KV Cache + activation/temp buffers + runtime overhead
```

对短上下文、低并发，权重占主导。对长上下文、高并发，KV Cache 会变成主导。

容量规划时不能只看“模型能不能放进显存”，还要问：

- 最大上下文是多少？
- 同时在线请求数是多少？
- 是否启用 GQA、KV quantization、prefix caching？
- 推理引擎为 KV Cache 预留了多少 block？

---

## 4. 批处理与队列调度理论

### 4.1 静态批处理的问题

静态 batch 把一组请求绑在一起，直到最长请求完成才释放 batch。若输出长度差异大，短请求会被长请求拖住。

浪费可以表示为：

```
waste = sum(max_len_in_batch - len_i)
```

输出长度越不均匀，浪费越大。

### 4.2 连续批处理的思想

连续批处理在每个 decode step 都重新组织 active batch：

```
step t:   [r1, r2, r3, r4]
step t+1: [r1, r2, r5, r4]  # r3 完成，r5 插入
```

它的优势是：

- 短请求完成后立即释放 slot。
- 新请求无需等待整个 batch 结束。
- GPU 每一步尽量保持满 batch。

这本质上是 token-level scheduling，而不是 request-level scheduling。

### 4.3 Batching 的代价

Batching 不是免费午餐。排队窗口越大，越容易组成大 batch，吞吐越高；但请求等待时间也越长，TTFT 上升。

可以把 TTFT 分解为：

```
TTFT = queue_wait + prefill_time + first_decode_step
```

优化吞吐时，要监控 `queue_wait` 是否吃掉了用户体验。

---

## 5. PagedAttention 与内存碎片

传统 KV Cache 预分配常见问题是“为最坏情况分配，为平均情况使用”。如果每个请求都按 `max_model_len` 预留连续空间，而真实输出远短于最大长度，显存利用率会很低。

PagedAttention 借鉴虚拟内存：

```
logical sequence blocks -> physical KV blocks
```

优势：

- 按需分配 KV block。
- 减少连续大块显存需求。
- 支持 block 级共享，例如 prefix caching。
- 降低碎片，提高并发上限。

PagedAttention 不改变模型数学结果，它改变的是 KV Cache 的物理布局和调度方式。

---

## 6. 投机解码的正确性直觉

投机解码用 draft model 先生成多个候选 token，再用 target model 一次验证。

关键点：target model 不是简单接受草稿，而是按概率校正接受/拒绝。因此在特定采样设置下，投机解码可以保持和 target model 原始分布一致。

适合场景：

| 条件 | 原因 |
|---|---|
| draft model 足够快 | 草稿成本必须远小于 target decode |
| draft 和 target 分布接近 | 接受率高才有加速 |
| decode 阶段占主导 | prefill 很长时收益会被稀释 |
| batch 不过大 | 大 batch 已经提高并行度，投机收益可能下降 |

投机解码的工程风险是系统复杂度增加：需要多模型加载、KV 管理、接受率监控和更复杂的调度。

---

## 7. 扩散模型推理的理论模型

扩散模型生成图像不是一次前向，而是多步去噪：

```
x_T -> x_{T-1} -> ... -> x_0
```

每一步都要调用 UNet 或 DiT，因此延迟近似为：

```
latency ≈ num_steps * model_forward_time + VAE_decode_time + overhead
```

示例代码：[`examples/diffusion_acceleration_sim.py`](./examples/diffusion_acceleration_sim.py)

```bash
cd aigc-learning/07-inference-and-deployment/examples
conda run -n aigc python diffusion_acceleration_sim.py --latent-size 32
```

### 7.1 Scheduler 在优化什么？

Scheduler 决定从高噪声到低噪声的路径。更好的 scheduler 可以用更少的步数达到类似质量。

从连续视角看，扩散采样可理解为求解 ODE/SDE。不同 scheduler 对应不同数值求解器和步长策略：

| 类型 | 直觉 |
|---|---|
| DDPM | 原始随机反向过程，步数多 |
| DDIM | 确定性跳步，步数少很多 |
| DPM++ | 高阶求解器，更少步数保质量 |
| Karras schedule | 把步长分配到更关键的噪声区间 |
| LCM/Turbo/Lightning | 模型被蒸馏到少步数轨道 |

### 7.2 为什么“只减少 steps”不一定有效？

普通模型训练时见到的是多步去噪轨道。如果推理时强行从 50 步改成 4 步，每一步需要完成过大的去噪跨度，误差会累积。

少步数模型通常需要蒸馏，让模型学会在粗步长下直接预测更接近最终图像的方向。

### 7.3 CFG 的代价

Classifier-Free Guidance 常见公式：

```
eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
```

它通常需要同时计算 conditional 和 unconditional 分支，因此会增加计算量。蒸馏模型常把 `guidance_scale` 设得很低甚至为 0，以减少额外成本并匹配训练分布。

---

## 8. 服务化的系统模型

一个推理服务可以分成五层：

```
Client -> API Gateway -> Scheduler/Queue -> Inference Engine -> Model Runtime
```

各层职责不同：

| 层 | 主要职责 |
|---|---|
| Client | 重试、超时、流式消费 |
| API Gateway | 鉴权、限流、参数校验、日志 |
| Scheduler/Queue | batching、优先级、背压 |
| Inference Engine | KV Cache、kernel、并行、采样 |
| Model Runtime | 权重、算子、设备执行 |

示例代码：

- [`examples/openai_compatible_toy_server.py`](./examples/openai_compatible_toy_server.py)
- [`examples/fastapi_gateway.py`](./examples/fastapi_gateway.py)

### 8.1 背压比盲目扩并发重要

当请求进入速度超过系统处理速度，队列会持续增长。此时继续接收请求只会让所有请求变慢，最终超时。

背压策略包括：

- 限制最大队列长度。
- 对超长 prompt 或超大 `max_tokens` 提前拒绝。
- 对低优先级请求降级或排队。
- 返回 429/503，让客户端退避重试。

### 8.2 超时要分层设置

LLM 请求可能很长，超时不能只设一个总值。建议拆成：

| 超时 | 含义 |
|---|---|
| queue timeout | 排队超过阈值则拒绝 |
| TTFT timeout | 首 token 太慢则中断 |
| idle timeout | 流式连接长时间无 token 则中断 |
| total timeout | 总生成时间上限 |

---

## 9. OpenAI 兼容接口的理论价值

OpenAI 兼容格式的价值不在“像 OpenAI”，而在于统一了客户端生态：

```
application code -> OpenAI SDK shape -> any compatible backend
```

这样模型后端可以在 OpenAI、vLLM、SGLang、llama.cpp、内部服务之间切换，而应用层代码基本不变。

需要注意：兼容通常分层级：

| 层级 | 示例 |
|---|---|
| 基础兼容 | `/v1/chat/completions`、messages、stream |
| 参数兼容 | temperature、top_p、max_tokens、stop |
| 高级兼容 | tools、structured output、logprobs、vision |

不要默认所有后端都完整支持高级能力。生产中应写兼容性测试。

---

## 10. Demo 与用户感知延迟

Demo 的目标不是展示所有能力，而是让用户快速形成正确预期。

流式输出改善的是感知延迟：

```
perceived latency ≈ TTFT
actual completion time ≈ E2E latency
```

如果 TTFT 足够低，即使完整回答需要几秒，用户也更容易接受。

示例代码：[`examples/demo_apps.py`](./examples/demo_apps.py)

```bash
cd aigc-learning/07-inference-and-deployment/examples
conda run -n aigc python demo_apps.py --mode self-test
```

---

## 11. 容量规划的最小模型

上线前至少做三类估算。

### 11.1 显存估算

```
required_memory = weights + max_kv_cache + runtime_buffer + safety_margin
```

安全余量建议不要低于 10%-15%。如果引擎需要 CUDA Graph 或额外 workspace，余量还要更高。

### 11.2 吞吐估算

```
required_tokens_per_sec = QPS * average_output_tokens
```

如果有长 prompt，还要单独估 prefill 吞吐：

```
required_prefill_tokens_per_sec = QPS * average_prompt_tokens
```

### 11.3 SLA 估算

```
P99_latency = P99_queue_wait + P99_prefill + P99_decode
```

平均值对线上 SLA 参考意义有限。排队系统一旦接近饱和，P99 会远早于平均值恶化。

---

## 12. 工程决策清单

做推理部署选型时，按以下顺序回答问题：

1. 模型大小、上下文长度、并发目标分别是多少？
2. 主要瓶颈是 TTFT、TPOT、吞吐、显存还是成本？
3. 请求长度分布是否长尾？是否需要连续批处理？
4. 是否需要结构化输出、工具调用、多模态？
5. 是否必须流式？客户端和网关是否正确处理 SSE？
6. 是否允许量化？质量评估集是什么？
7. 是否有冷启动、模型热更新、灰度发布需求？
8. 监控是否覆盖 queue、TTFT、TPOT、KV Cache、GPU utilization、error rate？
9. 失败时是降级、重试、排队还是拒绝？
10. Demo、API、生产服务是否共用同一套协议契约？

---

## 13. 小结

| 理论点 | 工程含义 |
|---|---|
| Prefill 计算密集 | 长 prompt 会拉高 TTFT，需要 prompt/cache 优化 |
| Decode 内存密集 | batching、量化、speculative decoding 对吞吐关键 |
| KV Cache 线性增长 | 长上下文和高并发首先打爆显存 |
| 连续批处理 | 用 token-level scheduling 提升 GPU 利用率 |
| PagedAttention | 通过 block 管理减少 KV 碎片和预留浪费 |
| 扩散采样多步迭代 | 速度主要由 steps 和单步模型成本决定 |
| 服务队列会放大尾延迟 | 必须做背压、限流和 P99 监控 |
| OpenAI 兼容是协议层抽象 | 让应用和推理后端解耦 |

**一句话**：推理部署的核心是用性能模型指导工程取舍，而不是堆工具名。
