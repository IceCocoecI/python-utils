# 01 · LLM 推理引擎

> LLM 推理不是简单的 `model.generate()`。
> 当你要服务 100 个并发用户时，推理引擎的选择决定了你需要 1 张 GPU 还是 10 张。

---

## 1. LLM 推理基础：两个阶段

LLM 的自回归生成分为两个截然不同的阶段：

```
┌──────────────────────────────────────────────────────────┐
│                  LLM 推理两阶段                           │
│                                                          │
│  [用户 prompt] ──→ Prefill ──→ [第一个 token]            │
│                      │                                    │
│                      ▼                                    │
│                    Decode ──→ token ──→ token ──→ ... EOS │
│                  (逐个生成)                                │
└──────────────────────────────────────────────────────────┘
```

### Prefill（预填充）

- **输入**：整个 prompt（可能上千 token）一次性输入。
- **特点**：**计算密集**（Compute-bound）——大量矩阵乘法可以并行。
- **瓶颈**：GPU 算力（FLOPS）。
- **类比**：读完一整本书，形成理解。

### Decode（解码）

- **输入**：每次只处理 1 个新 token。
- **特点**：**内存密集**（Memory-bound）——每一步都要读取全部模型权重和 KV Cache。
- **瓶颈**：显存带宽（GB/s）。
- **类比**：一个字一个字地写出回答。

```python
# 简化理解：为什么 decode 是内存密集的
# 7B 模型 FP16 权重 ≈ 14 GB
# 每生成一个 token 都要读一遍所有权重
# A100 80GB 的显存带宽 = 2 TB/s
# 理论上限 = 2000 / 14 ≈ 143 tokens/sec（单请求）
# 实际由于 KV Cache 等开销，通常更低
```

> **关键洞察**：decode 阶段 GPU 的算力严重空闲（利用率常常 < 5%），瓶颈在"搬数据"。
> 这就是为什么推理优化的核心是**减少显存访问**和**增加并行度（batching）**。

---

## 2. KV Cache：为什么必须存在

### 问题

Attention 的计算需要所有历史 token 的 Key 和 Value：

```
Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V
```

如果不缓存，生成第 N 个 token 时需要重新计算前 N-1 个 token 的 K、V，复杂度 O(N²)。

### 解法

缓存每一层、每一个 head 的 K、V 向量。生成新 token 时只计算当前 token 的 Q、K、V，然后把新的 K、V 追加到缓存。

### 显存开销公式

```
KV Cache 大小 = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch_size × dtype_bytes
```

**举例：LLaMA-2 7B，FP16，seq_len=4096，batch=1**

```python
num_layers = 32
num_kv_heads = 32  # (非 GQA 模型)
head_dim = 128
seq_len = 4096
batch_size = 1
dtype_bytes = 2  # FP16

kv_cache_bytes = 2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * dtype_bytes
kv_cache_gb = kv_cache_bytes / (1024 ** 3)
print(f"KV Cache: {kv_cache_gb:.2f} GB")  # ≈ 2.0 GB
```

> **现实冲击**：一个 70B 模型，128K 上下文，16 并发 —— KV Cache 可以轻松吃掉 100+ GB 显存。
> 这就是为什么 KV Cache 管理是推理引擎的核心战场。

---

## 3. 批处理策略

### 3.1 静态批处理（Static / Naive Batching）

```
┌──────────────────────────────────────┐
│  Batch（4 个请求）                    │
│                                      │
│  Req 1: [████████████████]  done     │
│  Req 2: [████████████████████] done  │
│  Req 3: [████]              waiting  │  ← 生成完了也要等
│  Req 4: [████████████]      waiting  │  ← 同上
│                                      │
│  全部完成后才能接收新请求              │
└──────────────────────────────────────┘
```

**问题**：短请求要等最长的那个完成，GPU 利用率低。

### 3.2 连续批处理（Continuous Batching）

```
┌──────────────────────────────────────┐
│  迭代级调度                           │
│                                      │
│  Step 1: [Req1, Req2, Req3, Req4]   │
│  Step 2: [Req1, Req2, Req3, Req4]   │
│  Step 3: [Req1, Req2, Req5, Req4]   │  ← Req3 完成，Req5 插入
│  Step 4: [Req1, Req2, Req5, Req6]   │  ← Req4 完成，Req6 插入
│                                      │
│  每一步都可以加入/移除请求             │
└──────────────────────────────────────┘
```

**优势**：
- GPU 利用率大幅提升（2-10x 吞吐提升）
- 短请求不用等长请求
- 新请求可以随时加入

> vLLM / SGLang / TensorRT-LLM 都实现了连续批处理。

---

## 4. PagedAttention（vLLM 的核心创新）

### 问题：KV Cache 的显存碎片化

传统实现为每个请求预分配一整块连续显存来存 KV Cache。但实际生成长度是未知的，导致：
- 预分配太多 → 浪费
- 预分配太少 → 需要重新分配和复制

**浪费通常高达 60-80%。**

### PagedAttention 的解法

借鉴操作系统的虚拟内存/分页机制：

```
┌──────────────────────────────────────────────┐
│  传统方式：连续分配                            │
│  Req1: [████████████░░░░░░░░]  预留但未使用   │
│  Req2: [████░░░░░░░░░░░░░░░░]  大量浪费      │
│                                              │
│  PagedAttention：分页管理                     │
│  物理 Block 池: [B0][B1][B2][B3][B4][B5]...  │
│  Req1 页表:  → B0 → B3 → B5（按需分配）      │
│  Req2 页表:  → B1 → B2（按需分配）            │
│                                              │
│  ✓ 近乎 0 碎片   ✓ 按需分配   ✓ 支持共享     │
└──────────────────────────────────────────────┘
```

**效果**：KV Cache 利用率从 ~20-40% 提升到 >95%，直接换来更大的并发和更高的吞吐。

---

## 5. vLLM：最流行的 LLM 推理引擎

### 5.1 安装

```bash
pip install vllm
```

> 需要 CUDA 12.1+ 和支持的 GPU（Ampere 及以上推荐）。

### 5.2 离线推理（Offline Inference）

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    repetition_penalty=1.1,
)

prompts = [
    "解释什么是 KV Cache",
    "用 Python 写一个快速排序",
    "翻译成英文：大模型推理优化是一个重要的研究方向",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Output: {generated!r}\n")
```

### 5.3 OpenAI 兼容服务器

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "什么是 PagedAttention？"},
    ],
    temperature=0.7,
    max_tokens=512,
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 5.4 关键参数

| 参数 | 说明 | 推荐值 |
|---|---|---|
| `--tensor-parallel-size` | 张量并行 GPU 数 | 模型放不下一张卡时增加 |
| `--max-model-len` | 最大上下文长度 | 按需设置，越大越耗显存 |
| `--gpu-memory-utilization` | GPU 显存使用比例 | 0.85-0.95 |
| `--quantization` | 量化方法 | `awq`, `gptq`, `fp8` |
| `--enforce-eager` | 禁用 CUDA Graph | 调试时使用 |
| `--enable-prefix-caching` | 前缀缓存 | 多轮对话场景开启 |

---

## 6. SGLang：结构化生成与 RadixAttention

### 6.1 核心创新

- **RadixAttention**：用 Radix Tree 管理 KV Cache，自动复用共享前缀（比 vLLM 的 prefix caching 更细粒度）。
- **结构化生成**：原生支持 JSON Schema / 正则约束输出格式。
- **前端语言**：提供 Python DSL 来编写复杂的 LLM 程序（多轮、分支、并行）。

### 6.2 安装与启动

```bash
pip install "sglang[all]"

python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 30000
```

### 6.3 OpenAI 兼容调用

SGLang 同样提供 OpenAI 兼容的 API：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "什么是 RadixAttention？"}],
    temperature=0.7,
    max_tokens=256,
)
print(response.choices[0].message.content)
```

### 6.4 结构化生成（JSON Mode）

```python
import json

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "给出三个中国城市的名称和人口（JSON 格式）"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "cities",
            "schema": {
                "type": "object",
                "properties": {
                    "cities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "population": {"type": "integer"},
                            },
                            "required": ["name", "population"],
                        },
                    }
                },
                "required": ["cities"],
            },
        },
    },
)

data = json.loads(response.choices[0].message.content)
print(json.dumps(data, ensure_ascii=False, indent=2))
```

### 6.5 前端语言（SGLang DSL）

```python
import sglang as sgl

@sgl.function
def multi_turn_qa(s, question_1, question_2):
    s += sgl.system("你是一个知识渊博的助手。")
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))

state = multi_turn_qa.run(
    question_1="什么是 Transformer？",
    question_2="它和 RNN 的核心区别是什么？",
)

print(state["answer_1"])
print(state["answer_2"])
```

---

## 7. TensorRT-LLM：NVIDIA 的极致优化

### 7.1 特点

- NVIDIA 官方出品，针对 NVIDIA GPU 深度优化。
- 编译式优化：将模型编译为优化后的 TensorRT 引擎。
- 支持 INT4/INT8/FP8 量化、KV Cache 量化。
- 支持投机解码（Speculative Decoding）。

### 7.2 工作流程

```
┌──────────────────────────────────────────────────────────┐
│  TensorRT-LLM 工作流                                     │
│                                                          │
│  HuggingFace 模型                                        │
│       │                                                  │
│       ▼                                                  │
│  模型转换 (convert_checkpoint)                            │
│       │                                                  │
│       ▼                                                  │
│  构建引擎 (trtllm-build)                                 │
│       │  - 选择量化方式                                   │
│       │  - 设置 TP/PP 并行度                              │
│       │  - 配置最大 batch/seq_len                         │
│       ▼                                                  │
│  TRT 引擎文件 (.engine)                                  │
│       │                                                  │
│       ▼                                                  │
│  推理 (通过 Python API 或 Triton 后端)                    │
└──────────────────────────────────────────────────────────┘
```

### 7.3 基本使用示例

```bash
# 安装（推荐使用 NVIDIA 官方 Docker 镜像）
# docker pull nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3

pip install tensorrt-llm

# 转换 checkpoint
python convert_checkpoint.py \
    --model_dir ./Qwen2.5-7B-Instruct \
    --output_dir ./trt_ckpt \
    --dtype float16

# 构建引擎
trtllm-build \
    --checkpoint_dir ./trt_ckpt \
    --output_dir ./trt_engine \
    --gemm_plugin float16 \
    --max_batch_size 16 \
    --max_input_len 2048 \
    --max_seq_len 4096
```

```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir("./trt_engine")

outputs = runner.generate(
    batch_input_ids=[tokenizer.encode("什么是 TensorRT-LLM？")],
    max_new_tokens=256,
    temperature=0.7,
)
print(tokenizer.decode(outputs[0][0]))
```

> **注意**：TensorRT-LLM 的使用门槛较高，版本更新频繁，API 变化大。
> 建议优先使用 vLLM/SGLang，性能不满足时再考虑 TensorRT-LLM。

---

## 8. llama.cpp / llama-cpp-python：CPU 与边缘设备推理

### 8.1 GGUF 格式

GGUF（GPT-Generated Unified Format）是 llama.cpp 使用的模型格式：

- 单文件，包含模型权重 + 元数据 + 分词器
- 支持多种量化：Q4_K_M, Q5_K_M, Q8_0, F16 等
- 针对 CPU 推理优化（SIMD 指令）
- 也支持 GPU offload（CUDA / Metal）

```bash
# 从 HuggingFace 下载 GGUF 模型
# 社区通常已经转好了各种量化版本
# 例：Qwen2.5-7B-Instruct-GGUF

# 也可以自己转换
pip install llama-cpp-python

# 使用 llama.cpp 的 convert 工具
python convert_hf_to_gguf.py ./Qwen2.5-7B-Instruct --outtype f16
# 再量化
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
```

### 8.2 量化等级选择

| 量化等级 | 大小（7B 模型） | 质量 | 速度 | 推荐场景 |
|---|---|---|---|---|
| F16 | ~14 GB | 最好 | 最慢 | GPU 推理，不需要量化 |
| Q8_0 | ~7.5 GB | 接近 F16 | 快 | GPU 显存充足时 |
| Q5_K_M | ~5.0 GB | 很好 | 很快 | 平衡首选 |
| Q4_K_M | ~4.2 GB | 好 | 最快 | 内存有限 / CPU 推理 |
| Q3_K_M | ~3.3 GB | 可接受 | 极快 | 极端内存限制 |
| Q2_K | ~2.7 GB | 明显下降 | 极快 | 不推荐 |

### 8.3 llama-cpp-python 使用

```bash
# CPU only
pip install llama-cpp-python

# 带 CUDA 加速
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./qwen2.5-7b-instruct-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,  # -1 = 全部 offload 到 GPU
    verbose=False,
)

# Chat 模式
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "什么是 GGUF 格式？"},
    ],
    temperature=0.7,
    max_tokens=512,
    stream=True,
)

for chunk in response:
    delta = chunk["choices"][0]["delta"]
    if "content" in delta:
        print(delta["content"], end="", flush=True)
```

### 8.4 Grammar 约束解码

llama.cpp 支持用 GBNF（类 BNF）语法约束输出格式：

```python
grammar_text = r'''
root   ::= object
object ::= "{" ws "\"name\"" ws ":" ws string "," ws "\"age\"" ws ":" ws number ws "}"
string ::= "\"" [a-zA-Z ]+ "\""
number ::= [0-9]+
ws     ::= [ \t\n]*
'''

from llama_cpp import LlamaGrammar

grammar = LlamaGrammar.from_string(grammar_text)

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "介绍一下自己（JSON 格式）"}],
    grammar=grammar,
    max_tokens=100,
)
print(response["choices"][0]["message"]["content"])
# 输出一定符合 {"name": "...", "age": ...} 格式
```

---

## 9. 投机解码（Speculative Decoding）

### 核心思想

用一个**小模型（Draft Model）** 快速生成多个候选 token，再用**大模型（Target Model）** 一次性验证。

```
┌────────────────────────────────────────────────┐
│  普通解码：                                     │
│  大模型 → t1 → 大模型 → t2 → 大模型 → t3       │
│  (3 次前向传播)                                 │
│                                                │
│  投机解码：                                     │
│  小模型 → t1,t2,t3,t4（草稿）                   │
│  大模型 → 验证（1 次前向传播）                   │
│  接受 t1,t2,t3 ✓  拒绝 t4 ✗ → 用大模型的 t4'  │
│  (小模型 4 次 + 大模型 1 次 ≈ 大模型 1 次成本)  │
└────────────────────────────────────────────────┘
```

**关键**：验证是并行的（像 prefill 一样），所以一次验证多个 token 的成本接近生成一个 token。

```python
# vLLM 中使用投机解码
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-72B-Instruct",
    speculative_model="Qwen/Qwen2.5-0.5B-Instruct",
    num_speculative_tokens=5,
    tensor_parallel_size=4,
)

outputs = llm.generate(
    ["写一首关于人工智能的诗"],
    SamplingParams(temperature=0.0, max_tokens=256),
)
```

> 投机解码在 `temperature=0` 时保证**完全无损**（和大模型独立生成的结果完全一致）。

---

## 10. 性能对比

| 引擎 | 核心优势 | 吞吐 | 延迟 | 硬件要求 | 易用性 | 适用场景 |
|---|---|---|---|---|---|---|
| **vLLM** | PagedAttention + 连续批处理 | ★★★★★ | ★★★★ | NVIDIA GPU | ★★★★★ | 通用 LLM 服务 |
| **SGLang** | RadixAttention + 结构化生成 | ★★★★★ | ★★★★ | NVIDIA GPU | ★★★★ | 结构化输出 / 复杂 LLM 程序 |
| **TensorRT-LLM** | 编译优化 + FP8 | ★★★★★ | ★★★★★ | NVIDIA GPU | ★★ | 极致性能（大规模部署） |
| **llama.cpp** | CPU/边缘 + GGUF 量化 | ★★★ | ★★★ | CPU/Metal/CUDA | ★★★★★ | 本地 / 边缘 / 低资源 |
| **Ollama** | llama.cpp 封装 | ★★★ | ★★★ | CPU/Metal/CUDA | ★★★★★★ | 个人使用 / 快速体验 |

---

## 11. 如何选择：决策树

```
你的场景是什么？
│
├─→ 个人使用 / 快速体验 / Mac 笔记本
│   └─→ Ollama 或 llama.cpp（GGUF 量化模型）
│
├─→ 团队 API 服务（< 100 QPS）
│   └─→ vLLM（最成熟、社区最大、OpenAI 兼容）
│
├─→ 需要结构化输出（JSON/正则约束）
│   └─→ SGLang（原生支持，性能好）
│
├─→ 大规模生产部署（> 1000 QPS）
│   ├─→ 追求极致性能 → TensorRT-LLM
│   └─→ 追求稳定易维护 → vLLM
│
├─→ 边缘设备 / 嵌入式 / CPU only
│   └─→ llama.cpp
│
└─→ 复杂 LLM 程序（多轮 / 分支 / 并行调用）
    └─→ SGLang（前端 DSL 天然支持）
```

---

## 12. 基准测试：如何正确衡量推理性能

### 12.1 核心指标

| 指标 | 含义 | 公式 / 测量方式 |
|---|---|---|
| **TTFT** | Time To First Token（首 token 延迟） | 从请求发出到第一个 token 返回 |
| **TPOT** | Time Per Output Token（每 token 延迟） | 后续每个 token 的平均生成时间 |
| **Throughput** | 吞吐 | 系统每秒处理的总 token 数 |
| **QPS** | 每秒请求数 | 每秒能完成的完整请求数 |
| **E2E Latency** | 端到端延迟 | TTFT + (output_tokens × TPOT) |

### 12.2 使用 benchmark 工具

```bash
# vLLM 自带 benchmark
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct --port 8000

# 使用 vLLM benchmark 脚本
python benchmarks/benchmark_serving.py \
    --backend openai \
    --base-url http://localhost:8000 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset-name sharegpt \
    --num-prompts 200 \
    --request-rate 10
```

### 12.3 常见基准测试陷阱

| 陷阱 | 说明 |
|---|---|
| 只测单请求延迟 | 不反映并发场景，必须测不同并发下的吞吐 |
| 不区分 TTFT 和 TPOT | Prefill 和 Decode 性能特征完全不同 |
| prompt 长度不真实 | 用固定短 prompt 测试不反映实际负载 |
| 忽略冷启动 | 第一次请求会有模型加载、CUDA Graph 构建等开销 |
| 不控制输出长度 | `max_tokens` 不同，结果不可比 |
| 混淆 tokens/s 的含义 | 是 per-request 还是 system-wide？ |

---

## 13. 常见坑

### 13.1 显存不够

```
# 常见错误
torch.cuda.OutOfMemoryError: CUDA out of memory

# 解决方案（按优先级）：
# 1. 降低 max_model_len
# 2. 使用量化（AWQ/GPTQ/FP8）
# 3. 降低 gpu_memory_utilization
# 4. 使用张量并行（多卡）
# 5. 使用更小的模型
```

### 13.2 模型加载慢

```python
# 大模型首次加载需要时间（权重从磁盘 → CPU → GPU）
# 技巧：
# 1. 使用 safetensors 格式（mmap 加载，比 pickle 快很多）
# 2. 本地缓存模型，避免每次下载
# 3. vLLM 的 --load-format auto 会自动选择最优方式
```

### 13.3 量化模型质量下降

```python
# 量化不是免费的——低精度会损失质量
# 推荐测试流程：
# 1. 先用 FP16 测一组 benchmark（作为基线）
# 2. 量化后在同一组 benchmark 上测
# 3. 对比关键指标：准确率、困惑度、人类评估
# 4. 不同任务对量化的敏感度不同

# 经验法则：
# - 7B+ 模型：Q4_K_M 通常可接受
# - 1-3B 模型：建议至少 Q5 或 Q8
# - 数学/代码任务对量化更敏感
```

### 13.4 OpenAI SDK 版本不兼容

```python
# vLLM / SGLang 提供 OpenAI 兼容接口，但某些高级功能可能不支持
# 常见问题：
# - tool_choice / function_calling 支持不完整
# - 某些参数（如 logprobs）行为可能有差异
# - 不同版本的 vLLM 对 openai SDK 版本的兼容性不同

# 建议：固定 openai SDK 版本
# pip install openai==1.40.0
```

### 13.5 并发测试时性能反而下降

```python
# 原因：过高的并发会导致 KV Cache 不够用，触发请求排队或 preemption
# vLLM 在显存不足时会 swap（把 KV Cache 移到 CPU），性能骤降

# 解决：
# 1. 监控 GPU 显存使用率（nvidia-smi）
# 2. 合理设置 max_num_seqs（最大并发数）
# 3. 观察是否有 preemption 日志
```

---

## 14. 小结

| 概念 | 一句话 |
|---|---|
| Prefill / Decode | 推理两阶段：计算密集 vs 内存密集 |
| KV Cache | 缓存历史 K/V 避免重复计算，但极耗显存 |
| Continuous Batching | 迭代级调度，短请求不等长请求 |
| PagedAttention | 分页管理 KV Cache，显存利用率 >95% |
| vLLM | 最流行、最易用、社区最大 |
| SGLang | RadixAttention + 结构化生成 |
| TensorRT-LLM | NVIDIA 极致优化，门槛高 |
| llama.cpp | CPU/边缘设备，GGUF 格式 |
| Speculative Decoding | 小模型打草稿，大模型验证 |

**一句话**：**推理优化 = 让 GPU 少搬数据、多干活、别闲着。**
