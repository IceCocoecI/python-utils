# 03 · 推理服务框架

> 模型能跑不代表能上线。
> 从 `model.generate()` 到"稳定服务 1000 个并发用户"之间，隔着一整套工程体系。

---

## 1. 从 Notebook 到生产：什么变了？

| 维度 | Notebook 实验 | 生产服务 |
|---|---|---|
| 请求方式 | 手动运行一次 | HTTP API，任意时间、任意数量 |
| 并发 | 1 个请求 | 数十到数千并发 |
| 可用性 | 挂了重启 | 99.9%+ SLA，自动恢复 |
| 延迟 | 无所谓 | P99 < 2 秒 |
| 监控 | `print()` | Prometheus + Grafana + 报警 |
| 资源 | 独占一张卡 | 多模型共享，动态调度 |
| 安全 | 无 | 认证、限流、输入校验 |
| 部署 | `python script.py` | Docker + K8s + CI/CD |

**核心挑战**：

```
┌──────────────────────────────────────────────────────┐
│  生产推理服务的核心挑战                                │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐      │
│  │ 高并发   │  │ 低延迟   │  │ 高可用        │      │
│  │ batching │  │ streaming│  │ health check  │      │
│  │ 队列管理 │  │ 异步处理 │  │ 优雅重启      │      │
│  └──────────┘  └──────────┘  └───────────────┘      │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐      │
│  │ 可观测   │  │ 资源管理 │  │ 安全          │      │
│  │ metrics  │  │ GPU 调度 │  │ auth / 限流   │      │
│  │ logging  │  │ 显存管理 │  │ 输入校验      │      │
│  └──────────┘  └──────────┘  └───────────────┘      │
└──────────────────────────────────────────────────────┘
```

---

## 2. OpenAI 兼容 API：事实标准

### 2.1 为什么重要

几乎所有 LLM 应用框架（LangChain、LlamaIndex、OpenAI SDK）都支持 OpenAI API 格式。
使你的模型兼容这个格式，意味着：

- 所有现有工具链直接可用
- 切换模型只需改 `base_url`，不改代码
- 用户迁移成本为零

### 2.2 核心接口

```
POST /v1/chat/completions

请求体：
{
    "model": "model-name",
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ],
    "temperature": 0.7,
    "max_tokens": 512,
    "stream": true
}

流式响应（SSE）：
data: {"choices": [{"delta": {"content": "Hello"}}]}
data: {"choices": [{"delta": {"content": " world"}}]}
data: [DONE]
```

---

## 3. FastAPI：最流行的 ML 服务框架

### 3.1 为什么选 FastAPI

- **原生 async**：适合 IO 密集的推理服务（等待 GPU 计算时不阻塞）。
- **自动文档**：Swagger UI (`/docs`) 和 ReDoc (`/redoc`)。
- **Pydantic 验证**：请求体自动校验。
- **性能**：底层基于 Starlette + Uvicorn，性能接近 Go/Node。
- **生态**：AIGC 社区事实标准（vLLM、SGLang 的 API Server 都用 FastAPI）。

### 3.2 基础模板：LLM 推理服务

```python
import asyncio
import time
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    yield
    del model, tokenizer
    torch.cuda.empty_cache()


app = FastAPI(title="LLM API", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    stream: bool = False


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str = "chatcmpl-001"
    object: str = "chat.completion"
    choices: list[ChatChoice]
    usage: dict


@app.get("/health")
async def health():
    return {"status": "ok", "gpu_available": torch.cuda.is_available()}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = tokenizer.apply_chat_template(
        [m.model_dump() for m in request.messages],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    if request.stream:
        return StreamingResponse(
            _stream_generate(inputs, request),
            media_type="text/event-stream",
        )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=request.temperature > 0,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return ChatResponse(
        choices=[
            ChatChoice(message=ChatMessage(role="assistant", content=response_text))
        ],
        usage={
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": len(new_tokens),
            "total_tokens": inputs["input_ids"].shape[1] + len(new_tokens),
        },
    )


async def _stream_generate(inputs, request: ChatRequest):
    """SSE 流式生成。"""
    from transformers import TextIteratorStreamer
    import json
    from threading import Thread

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    generation_kwargs = {
        **inputs,
        "max_new_tokens": request.max_tokens,
        "temperature": request.temperature,
        "do_sample": request.temperature > 0,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for text in streamer:
        chunk = {
            "id": "chatcmpl-001",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": text}}],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0)  # 让出事件循环

    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3.3 流式响应（SSE）详解

LLM 服务必须支持流式输出，原因：

1. **用户体验**：用户看到逐字出现，感觉更快（实际端到端延迟不变）。
2. **TTFT 敏感**：聊天场景下，首 token 延迟比总延迟更重要。
3. **超时友好**：长回复不会触发 HTTP 超时。

```python
# SSE (Server-Sent Events) 协议格式
# 每条消息以 "data: " 开头，两个换行结束
# 最后发送 "data: [DONE]" 表示结束

# 客户端调用示例
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 3.4 实战推荐：FastAPI + vLLM

在生产中，不要自己写生成逻辑，直接把 vLLM 当后端：

```python
from fastapi import FastAPI
from openai import AsyncOpenAI

app = FastAPI()

vllm_client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
)


@app.post("/chat")
async def chat(user_message: str):
    """在 vLLM 之上包一层业务逻辑。"""
    response = await vllm_client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": "你是一个翻译助手，只输出翻译结果。"},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return {"translation": response.choices[0].message.content}
```

> **架构建议**：vLLM 负责推理 + 调度，FastAPI 负责业务逻辑（鉴权、日志、后处理）。
> 这样可以独立扩缩容两层。

---

## 4. Triton Inference Server

### 4.1 是什么

NVIDIA Triton 是一个**通用的推理服务器**，支持多种框架（PyTorch、TensorRT、ONNX、TensorFlow）。

### 4.2 核心特性

| 特性 | 说明 |
|---|---|
| **动态批处理** | 自动合并小请求为大 batch |
| **模型仓库** | 规范化的模型目录结构 |
| **多模型** | 同一服务器上部署多个模型 |
| **Ensemble** | 多模型组成流水线（前处理 → 推理 → 后处理） |
| **多框架** | PyTorch, TensorRT, ONNX, TensorFlow, Python 后端 |
| **gRPC + HTTP** | 两种协议都支持 |
| **指标** | 内置 Prometheus 指标 |

### 4.3 模型仓库结构

```
model_repository/
├── text_generation/
│   ├── config.pbtxt
│   ├── 1/
│   │   └── model.py          # Python 后端
│   └── 2/                    # 版本 2（可选）
│       └── model.py
├── image_classifier/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx        # ONNX 模型
└── preprocessing/
    ├── config.pbtxt
    └── 1/
        └── model.py           # 前处理 Python 脚本
```

### 4.4 配置文件示例

```protobuf
# config.pbtxt
name: "text_generation"
backend: "python"
max_batch_size: 8

input [
  {
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

output [
  {
    name: "text_output"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

instance_group [
  {
    kind: KIND_GPU
    count: 1
    gpus: [ 0 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 100000
}
```

### 4.5 动态批处理

```
┌──────────────────────────────────────────────┐
│  动态批处理                                   │
│                                              │
│  请求队列:  [R1] [R2] [R3]   [R4]           │
│  时间:       t0   t1   t2     t3             │
│                                              │
│  等待窗口内（max_queue_delay）:               │
│  → 合并为一个 batch: [R1, R2, R3]            │
│  → 一次推理完成所有请求                       │
│                                              │
│  效果：吞吐 ↑ 3x，延迟仅增加 ~100ms          │
└──────────────────────────────────────────────┘
```

### 4.6 启动 Triton

```bash
# 使用 Docker（推荐）
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.07-py3 \
    tritonserver --model-repository=/models

# 端口说明：
# 8000 = HTTP
# 8001 = gRPC
# 8002 = Prometheus metrics
```

### 4.7 客户端调用

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")

text_input = np.array(["What is deep learning?"], dtype=object).reshape(1, 1)

inputs = [httpclient.InferInput("text_input", [1, 1], "BYTES")]
inputs[0].set_data_from_numpy(text_input)

outputs = [httpclient.InferRequestedOutput("text_output")]

result = client.infer("text_generation", inputs=inputs, outputs=outputs)
print(result.as_numpy("text_output"))
```

---

## 5. BentoML

### 5.1 核心思路

BentoML 把模型打包为标准化的 **Bento**（包含模型、代码、依赖），然后一键容器化和部署。

### 5.2 基本使用

```python
# service.py
import bentoml
from transformers import pipeline


@bentoml.service(
    resources={"gpu": 1},
    traffic={"timeout": 300},
)
class TextGeneration:
    def __init__(self):
        self.pipe = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-7B-Instruct",
            torch_dtype="float16",
            device_map="auto",
        )

    @bentoml.api
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        result = self.pipe(prompt, max_new_tokens=max_tokens)
        return result[0]["generated_text"]
```

```bash
# 本地运行
bentoml serve service:TextGeneration

# 构建 Bento
bentoml build

# 容器化
bentoml containerize text_generation:latest

# 运行 Docker
docker run --gpus all -p 3000:3000 text_generation:latest
```

---

## 6. Ray Serve

### 6.1 核心特性

- **分布式**：天然支持多节点、多 GPU 扩缩容。
- **组合**：多个模型可以组成 DAG（有向无环图）。
- **自动扩缩**：根据负载自动增减副本。
- **与 Ray 生态集成**：Ray Data、Ray Train 无缝配合。

### 6.2 基本使用

```python
import ray
from ray import serve
from transformers import pipeline


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_ongoing_requests": 5,
    },
)
class LLMDeployment:
    def __init__(self):
        self.pipe = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-7B-Instruct",
            torch_dtype="float16",
            device_map="auto",
        )

    async def __call__(self, request):
        data = await request.json()
        result = self.pipe(data["prompt"], max_new_tokens=data.get("max_tokens", 256))
        return {"text": result[0]["generated_text"]}


app = LLMDeployment.bind()

# 启动
# serve run main:app
```

### 6.3 模型组合（Pipeline）

```python
@serve.deployment
class Preprocessor:
    def preprocess(self, text: str) -> str:
        return text.strip().lower()


@serve.deployment(ray_actor_options={"num_gpus": 1})
class Model:
    def __init__(self):
        self.pipe = pipeline("text-generation", model="...")

    def generate(self, text: str) -> str:
        return self.pipe(text)[0]["generated_text"]


@serve.deployment
class Postprocessor:
    def postprocess(self, text: str) -> str:
        return text.strip()


@serve.deployment
class Pipeline:
    def __init__(self, preprocessor, model, postprocessor):
        self.preprocessor = preprocessor
        self.model = model
        self.postprocessor = postprocessor

    async def __call__(self, request):
        data = await request.json()
        text = await self.preprocessor.preprocess.remote(data["text"])
        generated = await self.model.generate.remote(text)
        result = await self.postprocessor.postprocess.remote(generated)
        return {"result": result}


preprocessor = Preprocessor.bind()
model = Model.bind()
postprocessor = Postprocessor.bind()
app = Pipeline.bind(preprocessor, model, postprocessor)
```

---

## 7. LitServe

LitServe（by Lightning AI）是一个轻量级的推理服务框架，核心卖点是**简单 + 自动 batching**。

```python
import litserve as ls
from transformers import pipeline


class LLMServingAPI(ls.LitAPI):
    def setup(self, device):
        self.pipe = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-7B-Instruct",
            torch_dtype="float16",
            device=device,
        )

    def decode_request(self, request):
        return request["prompt"]

    def predict(self, prompt):
        return self.pipe(prompt, max_new_tokens=256)[0]["generated_text"]

    def encode_response(self, output):
        return {"text": output}


server = ls.LitServer(LLMServingAPI(), accelerator="gpu", max_batch_size=8)
server.run(port=8000)
```

---

## 8. 负载均衡与扩缩容

### 8.1 基本架构

```
┌─────────────────────────────────────────────┐
│  典型生产架构                                │
│                                             │
│  Client                                     │
│    │                                        │
│    ▼                                        │
│  Nginx / Envoy / Cloud LB                   │
│    │                                        │
│    ├──→ API Server (Pod 1) ──→ vLLM (GPU 0)│
│    ├──→ API Server (Pod 2) ──→ vLLM (GPU 1)│
│    └──→ API Server (Pod 3) ──→ vLLM (GPU 2)│
│                                             │
│  横向扩展：增加更多 Pod + GPU               │
└─────────────────────────────────────────────┘
```

### 8.2 Nginx 反向代理配置

```nginx
upstream llm_backend {
    least_conn;
    server gpu-server-1:8000;
    server gpu-server-2:8000;
    server gpu-server-3:8000;
}

server {
    listen 80;

    location /v1/ {
        proxy_pass http://llm_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";

        # SSE 流式响应需要关闭缓冲
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}
```

---

## 9. 监控

### 9.1 关键指标

| 指标 | 含义 | 报警阈值 |
|---|---|---|
| `request_latency_p99` | P99 延迟 | > 5s |
| `request_throughput` | QPS | 接近容量上限 |
| `gpu_utilization` | GPU 使用率 | < 30%（浪费）或 > 95%（过载） |
| `gpu_memory_used` | 显存使用量 | > 90% |
| `active_requests` | 进行中的请求数 | > max_batch_size × 2 |
| `queue_length` | 排队请求数 | > 0 持续增长 |
| `error_rate` | 错误率 | > 1% |
| `ttft_p50` | 首 token 延迟中位数 | > 1s |

### 9.2 Prometheus 指标暴露

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI, Response

app = FastAPI()

REQUEST_COUNT = Counter("request_total", "Total requests", ["method", "status"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency",
                            buckets=[0.1, 0.5, 1, 2, 5, 10, 30])
ACTIVE_REQUESTS = Gauge("active_requests", "Currently processing requests")
GPU_MEMORY = Gauge("gpu_memory_used_bytes", "GPU memory used")


@app.middleware("http")
async def metrics_middleware(request, call_next):
    ACTIVE_REQUESTS.inc()
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    REQUEST_LATENCY.observe(duration)
    REQUEST_COUNT.labels(method=request.method, status=response.status_code).inc()
    ACTIVE_REQUESTS.dec()
    return response


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

### 9.3 vLLM 内置指标

vLLM 自带 Prometheus 指标端点：

```bash
# 启动 vLLM 时自动暴露 /metrics
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# 关键指标：
# vllm:num_requests_running    — 正在处理的请求
# vllm:num_requests_waiting    — 排队中的请求
# vllm:gpu_cache_usage_perc    — KV Cache 使用率
# vllm:avg_generation_throughput_toks_per_s — 吞吐
```

---

## 10. 认证、限流与错误处理

### 10.1 API Key 认证

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEYS = {"sk-prod-key-001", "sk-dev-key-001"}
api_key_header = APIKeyHeader(name="Authorization")


async def verify_api_key(api_key: str = Security(api_key_header)):
    token = api_key.replace("Bearer ", "")
    if token not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return token


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest):
    ...
```

### 10.2 限流

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)


@app.post("/v1/chat/completions")
@limiter.limit("60/minute")
async def chat(request: ChatRequest):
    ...
```

### 10.3 错误处理

```python
from fastapi import Request
from fastapi.responses import JSONResponse


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
            }
        },
    )


@app.exception_handler(torch.cuda.OutOfMemoryError)
async def oom_handler(request: Request, exc: torch.cuda.OutOfMemoryError):
    torch.cuda.empty_cache()
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "message": "GPU out of memory, please retry",
                "type": "insufficient_resources",
            }
        },
    )
```

---

## 11. Docker 容器化

### 11.1 Dockerfile 模板

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 11.2 requirements.txt 示例

```txt
torch>=2.4.0
transformers>=4.44.0
vllm>=0.6.0
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
prometheus-client>=0.20.0
```

### 11.3 构建与运行

```bash
# 构建
docker build -t llm-service:latest .

# 运行（挂载模型缓存，避免每次下载）
docker run --gpus all -d \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=hf_xxx \
    --name llm-service \
    llm-service:latest
```

### 11.4 Docker Compose（多服务）

```yaml
# docker-compose.yml
services:
  vllm:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8000:8000"
    volumes:
      - model-cache:/root/.cache/huggingface
    command: >
      --model Qwen/Qwen2.5-7B-Instruct
      --max-model-len 4096
      --gpu-memory-utilization 0.9

  api:
    build: ./api
    ports:
      - "80:8000"
    depends_on:
      - vllm
    environment:
      - VLLM_URL=http://vllm:8000

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus

volumes:
  model-cache:
```

---

## 12. GPU 资源管理

### 12.1 多模型共享 GPU

```python
# 方法 1：CUDA_VISIBLE_DEVICES 隔离
# 不同进程看到不同的 GPU

# 进程 1：LLM
# CUDA_VISIBLE_DEVICES=0 python llm_server.py

# 进程 2：扩散模型
# CUDA_VISIBLE_DEVICES=1 python diffusion_server.py
```

### 12.2 显存预算规划

```
┌──────────────────────────────────────────────┐
│  A100 80GB 显存预算示例                       │
│                                              │
│  模型权重 (7B FP16):     ~14 GB              │
│  KV Cache:               ~40 GB              │
│  CUDA Context:           ~1 GB               │
│  临时张量:               ~2 GB               │
│  ─────────────────────────                   │
│  总计:                   ~57 GB              │
│  剩余（安全余量）:       ~23 GB              │
│                                              │
│  → gpu_memory_utilization = 57/80 ≈ 0.71    │
│  → 实际设 0.85 让 vLLM 自己分配             │
└──────────────────────────────────────────────┘
```

---

## 13. 完整示例：FastAPI + vLLM 流式 API

```python
"""
生产级 LLM API 服务：FastAPI 作为网关，vLLM 作为推理后端。

启动顺序：
1. vllm serve Qwen/Qwen2.5-7B-Instruct --port 9000
2. uvicorn gateway:app --host 0.0.0.0 --port 8000
"""

import json
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

VLLM_BASE_URL = "http://localhost:9000"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(base_url=VLLM_BASE_URL, timeout=300)
    yield
    await app.state.client.aclose()


app = FastAPI(title="LLM Gateway", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    stream: bool = False


@app.get("/health")
async def health():
    try:
        resp = await app.state.client.get("/health")
        return {"status": "ok", "backend": resp.status_code == 200}
    except httpx.RequestError:
        raise HTTPException(503, "Backend unavailable")


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [m.model_dump() for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": request.stream,
    }

    if request.stream:
        return StreamingResponse(
            _proxy_stream(payload),
            media_type="text/event-stream",
        )

    resp = await app.state.client.post("/v1/chat/completions", json=payload)
    return resp.json()


async def _proxy_stream(payload: dict):
    """代理 vLLM 的 SSE 流到客户端。"""
    async with app.state.client.stream(
        "POST", "/v1/chat/completions", json=payload
    ) as resp:
        async for line in resp.aiter_lines():
            if line:
                yield f"{line}\n\n"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 14. 框架选择指南

| 场景 | 推荐方案 | 原因 |
|---|---|---|
| 快速原型 / Demo | FastAPI | 简单、灵活、文档好 |
| LLM 生产服务 | vLLM + FastAPI 网关 | vLLM 处理推理，FastAPI 处理业务 |
| 多模型 / 多框架 | Triton Inference Server | 原生支持多模型、动态 batching |
| 分布式 / 自动扩缩 | Ray Serve | 天然分布式 |
| 打包交付 | BentoML | 标准化 Bento + 一键容器化 |
| 极简轻量 | LitServe | API 简单，自动 batching |

---

## 15. 常见坑

### 15.1 FastAPI 中的 sync vs async

```python
# 错误：在 async 函数中运行 CPU/GPU 密集任务
@app.post("/generate")
async def generate(prompt: str):
    result = model.generate(prompt)  # 阻塞事件循环！
    return {"text": result}

# 正确：用 run_in_executor 把 CPU 密集任务放到线程池
import asyncio

@app.post("/generate")
async def generate(prompt: str):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, model.generate, prompt)
    return {"text": result}

# 或者直接用 def（FastAPI 会自动放到线程池）
@app.post("/generate")
def generate(prompt: str):
    result = model.generate(prompt)
    return {"text": result}
```

### 15.2 模型加载时机

```python
# 错误：在模块顶层加载模型
model = load_model()  # Worker 进程 fork 时会出问题

# 正确：在 lifespan 中加载
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield
```

### 15.3 SSE 流被 Nginx 缓冲

```nginx
# 症状：流式响应变成了一次性返回
# 原因：Nginx 默认开启 proxy_buffering

# 修复：
location /v1/chat/completions {
    proxy_pass http://backend;
    proxy_buffering off;         # 关键
    proxy_cache off;
    chunked_transfer_encoding on;
}
```

### 15.4 Docker 中 GPU 不可用

```bash
# 确保安装了 nvidia-container-toolkit
# 且 docker run 带 --gpus all 参数

# 验证：
docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 15.5 多 Worker 与 GPU

```bash
# 错误：多 Worker 每个都加载一份模型
uvicorn main:app --workers 4  # 4 份模型，显存爆炸

# 推理服务通常用单 Worker
# 并发靠 async + 推理引擎自身的 batching
uvicorn main:app --workers 1
```

---

## 16. 小结

| 组件 | 一句话 |
|---|---|
| OpenAI 兼容 API | 事实标准，让工具链无缝接入 |
| FastAPI | ML 服务的首选 Web 框架 |
| SSE 流式 | LLM 必备，TTFT 和用户体验的关键 |
| Triton | 多模型、多框架、动态 batching |
| BentoML | 打包 + 容器化一条龙 |
| Ray Serve | 分布式 + 自动扩缩容 |
| Prometheus + Grafana | 可观测性标配 |
| Docker | 容器化部署标配 |

**一句话**：**用 vLLM 做推理，FastAPI 做网关，Docker 做容器化，Prometheus 做监控。**
