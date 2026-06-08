# 模块 07：AIGC 推理与部署

> 训练只是起点，推理才是终局。
> 模型如果不能高效地跑在生产环境，再高的 SOTA 也只是论文里的数字。
> 本模块带你从"模型训好了"跨越到"用户能用了"。

---

## 为什么推理优化至关重要？

### 训练 vs 推理：完全不同的战场

| 维度 | 训练 | 推理 |
|---|---|---|
| 目标 | 最小化 loss | 最小化延迟、最大化吞吐 |
| 批大小 | 越大越好（利用率高） | 由用户请求决定，通常很小 |
| 精度 | FP32 / BF16 | FP16 / INT8 / INT4 都行 |
| 显存 | 主要用于梯度和优化器状态 | 主要用于 KV Cache 和模型权重 |
| 成本结构 | 一次性投入 | 持续产生（跟请求量成正比） |
| 容忍度 | 慢一点没关系 | 用户在等，每 100ms 都敏感 |

### 推理优化的三个核心指标

```
┌─────────────────────────────────────────────┐
│           推理优化三角                        │
│                                             │
│              Latency                        │
│             (首 token 延迟)                  │
│               /    \                        │
│              /      \                       │
│             /        \                      │
│         Cost ──────── Throughput            │
│       (每千 token     (tokens/sec           │
│        成本)           并发吞吐)             │
│                                             │
│  三者互相制约，优化就是在三者间找平衡          │
└─────────────────────────────────────────────┘
```

- **延迟（Latency）**：用户发出请求到看到第一个 token 的时间（TTFT）。
- **吞吐（Throughput）**：单位时间内系统能处理的总 token 数。
- **成本（Cost）**：每处理 1000 个 token 消耗的 GPU 时间 / 电费。

一个好的推理系统，能用更少的 GPU 服务更多的用户，同时保持低延迟。

---

## 学习内容

| # | 文档 | 核心话题 |
|---|---|---|
| 00 | [inference-and-deployment-theory](./00-inference-and-deployment-theory.md) | 推理性能模型 / KV Cache 显存模型 / 批处理调度 / 扩散采样理论 / 服务容量规划 |
| 01 | [llm-inference-engines](./01-llm-inference-engines.md) | vLLM / SGLang / TensorRT-LLM / llama.cpp / KV Cache / PagedAttention / 连续批处理 |
| 02 | [diffusion-acceleration](./02-diffusion-acceleration.md) | Scheduler 优化 / torch.compile / TensorRT / 注意力优化 / 蒸馏加速 / 实时生成 |
| 03 | [serving-frameworks](./03-serving-frameworks.md) | FastAPI / Triton Server / BentoML / Ray Serve / 监控 / Docker / GPU 调度 |
| 04 | [demo-and-frontend](./04-demo-and-frontend.md) | Gradio / Streamlit / 流式聊天 / 图像生成 UI / HuggingFace Spaces / Open WebUI |

---

## 示例代码（`examples/`）

`examples/` 目录提供了可在当前 `aigc` 环境运行的 CPU 小实验，避免下载大模型或依赖 GPU：

| 文件 | 说明 | 当前环境是否可跑 |
|---|---|---|
| [`common.py`](./examples/common.py) | KV Cache 计算、toy chat、SSE 响应等共享工具 | 是 |
| [`kv_cache_and_batching.py`](./examples/kv_cache_and_batching.py) | KV Cache 显存估算、静态批处理 vs 连续批处理、Paged KV 浪费模拟 | 是 |
| [`diffusion_acceleration_sim.py`](./examples/diffusion_acceleration_sim.py) | 用 tiny latent 模拟 scheduler 步数、延迟和质量权衡 | 是 |
| [`openai_compatible_toy_server.py`](./examples/openai_compatible_toy_server.py) | 标准库实现的 OpenAI 兼容 toy server，支持非流式和 SSE 流式 | 是 |
| [`fastapi_gateway.py`](./examples/fastapi_gateway.py) | FastAPI 网关模板；`--self-test` 不要求安装 FastAPI | 自测可跑，启动服务需可选依赖 |
| [`demo_apps.py`](./examples/demo_apps.py) | Gradio / Streamlit demo 模板；`--mode self-test` 不要求安装 UI 依赖 | 自测可跑，启动 UI 需可选依赖 |
| [`smoke_test.py`](./examples/smoke_test.py) | 一键运行本模块全部可验证示例 | 是 |

### 在当前 `aigc` 环境运行

```bash
cd aigc-learning/07-inference-and-deployment/examples

conda run -n aigc python kv_cache_and_batching.py --model llama2-7b --batch-size 4
conda run -n aigc python diffusion_acceleration_sim.py --latent-size 32
conda run -n aigc python openai_compatible_toy_server.py --self-test
conda run -n aigc python fastapi_gateway.py --self-test
conda run -n aigc python demo_apps.py --mode self-test

# 一键验证
conda run -n aigc python smoke_test.py
```

### 可选 Web/UI 依赖

当前 `aigc` 环境已具备 `torch`、`numpy`、`pydantic`、`openai`、`httpx`，但未安装 `fastapi`、`uvicorn`、`gradio`、`streamlit`。如需启动真实服务和 UI：

```bash
pip install fastapi uvicorn gradio streamlit prometheus-client
```

启动 toy OpenAI 兼容服务：

```bash
conda run -n aigc python openai_compatible_toy_server.py --host 127.0.0.1 --port 8000
```

安装可选依赖后启动 FastAPI 网关：

```bash
conda run -n aigc python fastapi_gateway.py --backend-url http://127.0.0.1:8000 --port 9000
```

安装可选依赖后启动 Demo：

```bash
conda run -n aigc python demo_apps.py --mode gradio
conda run -n aigc streamlit run demo_apps.py -- --mode streamlit
```

---

## 推荐配套资源

| 类型 | 资源 | 说明 |
|---|---|---|
| 文档 | [vLLM 官方文档](https://docs.vllm.ai/) | 最流行的 LLM 推理引擎 |
| 文档 | [SGLang 官方文档](https://sgl-project.github.io/) | RadixAttention + 结构化生成 |
| 文档 | [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) | NVIDIA 官方 LLM 推理优化 |
| 文档 | [llama.cpp](https://github.com/ggml-org/llama.cpp) | CPU/边缘设备推理 |
| 文档 | [Diffusers 优化指南](https://huggingface.co/docs/diffusers/optimization/overview) | 扩散模型加速全攻略 |
| 文档 | [FastAPI 文档](https://fastapi.tiangolo.com/) | 现代 Python Web 框架 |
| 文档 | [Triton Inference Server](https://github.com/triton-inference-server/server) | NVIDIA 生产级推理服务器 |
| 文档 | [Gradio 文档](https://www.gradio.app/docs/) | 最快搭 ML Demo 的方式 |
| 文档 | [Streamlit 文档](https://docs.streamlit.io/) | 数据应用 / 交互式 Demo |
| 论文 | [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | vLLM 核心论文 |
| 论文 | [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) | SGLang 核心论文 |

---

## 前置知识

开始本模块前，建议你已经掌握：

- [x] PyTorch 基础（Tensor、Module、device）→ 模块 02
- [x] Transformer 结构（Attention、KV Cache 概念）→ `02-deep-learning-libraries/05-transformer-from-scratch.md`
- [x] HuggingFace Transformers / Diffusers 基本用法 → 模块 02
- [x] Python async/await 基础（服务化会用到）→ `01-python-foundations/03-async-programming.md`
- [x] 基本的命令行和 Docker 概念

---

## 自检清单

学完本模块，你应该能自信地回答以下问题：

- [ ] LLM 推理中 prefill 和 decode 阶段分别是计算密集还是内存密集？为什么？
- [ ] KV Cache 的显存开销怎么估算？7B 模型、4096 上下文需要多少 GB？
- [ ] 连续批处理（Continuous Batching）相比静态批处理的优势是什么？
- [ ] PagedAttention 解决了什么问题？为什么传统 KV Cache 浪费显存？
- [ ] vLLM 和 SGLang 各自的核心创新是什么？什么场景选哪个？
- [ ] llama.cpp 的 GGUF 格式是什么？为什么它能在 CPU 上高效推理？
- [ ] 扩散模型为什么慢？有哪些方法可以减少采样步数？
- [ ] `torch.compile` 对扩散模型的加速原理是什么？
- [ ] FastAPI 中如何实现 SSE 流式输出？为什么 LLM 服务需要流式？
- [ ] Triton Inference Server 的动态批处理是怎么工作的？
- [ ] Gradio 的 `Interface` 和 `Blocks` 有什么区别？什么时候用哪个？
- [ ] 如何把一个 Gradio Demo 部署到 HuggingFace Spaces？
- [ ] 推理服务的监控应该关注哪些指标？
- [ ] 为什么 LLM API 普遍采用 OpenAI 兼容格式？
