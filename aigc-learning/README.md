# AIGC 算法工程师编程学习路线（AIGC Learning Lab）

> 为 AIGC（AI Generated Content）算法工程师量身定制的编程能力养成手册。
> 内容编排采用「基础 → 进阶 → 高级」三段式递进，每一节都配有可运行的示例代码、
> 顶级开源参考项目与精选教程链接。

---

## 学习目标

完成本学习 Lab 后，你将具备以下能力：

1. 熟练使用现代 Python（类型注解、装饰器、生成器、async/await、上下文管理器）编写工程级代码。
2. 独立使用 PyTorch 实现和训练深度学习模型，理解 `autograd`、`nn.Module`、`DataLoader` 的工作机制。
3. 掌握 HuggingFace 三件套（Transformers / Diffusers / Datasets），能够独立完成 LLM 微调与扩散模型推理。
4. 精通 NumPy、einops 这两件写模型代码的"神器"，能用最少的代码写出清晰的张量运算。
5. 熟悉 AIGC 常用的图像 / 音频数据处理工具链。
6. 掌握训练工程化全流程：实验追踪、配置管理、超参搜索、可复现性。
7. 理解分布式训练：DDP、FSDP、DeepSpeed、多维并行策略（DP/TP/PP/EP）。
8. 独立完成大模型微调与对齐：LoRA/QLoRA、SFT、RLHF/DPO/GRPO。
9. 掌握 AIGC 推理部署全栈：vLLM/SGLang、FastAPI 服务化、Gradio Demo。
10. 构建 LLM 应用：RAG 全流程、向量数据库、Agent 工程。
11. 理解前沿 AIGC 模型架构：LLM（LLaMA/Qwen/MoE）、图像/视频/语音生成。
12. 具备 GPU 编程基础：CUDA kernel、Triton、性能剖析、自定义算子。

---

## 目录结构

```
aigc-learning/
├── README.md                          # 当前文件：总览 + 学习路线
├── THEORY.md                          # 理论知识地图：按概念主线阅读
├── CHEATSHEET.md                      # 速查表：日常写代码最常用的片段
├── requirements.txt                   # 依赖清单（按模块分组）
│
├── 01-python-foundations/             # 模块 01：现代 Python 编程基础与进阶
│   ├── README.md
│   ├── 00-python-engineering-theory.md  # Python 工程心智模型
│   ├── 01-modern-python-basics.md     # 数据结构 / 魔法方法 / 惯用法 / itertools / 陷阱
│   ├── 02-advanced-features.md        # 装饰器 / 生成器 / 上下文管理器 / functools
│   ├── 03-async-programming.md        # asyncio / async-await / 并发模型
│   ├── 04-type-hints.md               # typing / ParamSpec / Protocol / Pydantic
│   ├── 05-engineering-best-practices.md  # 项目结构 / pytest / 调试 / profiling / ruff
│   └── examples/                      # 可运行示例
│
├── 02-deep-learning-libraries/        # 模块 02：深度学习核心库
│   ├── README.md
│   ├── 00-deep-learning-theory.md     # 张量 / 梯度 / 优化 / Attention / 扩散 / 显存预算
│   ├── 01-pytorch-fundamentals.md     # Tensor / autograd / nn.Module / device
│   ├── 02-pytorch-training-loop.md    # DataLoader / AMP / 显存预算 / 训练调试
│   ├── 03-huggingface-transformers.md # Tokenizer / AutoModel / Trainer / PEFT 入门
│   ├── 04-huggingface-diffusers.md    # Pipeline / Scheduler / 微调与 LoRA 入门
│   ├── 05-transformer-from-scratch.md # 注意力 / 位置编码 / 掩码 / KV cache / FlashAttention
│   ├── 06-transformer-principles-overview.md  # Transformer 白话速览
│   └── examples/
│
├── 03-data-and-scientific-computing/  # 模块 03：数据处理与科学计算
│   ├── README.md
│   ├── 00-data-and-scientific-computing-theory.md  # shape / dtype / layout / range / distribution
│   ├── 01-numpy-essentials.md         # ndarray / 广播 / 向量化 / 内存视图
│   ├── 02-einops-tutorial.md          # rearrange / reduce / repeat / Layers
│   ├── 03-image-processing.md         # Pillow / OpenCV / torchvision / Albumentations / FID
│   ├── 04-data-formats-and-pipelines.md  # JSONL / Parquet / WebDataset / safetensors / HF datasets
│   └── examples/
│
├── 04-training-engineering/           # 模块 04：训练与实验工程化
│   ├── README.md
│   ├── 00-training-engineering-theory.md  # 实验记录 / 配置空间 / 搜索空间 / 可复现边界
│   ├── 01-experiment-tracking.md      # TensorBoard / wandb / MLflow
│   ├── 02-config-management.md        # Hydra / OmegaConf / 结构化配置
│   ├── 03-hyperparameter-search.md    # Optuna / Ray Tune / 搜索策略
│   ├── 04-reproducibility.md          # 随机种子 / 确定性训练 / 环境与数据版本管理
│   └── examples/
│
├── 05-distributed-training/           # 模块 05：分布式训练
│   ├── README.md
│   ├── 00-distributed-training-theory.md  # 通信原语 / 显存分片 / 并行策略 / 性能模型
│   ├── 01-distributed-basics-and-ddp.md  # torch.distributed / DDP 完整教程
│   ├── 02-fsdp.md                     # FSDP / FSDP2 全参数分片
│   ├── 03-accelerate.md               # HuggingFace Accelerate 一键分布式
│   ├── 04-deepspeed.md                # ZeRO-1/2/3 / CPU Offload
│   ├── 05-parallelism-strategies.md   # DP / TP / PP / EP / 3D 并行
│   └── examples/
│
├── 06-finetuning-and-alignment/       # 模块 06：大模型微调与对齐
│   ├── README.md
│   ├── 00-finetuning-and-alignment-theory.md  # PEFT / 量化 / SFT / 偏好优化理论
│   ├── 01-peft-and-lora.md            # LoRA / QLoRA / DoRA / peft 库
│   ├── 02-quantization.md             # INT8/INT4 / bitsandbytes / GPTQ / AWQ / GGUF
│   ├── 03-sft-data-and-training.md    # SFT 数据构造 / Chat Template / TRL SFTTrainer
│   ├── 04-alignment-rlhf-dpo.md       # RLHF / DPO / GRPO / KTO / TRL
│   └── examples/
│
├── 07-inference-and-deployment/       # 模块 07：AIGC 推理与部署
│   ├── README.md
│   ├── 00-inference-and-deployment-theory.md  # 推理性能模型 / 调度 / 容量规划
│   ├── 01-llm-inference-engines.md    # vLLM / SGLang / TensorRT-LLM / llama.cpp
│   ├── 02-diffusion-acceleration.md   # Scheduler 优化 / torch.compile / TensorRT / 蒸馏
│   ├── 03-serving-frameworks.md       # FastAPI / Triton / BentoML / 生产化
│   ├── 04-demo-and-frontend.md        # Gradio / Streamlit / 前端 Demo
│   └── examples/
│
├── 08-llm-applications/               # 模块 08：LLM 应用开发
│   ├── README.md
│   ├── 00-llm-applications-theory.md  # RAG / 向量检索 / 编排 / Agent / 安全评估理论
│   ├── 01-rag-fundamentals.md         # RAG 全流程：切分 / Embedding / 检索 / 生成
│   ├── 02-vector-databases.md         # FAISS / Milvus / Chroma / pgvector
│   ├── 03-orchestration-frameworks.md # LangChain / LlamaIndex / LangGraph
│   └── 04-agent-engineering.md        # Tool Use / MCP / Planning / Multi-Agent
│
├── 09-frontier-models/                # 模块 09：前沿 AIGC 模型架构
│   ├── README.md
│   ├── 00-frontier-models-theory.md   # 跨模态生成模型统一理论框架
│   ├── 01-llm-architectures.md        # GPT / LLaMA / Qwen / DeepSeek / MoE
│   ├── 02-image-generation.md         # DDPM → SD → SDXL → DiT → Flux 演进
│   ├── 03-multimodal-models.md        # CLIP / LLaVA / Qwen-VL / 视觉语言模型
│   ├── 04-video-generation.md         # Sora / CogVideoX / Wan / HunyuanVideo
│   └── 05-speech-and-audio.md         # VALL-E / CosyVoice / F5-TTS / Whisper
│
└── 10-cuda-and-triton/                # 模块 10：工程深水区
    ├── README.md
    ├── 00-gpu-performance-theory.md   # GPU 执行模型 / Roofline / fusion / tiling / profiling
    ├── 01-gpu-architecture-and-cuda-basics.md  # GPU 架构 / CUDA 编程模型 / kernel 编写
    ├── 02-triton-programming.md       # Triton：用 Python 写 GPU kernel
    ├── 03-performance-profiling.md    # torch.profiler / Nsight Systems / 性能分析
    └── 04-custom-operators-and-extensions.md  # pybind11 / cpp_extension / FlexAttention
```

---

## 推荐学习顺序

如果你想先按理论主线建立整体框架，可以先读 [`THEORY.md`](./THEORY.md)，再回到下面的实践路线。

### 第 1 阶段：打牢基础（约 2–3 周）

1. `01-python-foundations/00-python-engineering-theory.md`
2. `01-python-foundations/01-modern-python-basics.md`
3. `03-data-and-scientific-computing/01-numpy-essentials.md`
4. `01-python-foundations/02-advanced-features.md`
5. `01-python-foundations/04-type-hints.md`
6. `01-python-foundations/05-engineering-best-practices.md`（测试/调试/profiling——太多人跳过这步）

### 第 2 阶段：深度学习实战（约 3–4 周）

7. `02-deep-learning-libraries/01-pytorch-fundamentals.md`
8. `02-deep-learning-libraries/02-pytorch-training-loop.md`
9. `03-data-and-scientific-computing/02-einops-tutorial.md`
10. `03-data-and-scientific-computing/03-image-processing.md`
11. `03-data-and-scientific-computing/04-data-formats-and-pipelines.md`
12. `02-deep-learning-libraries/05-transformer-from-scratch.md`（**核心中的核心**）

### 第 3 阶段：AIGC 框架应用（约 3 周）

13. `02-deep-learning-libraries/03-huggingface-transformers.md`
14. `02-deep-learning-libraries/04-huggingface-diffusers.md`
15. `01-python-foundations/03-async-programming.md`（配合服务化场景）

### 第 4 阶段：训练工程化与分布式（约 3–4 周）

16. `04-training-engineering/01-experiment-tracking.md`
17. `04-training-engineering/02-config-management.md`
18. `04-training-engineering/03-hyperparameter-search.md`
19. `04-training-engineering/04-reproducibility.md`
20. `05-distributed-training/01-distributed-basics-and-ddp.md`
21. `05-distributed-training/02-fsdp.md`
22. `05-distributed-training/03-accelerate.md`
23. `05-distributed-training/04-deepspeed.md`
24. `05-distributed-training/05-parallelism-strategies.md`

### 第 5 阶段：微调、对齐与推理部署（约 4 周）

25. `06-finetuning-and-alignment/01-peft-and-lora.md`（**必读**）
26. `06-finetuning-and-alignment/02-quantization.md`
27. `06-finetuning-and-alignment/03-sft-data-and-training.md`
28. `06-finetuning-and-alignment/04-alignment-rlhf-dpo.md`
29. `07-inference-and-deployment/00-inference-and-deployment-theory.md`
30. `07-inference-and-deployment/01-llm-inference-engines.md`
31. `07-inference-and-deployment/02-diffusion-acceleration.md`
32. `07-inference-and-deployment/03-serving-frameworks.md`
33. `07-inference-and-deployment/04-demo-and-frontend.md`

### 第 6 阶段：LLM 应用与前沿（约 4 周）

34. `08-llm-applications/00-llm-applications-theory.md`
35. `08-llm-applications/01-rag-fundamentals.md`
36. `08-llm-applications/02-vector-databases.md`
37. `08-llm-applications/03-orchestration-frameworks.md`
38. `08-llm-applications/04-agent-engineering.md`
39. `09-frontier-models/00-frontier-models-theory.md`
40. `09-frontier-models/01-llm-architectures.md`（**核心，建议反复阅读**）
41. `09-frontier-models/02-image-generation.md`
42. `09-frontier-models/03-multimodal-models.md`
43. `09-frontier-models/04-video-generation.md`
44. `09-frontier-models/05-speech-and-audio.md`

### 第 7 阶段：工程深水区（持续学习）

45. `10-cuda-and-triton/00-gpu-performance-theory.md`
46. `10-cuda-and-triton/01-gpu-architecture-and-cuda-basics.md`
47. `10-cuda-and-triton/02-triton-programming.md`
48. `10-cuda-and-triton/03-performance-profiling.md`
49. `10-cuda-and-triton/04-custom-operators-and-extensions.md`

> 日常写代码遇到忘了的 API，直接翻 [`CHEATSHEET.md`](./CHEATSHEET.md)。

---

## 推荐的顶级开源学习资源

| 类型 | 资源 | 说明 |
|---|---|---|
| 课程 | [Karpathy: Neural Networks Zero to Hero](https://karpathy.ai/zero-to-hero.html) | 从零手撸神经网络、Transformer、Tokenizer |
| 代码 | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | ~300 行代码复现 GPT-2 |
| 代码 | [karpathy/nanochat](https://github.com/karpathy/nanochat) | 单卡可训练的完整 ChatGPT 复现（含 SFT/推理/UI） |
| 课程 | [HuggingFace Diffusion Models Course](https://github.com/huggingface/diffusion-models-class) | 官方扩散模型入门课 |
| 文档 | [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics) | 官方入门教程 |
| 文档 | [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers) | Transformers 官方文档 |
| 文档 | [HuggingFace Diffusers Docs](https://huggingface.co/docs/diffusers) | Diffusers 官方文档 |
| 文档 | [einops 教程](https://einops.rocks/) | 张量操作 DSL |
| 文档 | [vLLM Docs](https://docs.vllm.ai/) | LLM 推理引擎文档 |
| 框架 | [LangChain Docs](https://python.langchain.com/) | LLM 应用开发框架 |
| 教程 | [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/) | OpenAI Triton GPU 编程教程 |
| 论文 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Transformer 原始论文 |
| 论文 | [LoRA](https://arxiv.org/abs/2106.09685) | 参数高效微调开山之作 |
| 论文 | [DPO](https://arxiv.org/abs/2305.18290) | 直接偏好优化，替代 RLHF |

---

## 如何使用本仓库

1. 按模块 README 指引顺序阅读 `.md` 教程。
2. 每个模块的 `examples/` 目录包含可运行脚本，**边读边跑**是最好的学习方式。
3. 建议使用 `uv` 或 `conda` 创建独立虚拟环境：

```bash
cd aigc-learning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

4. 推荐配套工具：VSCode / Cursor + Jupyter 扩展；GPU 环境建议 CUDA 12.x + PyTorch 2.4+。

---

## 学习心法

- **动手 > 看书**：每一份示例代码都要亲自跑一遍，并尝试修改参数观察变化。
- **读源码**：当你对某个库好奇时，直接去 GitHub 读它的实现，这是进阶最快的方式。
- **刻意练习**：学完一章后，找一个相关的小项目从零实现一遍（不看教程）。
- **费曼学习法**：尝试把你学到的内容写成博客或讲给别人听，能讲清楚才是真懂。
