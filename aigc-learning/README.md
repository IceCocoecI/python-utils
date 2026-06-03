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
├── CHEATSHEET.md                      # 速查表：日常写代码最常用的片段
├── requirements.txt                   # 依赖清单（按模块分组）
│
├── 01-python-foundations/             # 模块 01：现代 Python 编程基础与进阶
│   ├── README.md
│   ├── 01-modern-python-basics.md     # 数据结构 / 魔法方法 / 惯用法 / itertools / 陷阱
│   ├── 02-advanced-features.md        # 装饰器 / 生成器 / 上下文管理器 / functools
│   ├── 03-async-programming.md        # asyncio / async-await / 并发模型
│   ├── 04-type-hints.md               # typing / ParamSpec / Protocol / Pydantic
│   ├── 05-engineering-best-practices.md  # 项目结构 / pytest / 调试 / profiling / ruff
│   └── examples/                      # 可运行示例
│
├── 02-deep-learning-libraries/        # 模块 02：深度学习核心库
│   ├── README.md
│   ├── 01-pytorch-fundamentals.md     # Tensor / autograd / nn.Module / device
│   ├── 02-pytorch-training-loop.md    # DataLoader / AMP / 显存预算 / 训练调试
│   ├── 03-huggingface-transformers.md # Tokenizer / AutoModel / Trainer / PEFT 入门
│   ├── 04-huggingface-diffusers.md    # Pipeline / Scheduler / 微调与 LoRA 入门
│   ├── 05-transformer-from-scratch.md # 注意力 / 位置编码 / 掩码 / KV cache / FlashAttention
│   └── examples/
│
├── 03-data-and-scientific-computing/  # 模块 03：数据处理与科学计算
│   ├── README.md
│   ├── 01-numpy-essentials.md         # ndarray / 广播 / 向量化 / 内存视图
│   ├── 02-einops-tutorial.md          # rearrange / reduce / repeat / Layers
│   ├── 03-image-processing.md         # Pillow / OpenCV / torchvision / Albumentations / FID
│   ├── 04-data-formats-and-pipelines.md  # JSONL / Parquet / WebDataset / safetensors / HF datasets
│   └── examples/
│
├── 04-training-engineering/           # 模块 04：训练与实验工程化
│   ├── README.md
│   ├── 01-experiment-tracking.md      # TensorBoard / W&B / MLflow
│   ├── 02-config-management.md        # OmegaConf / Hydra / 配置组合
│   ├── 03-hyperparameter-search.md    # Optuna / Ray Tune / 搜索策略
│   └── 04-reproducibility.md          # 随机种子 / 确定性模式 / 环境管理 / 数据版本
│
└── 05-distributed-training/           # 模块 05：分布式训练
    ├── README.md
    ├── 01-distributed-basics-and-ddp.md  # DDP / torchrun / 多机训练
    ├── 02-fsdp.md                     # FSDP / FSDP2 / 分片策略
    ├── 03-accelerate.md               # HuggingFace Accelerate / 一键分布式
    ├── 04-deepspeed.md                # DeepSpeed / ZeRO-1/2/3 / CPU Offload
    └── 05-parallelism-strategies.md   # DP / TP / PP / EP / SP / 3D 并行
```

---

## 推荐学习顺序

### 第 1 阶段：打牢基础（约 2–3 周）

1. `01-python-foundations/01-modern-python-basics.md`
2. `03-data-and-scientific-computing/01-numpy-essentials.md`
3. `01-python-foundations/02-advanced-features.md`
4. `01-python-foundations/04-type-hints.md`
5. `01-python-foundations/05-engineering-best-practices.md`（测试/调试/profiling——太多人跳过这步）

### 第 2 阶段：深度学习实战（约 3–4 周）

6. `02-deep-learning-libraries/01-pytorch-fundamentals.md`
7. `02-deep-learning-libraries/02-pytorch-training-loop.md`
8. `03-data-and-scientific-computing/02-einops-tutorial.md`
9. `03-data-and-scientific-computing/03-image-processing.md`
10. `03-data-and-scientific-computing/04-data-formats-and-pipelines.md`
11. `02-deep-learning-libraries/05-transformer-from-scratch.md`（**核心中的核心**）

### 第 3 阶段：AIGC 框架应用（约 3 周）

12. `02-deep-learning-libraries/03-huggingface-transformers.md`
13. `02-deep-learning-libraries/04-huggingface-diffusers.md`
14. `01-python-foundations/03-async-programming.md`（配合服务化场景）

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

---

## 可选扩展大纲（后续规划）

完成 01–05 模块后，建议按以下顺序深入更高阶主题。**当前仓库仅列出大纲，后续逐步展开。**

### 模块 06：大模型微调与对齐

- 参数高效微调：LoRA / QLoRA / DoRA（`peft`）
- 量化训练：`bitsandbytes` / GPTQ / AWQ
- 指令微调：SFT 数据构造与训练
- 对齐方法：RLHF / DPO / GRPO（`trl`）

### 模块 07：AIGC 推理与部署

- LLM 推理引擎：`vLLM` / `SGLang` / `TensorRT-LLM` / `llama.cpp`
- 扩散模型加速：`torch.compile` / TensorRT / ONNX
- 服务框架：`FastAPI` + `Triton Inference Server`
- 前端 Demo：`gradio` / `streamlit`

### 模块 08：LLM 应用开发

- RAG 全流程：文档切分、embedding、向量检索
- 向量数据库：`FAISS` / `Milvus` / `Chroma`
- 框架：`LangChain` / `LlamaIndex` / `LangGraph`
- Agent 工程：Tool Use / Planning / Multi-Agent

### 模块 09：前沿 AIGC 模型架构

- LLM 架构：GPT / LLaMA / Qwen / Mixtral（MoE）
- 图像生成：DDPM → DDIM → SD1.5 → SDXL → DiT → Flux
- 多模态：CLIP / BLIP / Qwen-VL / LLaVA
- 视频生成：Sora 系 / CogVideoX / Wan / HunyuanVideo
- 语音合成：VALL-E / CosyVoice / F5-TTS

### 模块 10：工程深水区

- CUDA 基础：kernel 编写、共享内存、warp-level 原语
- Triton 入门：用 Python 写 GPU kernel
- 性能剖析：`nsight systems` / `torch.profiler`
- C++/Python 混合编程：`pybind11` / `torch.utils.cpp_extension`

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
