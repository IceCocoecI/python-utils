# 环境与运行能力矩阵

> 先判断“当前机器能跑什么”，再决定学到什么深度。
> 默认环境名为 `aigc`。

---

## 快速检查

```bash
conda run -n aigc python --version
conda run -n aigc python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
conda run -n aigc python -c "import numpy, transformers; print('basic deps ok')"
```

如果已经激活环境：

```bash
conda activate aigc
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 能力矩阵

| 内容 | CPU | NVIDIA GPU | 网络 | API Key | 说明 |
|---|---:|---:|---:|---:|---|
| `01-python-foundations` | 可 | 不需要 | 不需要 | 不需要 | 全部基础示例可 CPU 跑 |
| `02-deep-learning-libraries` toy 示例 | 可 | 可选 | 不需要 | 不需要 | `transformers_quickstart.py` 默认离线小模型 |
| `02` 真实 HF 模型 | 可但慢 | 推荐 | 需要 | 不需要 | 首次运行需要下载模型 |
| `03-data-and-scientific-computing` | 可 | 不需要 | 不需要 | 不需要 | 图像和数据格式示例本地合成 |
| `04-training-engineering` | 可 | 可选 | 不需要 | W&B 可选 | 默认不依赖外部服务 |
| `05-distributed-training` CPU demos | 可 | 不需要 | 不需要 | 不需要 | 内存数学和通信概念可 CPU 跑 |
| `05` 真实多卡训练 | 不适合 | 需要 | 不需要 | 不需要 | 需要多 GPU 或多机环境 |
| `06-finetuning-and-alignment` toy 示例 | 可 | 可选 | 不需要 | 不需要 | tiny LoRA、DPO loss、量化模拟可 CPU 跑 |
| `06` 真实 LLM 微调 | 不推荐 | 需要 | 通常需要 | 不需要 | 7B 级模型建议大显存 GPU |
| `07-inference-and-deployment` toy 服务 | 可 | 可选 | 不需要 | 不需要 | OpenAI-compatible toy server 可本地跑 |
| `07` vLLM/SGLang | 不适合 | 需要 | 模型下载需要 | 不需要 | 依赖 CUDA、驱动和模型权重 |
| `08-llm-applications` toy RAG | 可 | 不需要 | 不需要 | 不需要 | `toy_rag.py` 使用本地文档和哈希 embedding |
| `08` 真实 RAG | 可 | 可选 | 通常需要 | 可选 | 取决于 embedding/LLM 来源 |
| `09-frontier-models` | 可 | 不需要 | 不需要 | 不需要 | 主要是阅读和架构理解 |
| `10-cuda-and-triton` profiling toy | 可 | 推荐 | 不需要 | 不需要 | CPU 可跑，GPU 下信息更完整 |
| `10` Triton/CUDA kernel | 不适合 | 需要 | 不需要 | 不需要 | 需要 NVIDIA GPU 和匹配的软件栈 |

---

## 推荐环境层级

### Level 0：CPU 学习环境

适合：

- Python、NumPy、PyTorch 基础。
- 小模型训练。
- toy RAG。
- toy serving。
- 理论学习和自检。

命令：

```bash
conda run -n aigc python aigc-learning/08-llm-applications/examples/toy_rag.py --self-test
conda run -n aigc python aigc-learning/10-cuda-and-triton/examples/torch_profiler_demo.py --steps 3
```

### Level 1：单卡 GPU 环境

适合：

- AMP、显存观察、真实推理。
- 小规模 LoRA/QLoRA。
- profiling 和基础性能优化。

检查：

```bash
nvidia-smi
conda run -n aigc python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Level 2：多卡 / 多机环境

适合：

- DDP、FSDP、DeepSpeed。
- vLLM/SGLang 张量并行。
- 大模型训练和高并发服务。

检查：

```bash
conda run -n aigc python -m torch.distributed.run --help
```

---

## 依赖原则

- `requirements.txt` 不固定 `torch`，因为 PyTorch 版本必须匹配 CUDA/驱动。
- 推理引擎如 vLLM、SGLang、TensorRT-LLM 不放进默认依赖，按 GPU 环境单独安装。
- 新增示例默认使用合成数据和本地文件，避免学习过程被网络和账号阻塞。

