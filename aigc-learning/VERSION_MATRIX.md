# Version Matrix

> 这个矩阵不是强锁定版本，而是学习环境的兼容建议。
> 如果你需要生产环境，请用项目级 lockfile 或容器镜像固定版本。

---

## 当前推荐基线

| 组件 | 建议版本 | 说明 |
|---|---|---|
| Python | 3.10 - 3.12 | 示例代码使用现代类型语法，推荐 3.11 |
| PyTorch | 2.4+ | `torch.compile`、profiler、现代 AMP API 更稳定 |
| CUDA | 12.x | 真实 GPU 训练/推理建议 CUDA 12 系 |
| NumPy | 1.26+ | 与当前科学计算生态兼容 |
| Transformers | 4.44+ | 覆盖现代 LLM API 和 chat template |
| Diffusers | 0.30+ | 覆盖现代 pipeline 和 scheduler |
| Accelerate | 0.33+ | HF 分布式和 device map 底座 |
| Datasets | 2.20+ | 数据处理示例使用常规 API |
| PEFT | 0.12+ | LoRA/QLoRA 示例基础 |
| FastAPI | 0.115+ | 推理服务示例 |
| OpenAI SDK | 1.40+ | OpenAI-compatible 客户端 |
| Gradio | 5.x | Demo 应用 |

---

## PyTorch 安装建议

CPU 版：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

CUDA 12.4 示例：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

安装后检查：

```bash
conda run -n aigc python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

---

## 推理引擎兼容提示

| 引擎 | 安装建议 | 风险点 |
|---|---|---|
| vLLM | 按官方文档和 CUDA 版本安装 | 与 PyTorch/CUDA/驱动强绑定 |
| SGLang | 按官方文档安装 `sglang[all]` | 依赖较多，版本组合敏感 |
| TensorRT-LLM | 优先 NVIDIA 官方容器 | 本地编译和驱动要求高 |
| llama.cpp / llama-cpp-python | 按 CPU/GPU 后端分别安装 | Metal/CUDA/OpenBLAS 后端差异 |

默认 `requirements.txt` 不安装这些引擎，避免破坏基础学习环境。

---

## 常见组合

### CPU 学习组合

```text
Python 3.11
PyTorch CPU wheel
NumPy / pandas / transformers / datasets
```

适合：

- `01` 到 `04` 的大部分内容。
- `06` toy 示例。
- `07` toy server。
- `08` toy RAG。
- `10` CPU profiler demo。

### 单卡 NVIDIA 学习组合

```text
Python 3.11
NVIDIA Driver compatible with CUDA 12.x
PyTorch CUDA 12.x wheel
Transformers / Accelerate / PEFT
```

适合：

- 真实模型推理。
- 小规模 LoRA/QLoRA。
- GPU profiler。
- 部分 Triton 示例。

### 推理服务组合

```text
Python 3.10/3.11
CUDA 12.x
PyTorch version required by vLLM/SGLang
vLLM or SGLang pinned by official docs
OpenAI SDK pinned in client app
```

适合：

- OpenAI-compatible 本地服务。
- 高并发压测。
- 多卡推理。

---

## 升级检查清单

- [ ] `conda run -n aigc python aigc-learning/scripts/check_links.py`
- [ ] `conda run -n aigc python aigc-learning/08-llm-applications/examples/toy_rag.py --self-test`
- [ ] `conda run -n aigc python aigc-learning/10-cuda-and-triton/examples/torch_profiler_demo.py --steps 3`
- [ ] `conda run -n aigc python aigc-learning/07-inference-and-deployment/examples/smoke_test.py`

