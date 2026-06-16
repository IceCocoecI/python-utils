# Troubleshooting

> 常见问题先查这里。优先保证 toy 示例能跑，再处理真实模型、GPU、外部服务。

---

## Conda 环境

### `conda: command not found`

说明 shell 没有加载 conda。先确认 conda 安装路径，再执行初始化：

```bash
conda init bash
```

重新打开终端后再试：

```bash
conda activate aigc
```

### `EnvironmentNameNotFound: aigc`

说明环境不存在。可以新建：

```bash
conda create -n aigc python=3.11 -y
conda activate aigc
pip install -r aigc-learning/requirements.txt
```

PyTorch 建议按你的 CUDA 版本单独安装，参考 [requirements.txt](./requirements.txt) 顶部说明。

---

## PyTorch / CUDA

### `torch.cuda.is_available()` 是 `False`

先确认：

```bash
nvidia-smi
conda run -n aigc python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

常见原因：

- 机器没有 NVIDIA GPU。
- 驱动不可用或版本过旧。
- 安装了 CPU 版 PyTorch。
- CUDA wheel 与驱动不兼容。

CPU 学习路线仍然可以继续，优先跑 toy 示例。

### CUDA OOM

处理顺序：

1. 降低 batch size。
2. 降低 sequence length / `max_model_len`。
3. 使用 bf16/fp16。
4. 开启 gradient checkpointing。
5. 使用 LoRA/QLoRA。
6. 使用更小模型。
7. 多卡分片或换更大显存。

---

## HuggingFace 下载

### 模型下载失败

常见原因是网络、代理、权限或模型需要登录。先跑默认离线 toy 示例，确认代码路径没问题：

```bash
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/transformers_quickstart.py
```

如果必须下载真实模型：

```bash
huggingface-cli login
```

也可以设置缓存目录：

```bash
export HF_HOME=/path/to/hf-cache
```

### `pad_token` 未设置

LLM tokenizer 常见问题。通常可以：

```python
tokenizer.pad_token = tokenizer.eos_token
```

训练前要确认 chat template、padding side、label mask 是否符合任务。

---

## Transformers / Device

### `device_map="auto"` 和 `.to(device)` 冲突

使用 `device_map="auto"` 时，Accelerate 已经管理模型分布。不要再手动：

```python
model.to("cuda")
```

推荐二选一：

```python
AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
```

或：

```python
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
```

---

## FastAPI / 服务

### 端口被占用

换端口：

```bash
uvicorn app:app --host 127.0.0.1 --port 8001
```

或者查占用：

```bash
lsof -i :8000
```

### OpenAI-compatible 客户端请求失败

检查：

- `base_url` 是否以 `/v1` 结尾。
- API key 是否传了占位字符串。
- 服务是否实现了 `/v1/chat/completions`。
- 请求字段是否包含 `model` 和 `messages`。

---

## RAG

### 检索结果不相关

优先检查：

1. chunk 是否太短或太长。
2. metadata 是否丢失。
3. embedding 是否归一化。
4. 查询和文档语言是否一致。
5. top-k 是否太小。
6. 是否需要 reranker。

toy RAG 只用于理解流程，真实效果需要真实 embedding 和更严格评估。

---

## Profiling

### profiler 输出看不懂

先只看三类信息：

- 总耗时最多的 op。
- CPU time 和 CUDA time 谁更高。
- 是否存在频繁同步或数据搬运。

CPU-only 环境没有 CUDA trace 是正常的。

### `torch.profiler` 生成 trace 后不知道怎么打开

如果脚本输出 trace 到 `outputs/profiler_demo`，可以用 TensorBoard：

```bash
tensorboard --logdir aigc-learning/10-cuda-and-triton/examples/outputs/profiler_demo
```

