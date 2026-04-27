# AIGC 工程师速查表（Cheatsheet）

> 不教概念，只给代码片段——**写代码时想不起来就查这里**。
> 配合三大模块系统学习后使用效果更佳。

---

## 1. Python 必背片段

### 1.1 模板 / 配置

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    lr: float = 1e-4
    batch_size: int = 32
    output_dir: Path = Path("./out")
    tags: list[str] = field(default_factory=list)
```

### 1.2 日志

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("step=%d loss=%.4f", step, loss)
```

### 1.3 可复现随机

```python
import random, os, numpy as np, torch
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)
```

### 1.4 计时装饰器

```python
import functools, time
def timer(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        t = time.perf_counter()
        out = fn(*a, **kw)
        print(f"{fn.__name__}: {time.perf_counter()-t:.3f}s")
        return out
    return wrapper
```

### 1.5 retry

```python
def retry(times=3, delay=1.0):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            for i in range(times):
                try: return fn(*a, **kw)
                except Exception as e:
                    if i == times - 1: raise
                    time.sleep(delay * 2**i)
        return wrapper
    return deco
```

---

## 2. NumPy / einops

```python
x.astype(np.float32)
x.mean(axis=0)
x / np.linalg.norm(x, axis=-1, keepdims=True)
np.argsort(-scores)[:k]
np.random.default_rng(42).normal(size=(3, 4))

from einops import rearrange, reduce, repeat
rearrange(x, "b c h w -> b h w c")
rearrange(x, "b (h p1) (w p2) c -> b (h w) (p1 p2 c)", p1=16, p2=16)
rearrange(qkv, "b t (three h d) -> three b h t d", three=3, h=8)
reduce(x, "b c h w -> b c", "mean")
repeat(mask, "b t -> b h t", h=8)
```

---

## 3. PyTorch

### 3.1 Tensor 基础

```python
x = torch.zeros(3, 4, device="cuda", dtype=torch.bfloat16)
x.shape, x.dtype, x.device

x.view(2, 6)
x.reshape(-1, 4)
x.permute(1, 0)
x.unsqueeze(0).squeeze(0)
x.expand(2, 3, 4)
x.contiguous()

x.masked_fill_(mask == 0, float("-inf"))
x.scatter_(dim=1, index=idx, src=val)
x.gather(dim=1, index=idx)
torch.topk(x, k=5, dim=-1)
```

### 3.2 模型

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("pe", positional_encoding(max_len=4096))
        self.proj = nn.Linear(768, 768)
    def forward(self, x):
        return self.proj(x + self.pe[:x.size(1)])

sum(p.numel() for p in model.parameters()) / 1e6
sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

for name, m in model.named_modules():
    if isinstance(m, nn.Linear): print(name, m.weight.shape)
```

### 3.3 训练骨架

```python
scaler = torch.amp.GradScaler("cuda")
optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)

for batch in loader:
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss
    optim.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optim); scaler.update(); sched.step()
```

### 3.4 推理 / 生成

```python
@torch.inference_mode()
def generate(model, ids, max_new=100):
    for _ in range(max_new):
        logits = model(ids)[:, -1]
        next_id = logits.argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=-1)
    return ids

torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "step": step},
           "ckpt.pt")
from safetensors.torch import save_file, load_file
save_file(model.state_dict(), "model.safetensors")
```

### 3.5 显存

```python
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated() / 1024**3, "GB")
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated() / 1024**3
```

---

## 4. HuggingFace

### 4.1 Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
)

msgs = [{"role": "user", "content": "hello"}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=200, do_sample=True,
                     temperature=0.7, top_p=0.9)
print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

### 4.2 Datasets

```python
from datasets import load_dataset
ds = load_dataset("json", data_files="train.jsonl", split="train")
ds = ds.map(lambda x: tok(x["text"], truncation=True), batched=True, num_proc=8)
ds = ds.filter(lambda x: len(x["input_ids"]) > 10)
ds.save_to_disk("./ds_tokenized")
```

### 4.3 LoRA (PEFT)

```python
from peft import LoraConfig, get_peft_model

cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                 target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
```

### 4.4 Diffusers

```python
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

image = pipe("a cat astronaut on mars", num_inference_steps=25,
             guidance_scale=7.5).images[0]
image.save("out.png")
```

---

## 5. 数据格式

```python
import json
with open("out.jsonl", "w") as f:
    for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")

import pandas as pd
df.to_parquet("data.parquet", compression="zstd")
pd.read_parquet("data.parquet", columns=["id", "text"])

from safetensors.torch import save_file, load_file
save_file(state_dict, "model.safetensors")
sd = load_file("model.safetensors", device="cuda")
```

---

## 6. 图像

```python
from PIL import Image
img = Image.open("x.jpg").convert("RGB").resize((512, 512))
img.save("out.png")

import cv2
bgr = cv2.imread("x.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

import torchvision.transforms.v2 as T
train_tf = T.Compose([
    T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
    T.ToImage(), T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

## 7. 异步 LLM API

```python
import asyncio, aiohttp

async def call(session, prompt):
    async with session.post(URL, json={"prompt": prompt}) as r:
        return await r.json()

async def main(prompts):
    sem = asyncio.Semaphore(10)
    async with aiohttp.ClientSession() as s:
        async def _one(p):
            async with sem:
                return await call(s, p)
        return await asyncio.gather(*(_one(p) for p in prompts))

results = asyncio.run(main(prompts))
```

---

## 8. 常用命令

```bash
nvidia-smi
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1
watch -n 1 nvidia-smi
nvtop

ps -ef | grep python
kill -9 <pid>

du -sh ./*
df -h
ncdu .

ssh -L 6006:localhost:6006 user@server
tensorboard --logdir ./runs

du -h model.safetensors

ruff check --fix . && ruff format .
pytest -xvs tests/
mypy src/

uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip compile pyproject.toml -o requirements.lock.txt

git log --oneline --all --graph -20
git blame -L 100,150 file.py
```

---

## 9. 常用 HuggingFace 模型参考（按 2026 年常用度）

### LLM

| 模型 | 代表尺寸 | 特点 |
|---|---|---|
| Qwen2.5 | 0.5B~72B | 中英双语、指令强、可商用 |
| Llama-3.x | 8B/70B/405B | 开源标杆 |
| Mistral / Mixtral | 7B/8x7B/8x22B | MoE 架构 |
| DeepSeek-V2/V3 | MoE | 推理+代码强 |

### 文生图

| 模型 | 特点 |
|---|---|
| SDXL | 社区生态最广 |
| SD3 | 文本渲染强 |
| FLUX.1 [dev] | 2024 后开源 SOTA |

### Embedding

| 模型 | 维度 | 用途 |
|---|---|---|
| bge-m3 | 1024 | 多语言、长文本 |
| text-embedding-3-large (OpenAI) | 3072 | 商用 API |
| jina-embeddings-v3 | 可变 | 开源生产级 |

---

## 10. 调试速查

| 症状 | 第一步看 |
|---|---|
| CUDA OOM | 减 batch / 开 gradient checkpointing / 量化 |
| Loss NaN | `/√d_k`、log(0+eps)、学习率、输入 check |
| Loss 不降 | 1 batch overfit 测试、检查 label |
| GPU 利用率低 | `num_workers` / `pin_memory` / prefetch |
| 训练慢 | `torch.compile` / `bf16` / 检查 `.item()` 同步 |
| device mismatch | `.to(device)` 漏了 |
| size mismatch | 打印 `(shape, dtype)`，通常 reshape/permute 写错 |
| BGR/RGB 颜色怪 | OpenCV 默认 BGR，要 `cvtColor` |

---

**打印这一页贴在显示器旁。**（开玩笑，但你会发现随手翻看的频率很高。）
