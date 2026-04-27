# 04 · 数据格式与大规模数据管线

> AIGC 训练的一大瓶颈往往**不是模型，而是数据管线**。
> 100GB 文本、1TB 图像、10TB 视频……你不能再用 `json.load()` 一口气读进内存。
> 这一节告诉你在不同规模下该用什么格式。

---

## 1. 格式选型一览

| 数据规模 | 数据类型 | 推荐格式 | 代表库 |
|---|---|---|---|
| < 100 MB | 结构化 | JSON / CSV | stdlib / pandas |
| 100 MB – 10 GB | 文本（LLM 语料） | **JSONL** | stdlib / `datasets` |
| 1 GB – 1 TB | 表格 / 特征 | **Parquet** | pyarrow / pandas |
| 1 GB – 10 GB | Tensor / 模型权重 | **safetensors** | safetensors |
| 10 GB – 100 GB | 数组 / 张量批量 | HDF5 / Zarr | h5py / zarr |
| 100 GB – 10 TB | 图像 / 多模态 | **WebDataset** (.tar) | webdataset |
| 10 TB+ | 任意 | HF `datasets` streaming | datasets |

**一句话**：文本 → JSONL；表格 → Parquet；张量 → safetensors；图像语料 → WebDataset。

---

## 2. JSON / JSONL：LLM 训练语料的标配

### 2.1 JSON vs JSONL 的关键区别

```json
[{"text": "..."}, {"text": "..."}]
```

`JSON Lines`（每行一个 JSON 对象）：

```
{"text": "..."}
{"text": "..."}
{"text": "..."}
```

**为什么 JSONL 是 LLM 语料的事实标准？**
- **流式读取**：100GB 文件不用全加载到内存，逐行读。
- **可追加**：新样本直接 `echo >> data.jsonl`。
- **分布式切片**：按行数平均分配给不同 worker。
- **生态统一**：HuggingFace `datasets`、`trl`、OpenAI SDK 都吃 JSONL。

### 2.2 读写示例

```python
import json
from pathlib import Path


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    """流式读取：不会一次性把整个文件读进内存。"""
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


for sample in read_jsonl(Path("corpus.jsonl")):
    process(sample)
```

### 2.3 LLM 训练典型 JSONL schema

**预训练语料**：
```json
{"text": "The quick brown fox jumps over the lazy dog. ..."}
```

**SFT 指令数据**：
```json
{"messages": [
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Write a poem."},
  {"role": "assistant", "content": "Roses are red..."}
]}
```

**DPO 偏好数据**：
```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

**多模态**：
```json
{"image": "images/001.jpg", "caption": "A cat on the mat."}
```

### 2.4 用 HF `datasets` 加载

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="corpus.jsonl", split="train")
print(ds[0])

ds = load_dataset("json", data_files="corpus/*.jsonl", split="train")

ds = load_dataset("json", data_files="corpus/*.jsonl",
                  split="train", streaming=True)
for sample in ds.take(5):
    print(sample)
```

---

## 3. Parquet：表格数据的王者

### 3.1 Parquet 为什么快？

- **列式存储**：只读你要的列，不用读整行。
- **高压缩比**：snappy / zstd 压缩，比 CSV 小 5–10×。
- **schema 自描述**：dtype 内嵌，不用你猜。
- **分区 / 统计信息**：支持谓词下推（filter 推到存储层）。

### 3.2 Pandas + PyArrow

```python
import pandas as pd

df = pd.DataFrame({
    "id": range(10000),
    "text": ["sample " * 10] * 10000,
    "length": [100] * 10000,
})
df.to_parquet("data.parquet", compression="zstd")

df2 = pd.read_parquet("data.parquet")

df3 = pd.read_parquet("data.parquet", columns=["id", "length"])

df4 = pd.read_parquet("data.parquet", filters=[("length", ">", 50)])
```

### 3.3 PyArrow 直接操作（大文件推荐）

```python
import pyarrow.parquet as pq

parquet_file = pq.ParquetFile("huge.parquet")
for batch in parquet_file.iter_batches(batch_size=10000):
    df = batch.to_pandas()
    process(df)
```

### 3.4 HF `datasets` 天然支持

```python
from datasets import load_dataset
ds = load_dataset("parquet", data_files="data.parquet", split="train")
```

事实上，HF Hub 上的大部分数据集**底层就是 Parquet**。

---

## 4. safetensors：模型权重的现代标准

### 4.1 为什么不用 `torch.save`？

`torch.save` 用 Python pickle——**反序列化时会执行任意代码**，有严重安全隐患。
`safetensors` 是纯数据格式，不含代码，加载速度还快。

```python
from safetensors.torch import save_file, load_file

state_dict = model.state_dict()
save_file(state_dict, "model.safetensors")

sd = load_file("model.safetensors", device="cuda")
model.load_state_dict(sd)

from safetensors import safe_open
with safe_open("model.safetensors", framework="pt") as f:
    print(f.keys())
    for key in f.keys():
        tensor = f.get_tensor(key)
```

### 4.2 HuggingFace 全面拥抱

`transformers` / `diffusers` 保存模型默认就是 `.safetensors`：

```python
model.save_pretrained("./my-model", safe_serialization=True)
```

生产环境**必须用 safetensors**，别用 .bin / .pt。

### 4.3 Lazy Loading（大模型必备）

```python
with safe_open("model.safetensors", framework="pt", device="cuda") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        if "qkv" in key:
            process(tensor)
```

加载 70B 模型时，peak memory 几乎是 0——因为只有被请求的张量才真正读出来。

---

## 5. HDF5：大规模数值数据

### 5.1 何时用？

- 已有的数组型数据（特征向量、预计算的 embedding、科学计算输出）
- 需要**按索引快速切片**访问
- 传统机器学习管线的历史遗留

### 5.2 `h5py` 用法

```python
import h5py
import numpy as np

with h5py.File("embeddings.h5", "w") as f:
    f.create_dataset("features", data=np.random.randn(100_000, 768), compression="gzip")
    f.create_dataset("labels", data=np.random.randint(0, 10, size=100_000))
    f.attrs["model"] = "bge-small-en"

with h5py.File("embeddings.h5", "r") as f:
    print(f.attrs["model"])
    features = f["features"][1000:1100]
    labels = f["labels"][:]
```

**提示**：AIGC 时代 HDF5 用得越来越少，Parquet + Arrow 已经吃掉了大部分场景。但如果同事丢给你一个 `.h5` 文件，至少要知道怎么打开。

---

## 6. WebDataset：海量图像 / 多模态数据集

### 6.1 为什么需要？

训练 Stable Diffusion / CLIP 时，数据集动辄几十亿张图。
此时**小文件数**本身就是瓶颈（inode、随机 I/O）。

**WebDataset** 的思路：把数据打包成大 `.tar` 文件，按序列读。

```
shards/
├── data-000000.tar  (包含 10000 张图 + 对应 json)
├── data-000001.tar
├── data-000002.tar
└── ...
```

### 6.2 典型使用

```python
import webdataset as wds
from torch.utils.data import DataLoader

def decode(sample):
    return {
        "image": sample["jpg"],
        "caption": sample["json"]["caption"],
    }

ds = (
    wds.WebDataset("shards/data-{000000..000099}.tar", shardshuffle=True)
    .shuffle(1000)
    .decode("pil")
    .map(decode)
    .to_tuple("image", "caption")
)

loader = DataLoader(ds, batch_size=32, num_workers=8)
for images, captions in loader:
    ...
```

### 6.3 实战情况

- **LAION-5B**（50 亿图文对，Stable Diffusion 训练集）分发的就是 WebDataset 格式。
- 分布式场景下按 shard 分配给 worker，天然切分。
- 缺点：随机访问差——**只适合纯顺序训练**。

---

## 7. HuggingFace `datasets`：一库通吃

### 7.1 核心能力

- 原生支持 JSON / CSV / Parquet / WebDataset / Hub 上 10 万+ 数据集
- 底层是 **Apache Arrow**，内存零拷贝
- 支持 **streaming** 模式：TB 数据秒开
- 和 `transformers.Trainer`、`trl` 无缝对接

### 7.2 高频用法

```python
from datasets import load_dataset, Dataset, concatenate_datasets

ds = load_dataset("allenai/c4", "en", split="train[:1%]")

ds = Dataset.from_dict({"text": ["hi", "world"], "label": [0, 1]})

ds = ds.map(lambda x: {"length": len(x["text"])})

ds = ds.map(lambda batch: {"tokens": tok(batch["text"])["input_ids"]},
            batched=True, num_proc=8)

ds_long = ds.filter(lambda x: x["length"] > 100)

splits = ds.train_test_split(test_size=0.1, seed=42)
train, test = splits["train"], splits["test"]

ds_big = load_dataset("allenai/c4", "en", split="train", streaming=True)
for i, sample in enumerate(ds_big):
    if i >= 5:
        break
    print(sample["text"][:50])

ds.save_to_disk("./my-ds")
ds2 = Dataset.load_from_disk("./my-ds")
```

### 7.3 性能神器：Apache Arrow 零拷贝

`datasets` 底层把数据存成 Arrow 表格（列式、mmap）：
- 首次 `load` 后 cache 到磁盘（`~/.cache/huggingface/`）。
- 再次 load 是 **mmap**，不占实际内存。
- `map` 可以持久化结果，下次跳过。

---

## 8. 选型决策树

```
数据是什么？
├── 结构化表格（DataFrame） → Parquet
├── 文本语料 / 对话
│   ├── 小（< 10 GB）         → JSONL
│   └── 大（> 10 GB）         → HF datasets + JSONL 分片
├── 图像 / 多模态语料
│   ├── 小数据集（< 1 GB）    → 文件夹 + PIL
│   ├── 中（1–100 GB）        → HF datasets Parquet
│   └── 大（> 100 GB）        → WebDataset (.tar)
├── 模型权重                  → safetensors
├── 预计算特征 / embedding    → Parquet 或 HDF5
└── 视频                      → .mp4 + 索引 (decord)
```

---

## 9. 数据管线性能优化实战

### 9.1 瓶颈定位

```python
import time

t0 = time.perf_counter()
for i, batch in enumerate(loader):
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()

    loss = train_step(batch)
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()

    if i % 10 == 0:
        ratio = t_train / (t_load + t_train)
        print(f"step {i} load={t_load*1000:.0f}ms train={t_train*1000:.0f}ms "
              f"gpu_util≈{ratio:.0%}")
```

`gpu_util` < 70% 说明数据管线是瓶颈。

### 9.2 调优手段（优先级从高到低）

1. **num_workers**：调到 CPU 核数的一半到全部。
2. **pin_memory=True** + `.to(device, non_blocking=True)`。
3. **persistent_workers=True**：避免每 epoch 重启 worker。
4. **prefetch_factor**：每个 worker 提前准备多少 batch（默认 2，可加到 4）。
5. **离线预处理**：把 tokenize / decode 结果保存成 Parquet，训练时不做 CPU 密集操作。
6. **数据增强放 GPU**：`kornia` 而不是 CPU 上的 torchvision。

### 9.3 `DataLoader` 最优配置模板

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=min(8, os.cpu_count() // 2),
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    drop_last=True,
)
```

---

## 10. 常见坑

### 10.1 编码问题

所有文本文件处理都加 `encoding="utf-8"`——默认编码在 Windows/Linux 不同，跨平台必踩。

### 10.2 大 JSONL 每行不能太长

```python
import json, orjson

samples = list(json.loads(line) for line in f)
samples = list(orjson.loads(line) for line in f)
```

### 10.3 pandas 读 CSV 时内存爆炸

`CSV` 没 schema，pandas 会推断——`int64` / `object` 占用巨大。
**要么转 Parquet，要么显式 `dtype=` 指定**：

```python
df = pd.read_csv("big.csv", dtype={"id": "int32", "score": "float32"})
```

### 10.4 WebDataset 的 shard 命名要连续

`data-{000000..000099}.tar` 必须存在对应文件，缺一个就报错。

---

## 小结

| 场景 | 推荐 | 为什么 |
|---|---|---|
| LLM 预训练 / SFT | JSONL + HF datasets | 生态统一、流式读取 |
| 多模态 / 图像大语料 | WebDataset | 百亿规模 I/O 优化 |
| 表格 / 特征 | Parquet | 列式、高压缩、谓词下推 |
| 模型权重 | safetensors | 安全、快、HF 默认 |
| 预计算特征 | Parquet > HDF5 | Arrow 生态更现代 |

**一条黄金规则**：**数据管线先跑通一个小规模 demo，再扩大**。不要一上来就想 TB 级——99% 的 bug 在小规模就能暴露。

下一步可以动手：找一个 HF Hub 上的公开数据集（如 `HuggingFaceH4/ultrachat_200k`），用 JSONL / Parquet 两种方式加载，体会差异。
