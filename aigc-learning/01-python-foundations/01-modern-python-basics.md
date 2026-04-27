# 01 · 现代 Python 基础

> 本节假设你已经会写基础 Python（变量、if/for、函数）。
> 目标：升级到"Pythonic"写法——用现代、优雅、可维护的风格完成同样的任务。

---

## 1. 数据结构精要

### 1.1 列表推导与字典推导

```python
# 不要写
squares = []
for x in range(10):
    squares.append(x * x)

# 要写
squares = [x * x for x in range(10)]

# 字典推导
name2id = {name: idx for idx, name in enumerate(["alice", "bob", "carol"])}

# 集合推导
unique_lens = {len(w) for w in ["hi", "hello", "hey"]}
```

在 AIGC 代码里大量用来构建 `tokenizer` 的 vocab、数据集的索引映射等场景。

### 1.2 `collections` 必会三件套

```python
from collections import Counter, defaultdict, deque

# 统计 token 频率
freq = Counter(tokens)
top10 = freq.most_common(10)

# 默认值字典（避免 KeyError）
index = defaultdict(list)
for i, word in enumerate(corpus):
    index[word].append(i)

# 双端队列：做 KV cache 或滑动窗口时常用
window = deque(maxlen=512)
```

### 1.3 `dataclass`：配置类的标准写法

```python
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    model_name: str = "gpt2"
    lr: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    dropout: float = 0.1
    # 可变默认值必须用 field(default_factory=...)
    tags: list[str] = field(default_factory=list)

cfg = TrainingConfig(lr=3e-4, tags=["exp01", "baseline"])
print(cfg)
```

**何时用 `dataclass` vs `pydantic.BaseModel`？**
- `dataclass`：内部配置、无需数据验证
- `pydantic.BaseModel`：需要类型校验、JSON 序列化（API 入参/配置文件）

---

## 2. 文件与路径：用 `pathlib`，别再用 `os.path`

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent
data_dir = ROOT / "data"
data_dir.mkdir(parents=True, exist_ok=True)

for img_path in data_dir.glob("**/*.jpg"):
    print(img_path.stem, img_path.suffix, img_path.stat().st_size)

text = (data_dir / "prompts.txt").read_text(encoding="utf-8")
(data_dir / "out.json").write_text('{"ok": true}', encoding="utf-8")
```

优势：跨平台、链式 API、面向对象。

---

## 3. 字符串：拥抱 f-string

```python
name, acc = "llama-3-8b", 0.847

print(f"model={name}, acc={acc:.4f}")

print(f"{acc=:.2%}")

tensor_shape = (2, 3, 4)
print(f"{tensor_shape=}")
```

日志里写 `logger.info(f"epoch={epoch} loss={loss:.4f}")` 是算法代码里的标准姿势。

---

## 4. 日志：别再用 `print`

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

logger.info("loaded %d samples", len(dataset))
logger.warning("learning rate too large: %.2e", lr)
```

**原则**：脚本用 `logging.basicConfig` 一次性配置；库代码里只用 `logger = logging.getLogger(__name__)` 获取 logger，不要自己配置 handler。

---

## 5. 异常处理：精准捕获，不要"裸 except"

```python
try:
    model = load_checkpoint(ckpt_path)
except FileNotFoundError:
    logger.error("checkpoint missing: %s", ckpt_path)
    raise
except RuntimeError as e:
    logger.exception("checkpoint corrupted")
    raise RuntimeError(f"failed to load {ckpt_path}") from e
```

要点：
- 不要 `except:` 或 `except Exception:` 吞掉所有异常。
- 想保留原始堆栈用 `raise ... from e`。
- `logger.exception(...)` 会自动带上 traceback。

---

## 6. 面向对象精要

AIGC 代码里 OOP 无处不在（`nn.Module`、`Dataset`、`Pipeline`）。
重点掌握三个特性：

### 6.1 `__init__` / `__call__` / `__repr__`

```python
class Prompter:
    def __init__(self, template: str) -> None:
        self.template = template

    def __call__(self, text: str) -> str:
        return self.template.format(text=text)

    def __repr__(self) -> str:
        return f"Prompter(template={self.template!r})"

p = Prompter("请总结：{text}")
print(p("这是一段文本"))
print(p)
```

这就是为什么 `nn.Module` 的实例可以直接 `model(x)` 调用——因为它实现了 `__call__`。

### 6.2 `@property`：属性的受控访问

```python
class Dataset:
    def __init__(self, items: list[int]) -> None:
        self._items = items

    @property
    def size(self) -> int:
        return len(self._items)

d = Dataset([1, 2, 3])
print(d.size)
```

### 6.3 `classmethod` / `staticmethod`

`classmethod` 最经典的场景是 `from_pretrained`：

```python
class Model:
    def __init__(self, weights: dict) -> None:
        self.weights = weights

    @classmethod
    def from_pretrained(cls, name: str) -> "Model":
        weights = _download_weights(name)
        return cls(weights)
```

这就是 HuggingFace 所有模型都有 `Model.from_pretrained("...")` 的原理。

### 6.4 魔法方法（Dunder Methods）：理解 PyTorch Dataset / Module 的关键

Python 很多"看起来像语法糖"的特性，背后是某个 `__xxx__` 方法——记住这几个，PyTorch 和 HuggingFace 的 API 瞬间变透明。

| 方法 | 触发时机 | AIGC 场景 |
|---|---|---|
| `__init__` | `MyClass(...)` | 初始化 |
| `__call__` | `obj(x)` | `model(x)` 能调用的原因 |
| `__len__` | `len(obj)` | `Dataset` 必须实现 |
| `__getitem__` | `obj[i]`、`obj["key"]` | `Dataset[i]`、`dict`-like API |
| `__iter__` + `__next__` | `for x in obj` | `DataLoader` 内部协议 |
| `__contains__` | `x in obj` | 自定义 vocabulary |
| `__repr__` | `repr(obj)` / REPL 显示 | 调试友好 |
| `__eq__` / `__hash__` | `a == b` / 作为 dict key | 比较、缓存 |
| `__enter__` / `__exit__` | `with obj:` | 上下文管理器 |

**最小 Dataset 实现**（这就是 PyTorch `Dataset` 抽象的本质）：

```python
class MyDataset:
    def __init__(self, data: list):
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        return {"x": self._data[idx], "label": idx % 2}

ds = MyDataset([10, 20, 30, 40])
print(len(ds))
print(ds[2])
for sample in ds:
    print(sample)
```

**你看**：`for sample in ds` 没有显式 `__iter__`——Python 只要看到有 `__len__` + `__getitem__`，就能自动迭代。这就是"鸭子类型"。

---

## 7. 必学的 Python 惯用法（Idioms）

### 7.1 `enumerate` / `zip` / `reversed`

```python
for i, x in enumerate(batch):
    ...

for prompt, response in zip(prompts, responses, strict=True):
    ...

for token in reversed(sequence):
    ...
```

**注意**：Python 3.10+ 推荐给 `zip` 加 `strict=True`，两边长度不一致时立即报错（常见的训练 bug）。

### 7.2 解包（unpacking）

```python
first, *rest = [1, 2, 3, 4]

a, b = b, a

*head, last = sequence
head, *middle, tail = sequence

merged = {**defaults, **overrides}
merged = defaults | overrides
```

AIGC 里常用 `**kwargs` 转发参数：

```python
def make_dataloader(**kwargs):
    defaults = {"batch_size": 32, "num_workers": 4, "pin_memory": True}
    return DataLoader(dataset, **{**defaults, **kwargs})
```

### 7.3 海象运算符 `:=`（Python 3.8+）

```python
if (n := len(batch)) > 1000:
    logger.warning("batch too large: %d", n)

while (line := f.readline()):
    process(line)
```

写 inference server 时用来"边拿边判断"很好用。

### 7.4 Pattern Matching `match-case`（Python 3.10+）

处理复杂的结构化数据（如 LangGraph / Agent 的消息路由）：

```python
def handle(msg: dict) -> str:
    match msg:
        case {"role": "user", "content": str(text)}:
            return f"user said: {text}"
        case {"role": "assistant", "tool_calls": [*tools]}:
            return f"assistant called {len(tools)} tools"
        case {"role": "system"} as m:
            return f"system: {m}"
        case _:
            return "unknown"
```

`match-case` 比一连串 `if-elif` 清晰 10 倍。

### 7.5 `itertools`：迭代器瑞士军刀

```python
from itertools import chain, islice, groupby, starmap, product, accumulate

for item in chain(list1, list2, list3):
    ...

for item in islice(iter(huge_dataset), 100):
    ...

for key, group in groupby(sorted(samples, key=lambda s: s["lang"]), key=lambda s: s["lang"]):
    print(key, list(group))

for lr, bs in product([1e-4, 3e-4], [32, 64, 128]):
    ...
```

---

## 8. 常见陷阱（血泪教训）

### 8.1 可变默认参数：Python 最臭名昭著的坑

```python
def append_log(msg, logs=[]):
    logs.append(msg)
    return logs

a = append_log("x")
b = append_log("y")
print(b)
```

**原因**：默认值在**函数定义时**创建一次，后续调用共享同一个 list。
**修复**：

```python
def append_log(msg, logs=None):
    if logs is None:
        logs = []
    logs.append(msg)
    return logs
```

同理，`dataclass` 的可变默认必须用 `field(default_factory=list)`，之前已经看到过。

### 8.2 闭包捕获循环变量

```python
handlers = [lambda: print(i) for i in range(3)]
for h in handlers:
    h()

handlers = [lambda i=i: print(i) for i in range(3)]
```

写回调、装饰器时极其容易踩。

### 8.3 浅拷贝 vs 深拷贝

```python
import copy
a = {"tokens": [1, 2, 3]}
b = a.copy()
b["tokens"].append(4)
print(a["tokens"])

c = copy.deepcopy(a)
```

Tensor / ndarray 同理——`.clone()` vs `.detach()` 要分清。

### 8.4 自定义异常：信息要具体

```python
class ModelLoadError(RuntimeError):
    """raised when checkpoint loading fails."""

def load(path):
    try:
        return torch.load(path)
    except Exception as e:
        raise ModelLoadError(
            f"failed to load checkpoint from {path}: {type(e).__name__}: {e}"
        ) from e
```

**原则**：异常要告诉调用者"具体哪一步错了"，而不是 `Exception: error`。

---

## 9. 虚拟环境与包管理

推荐 2026 年的最佳实践：**用 `uv` 代替 pip/conda**（速度快 10 倍以上）。

```bash
pip install uv

uv venv
source .venv/bin/activate

uv pip install torch transformers

uv pip install -r requirements.txt
```

备选：
- 纯 Python 项目：`poetry`
- 需要复杂 CUDA / C++ 环境：`conda` / `mamba`

---

## 10. 一个完整的"工程风格"示例

```python
"""A minimal, production-style script template."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Config:
    input_path: Path
    output_path: Path
    top_k: int = 10


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    ns = parser.parse_args()
    return Config(input_path=ns.input, output_path=ns.output, top_k=ns.top_k)


def run(cfg: Config) -> None:
    logger.info("config: %s", asdict(cfg))
    texts = cfg.input_path.read_text(encoding="utf-8").splitlines()
    results = [{"text": t, "score": len(t)} for t in texts[: cfg.top_k]]
    cfg.output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    logger.info("wrote %d items to %s", len(results), cfg.output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
```

这套骨架在 AIGC 项目里非常常见：`argparse` 收参数 → `dataclass` 结构化配置 → 主函数调用 → `logging` 记录。

---

## 小结

| 旧写法 | 新写法 |
|---|---|
| `os.path.join(...)` | `Path(...) / ...` |
| `"x=" + str(x)` | `f"{x=}"` |
| `print("warn: ...")` | `logger.warning(...)` |
| 手写 `__init__`/`__repr__` | `@dataclass` |
| `except:` | `except SpecificError as e: ... from e` |

下一节我们进入 Python 的**进阶特性**：装饰器、生成器、上下文管理器。这些是理解 PyTorch、Transformers、FastAPI 等框架源码的基础。
