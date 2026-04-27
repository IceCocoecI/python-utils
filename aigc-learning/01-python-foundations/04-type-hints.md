# 04 · 类型注解与 Pydantic

> 类型注解是 Python 近 5 年最重要的变化。
> 顶级 AIGC 项目（`transformers`、`diffusers`、`vllm`、`fastapi`）几乎每一行函数签名都有类型注解。
> 这不仅能让 IDE 自动补全、提前暴露 bug，更是写"工程级"代码的分水岭。

---

## 1. 为什么需要类型注解？

```python
def softmax(x, dim=None, dtype=None):
    ...

def softmax(
    x: torch.Tensor,
    dim: int | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    ...
```

第二版的优势：
- IDE 立即告诉你 `x.` 后面有什么方法。
- `mypy`/`pyright` 会在你传错类型时直接报错。
- 别人读代码时一眼知道输入输出。

**重要**：Python 的类型注解**不会在运行时强制检查**（除非你主动用 `pydantic`），它主要服务于静态分析和文档。

---

## 2. 基础语法

### 2.1 基本类型

```python
name: str = "llama"
steps: int = 1000
lr: float = 1e-4
training: bool = True
```

### 2.2 容器类型（Python 3.9+ 推荐内建泛型）

```python
tokens: list[int] = [1, 2, 3]
vocab: dict[str, int] = {"hello": 1}
shape: tuple[int, int, int] = (3, 224, 224)
unique_labels: set[str] = {"cat", "dog"}

from typing import List, Dict
tokens: List[int] = [1, 2, 3]
```

### 2.3 可选与联合

```python
from typing import Optional

def load(path: str | None = None) -> dict | list:
    ...

def load(path: Optional[str] = None) -> dict:
    ...
```

### 2.4 函数签名

```python
from collections.abc import Callable

def apply(fn: Callable[[int, int], int], a: int, b: int) -> int:
    return fn(a, b)

apply(lambda x, y: x + y, 1, 2)
```

### 2.5 Any 与 Never

```python
from typing import Any, NoReturn

def parse(raw: Any) -> dict: ...

def raise_error(msg: str) -> NoReturn:
    raise RuntimeError(msg)
```

`Any` 会关闭类型检查——**能不用就不用**。

---

## 3. 进阶：Generic 与 TypeVar

```python
from typing import TypeVar

T = TypeVar("T")

def first(items: list[T]) -> T:
    return items[0]

x: int = first([1, 2, 3])
y: str = first(["a", "b"])
```

Python 3.12+ 新语法（PEP 695）：

```python
def first[T](items: list[T]) -> T:
    return items[0]
```

---

## 4. `Protocol`：鸭子类型的静态检查

`Protocol` 让你不用继承也能描述"只要有这个方法就可以"。

```python
from typing import Protocol

class SupportsForward(Protocol):
    def forward(self, x) -> object: ...

def run_inference(model: SupportsForward, x):
    return model.forward(x)
```

`SupportsForward` 不需要任何类显式继承它。这就是 `torch.nn.Module`、`transformers.PreTrainedModel`、`diffusers.ModelMixin` 之间能"互操作"的原理——大家都遵守相同的协议。

---

## 5. `TypedDict`：字典的结构化声明

```python
from typing import TypedDict

class BatchSample(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]

def train_step(batch: BatchSample) -> float:
    ...
```

很多 HuggingFace 数据管线的返回值就是这种"结构化字典"。

---

## 6. `Literal`：字面量类型

```python
from typing import Literal

Precision = Literal["fp16", "bf16", "fp32"]

def cast(tensor, dtype: Precision):
    ...
```

配置类里用 `Literal` 比字符串校验优雅得多。

---

## 7. `ParamSpec` 与 `Concatenate`（装饰器类型保留）

普通装饰器会让 IDE 失去原函数的参数提示。`ParamSpec` 解决了这个问题：

```python
from typing import ParamSpec, TypeVar, Callable
import functools

P = ParamSpec("P")
R = TypeVar("R")

def log_call(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@log_call
def train_model(lr: float, steps: int) -> dict:
    return {"loss": 0.1}
```

写装饰器的标准姿势就是 `Callable[P, R] -> Callable[P, R]`。

---

## 8. Pydantic：类型注解 + 运行时校验

`dataclass` 不做运行时校验，但 `pydantic` 做。Pydantic v2 是 AIGC 领域的标配（FastAPI、OpenAI SDK、LangChain 都用它）。

```python
from pydantic import BaseModel, Field

class InferenceRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4096)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    stream: bool = False

req = InferenceRequest(prompt="hi", temperature=1.5)
print(req.model_dump_json())

InferenceRequest(prompt="hi", temperature=3)
```

**与 FastAPI 结合**：

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/generate")
def generate(req: InferenceRequest) -> dict:
    return {"text": f"echo: {req.prompt}"}
```

FastAPI 会自动：
1. 把 JSON 请求体反序列化为 `InferenceRequest`。
2. 执行所有字段校验。
3. 校验失败自动返回 422 错误。
4. 自动生成 OpenAPI 文档。

这就是为什么 LLM 服务基本都用 FastAPI + Pydantic。

---

## 9. 静态检查工具：`mypy` / `pyright` / `ruff`

推荐的组合（2026 年最佳实践）：

```bash
pip install ruff mypy

ruff check .
ruff format .

mypy src/
```

`ruff` 已经吃掉了 `black` + `isort` + `flake8`，现在是事实标准。
`mypy` / `pyright` 则是类型检查。
在 CI 里把它们都跑一遍，代码质量立刻上一个台阶。

---

## 10. 实战模板：带类型注解的训练脚本骨架

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch.utils.data import DataLoader


class TrainableModel(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def parameters(self) -> list[torch.Tensor]: ...


@dataclass
class TrainConfig:
    lr: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    device: str = "cuda"
    ckpt_dir: Path = Path("./checkpoints")


def train(
    model: TrainableModel,
    train_loader: DataLoader,
    config: TrainConfig,
) -> dict[str, float]:
    ...
    return {"final_loss": 0.12, "best_acc": 0.91}
```

---

## 小结

| 工具 | 作用 |
|---|---|
| 基础注解（`list[int]`, `str | None`） | 让 IDE 能提示 |
| `TypeVar` / `Generic` | 泛型函数 |
| `Protocol` | 鸭子类型的静态检查 |
| `TypedDict` | 结构化字典 |
| `Literal` | 字面量枚举 |
| `ParamSpec` | 装饰器保留参数签名 |
| `pydantic.BaseModel` | 运行时校验 + JSON 序列化 |
| `mypy` / `pyright` | 静态检查 |
| `ruff` | 格式化 + linting |

**一句话原则**：**凡是对外的函数、所有配置类、所有数据流的接口——全部加类型注解。**

至此，`01-python-foundations` 模块完成。接下来进入 `02-deep-learning-libraries`，动手训练真实模型。
