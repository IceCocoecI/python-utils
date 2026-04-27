"""类型注解综合示例 + Pydantic。
运行：python type_hints_demo.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, ParamSpec, Protocol, TypedDict, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class SupportsInference(Protocol):
    """Protocol：只要实现 generate 方法就满足这个协议。"""
    def generate(self, prompt: str, max_tokens: int) -> str: ...


class LocalModel:
    def generate(self, prompt: str, max_tokens: int) -> str:
        return f"[local] answer to {prompt} ({max_tokens} tok)"


class RemoteModel:
    def generate(self, prompt: str, max_tokens: int) -> str:
        return f"[remote] answer to {prompt} ({max_tokens} tok)"


def run_prompt(model: SupportsInference, prompt: str) -> str:
    """不管是 Local 还是 Remote，只要有 generate 方法就能传进来。"""
    return model.generate(prompt, max_tokens=128)


class TokenizedSample(TypedDict):
    """TypedDict：结构化字典。"""
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def compute_loss(batch: TokenizedSample) -> float:
    return sum(batch["labels"]) / max(len(batch["labels"]), 1)


Precision = Literal["fp16", "bf16", "fp32"]


@dataclass
class TrainConfig:
    """dataclass：简单配置类。"""
    lr: float = 1e-4
    batch_size: int = 32
    precision: Precision = "bf16"


def log_calls(func: Callable[P, R]) -> Callable[P, R]:
    """带 ParamSpec 的装饰器——IDE 会保留原参数提示。"""
    import functools
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"[log] {func.__name__}(args={args}, kwargs={kwargs})")
        return func(*args, **kwargs)
    return wrapper


@log_calls
def train_step(lr: float, grad_norm: float) -> dict[str, float]:
    return {"loss": 0.1, "lr": lr, "grad_norm": grad_norm}


def demo_pydantic():
    """Pydantic v2：运行时校验。"""
    try:
        from pydantic import BaseModel, Field
    except ImportError:
        print("(skip pydantic: pip install pydantic to enable)")
        return

    class InferenceRequest(BaseModel):
        prompt: str = Field(min_length=1, max_length=4096)
        max_tokens: int = Field(default=256, ge=1, le=4096)
        temperature: float = Field(default=0.7, ge=0, le=2)
        stream: bool = False

    ok = InferenceRequest(prompt="hi", temperature=1.0)
    print("valid:", ok.model_dump())

    try:
        InferenceRequest(prompt="", temperature=3.0)
    except Exception as e:
        print("caught invalid input:")
        print(str(e)[:200])


if __name__ == "__main__":
    print("== Protocol 鸭子类型 ==")
    print(run_prompt(LocalModel(), "what is LoRA?"))
    print(run_prompt(RemoteModel(), "what is LoRA?"))

    print("\n== TypedDict ==")
    batch: TokenizedSample = {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "labels": [0, 1, 1],
    }
    print("loss:", compute_loss(batch))

    print("\n== Literal + dataclass ==")
    cfg = TrainConfig(precision="bf16")
    print(cfg)

    print("\n== ParamSpec 装饰器 ==")
    print(train_step(1e-4, 0.9))

    print("\n== Pydantic 校验 ==")
    demo_pydantic()
