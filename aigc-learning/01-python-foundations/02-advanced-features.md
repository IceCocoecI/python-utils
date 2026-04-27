# 02 · 装饰器 / 生成器 / 上下文管理器

> 这三个特性撑起了 Python 生态的半壁江山：
> - FastAPI `@app.get(...)` 是装饰器
> - PyTorch `DataLoader` 每次迭代本质是生成器协议
> - `with torch.no_grad():` 是上下文管理器
>
> 理解它们，你才能读懂主流 AIGC 框架的源码。

---

## 1. 装饰器（Decorator）

### 1.1 核心本质

**装饰器就是一个"函数加工厂"：接收一个函数，返回一个被加工过的函数。**

```python
def log_call(func):
    def wrapper(*args, **kwargs):
        print(f"calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"  returned {result!r}")
        return result
    return wrapper

@log_call
def add(a, b):
    return a + b

add(3, 5)
```

`@log_call` 等价于 `add = log_call(add)`。

### 1.2 别忘了 `@functools.wraps`

不加 `wraps` 会丢失原函数的 `__name__`、docstring、类型注解。

```python
import functools

def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper
```

### 1.3 带参数的装饰器（装饰器工厂）

三层嵌套：工厂 → 装饰器 → 包装器。

```python
import functools, time

def retry(n_times: int = 3, delay: float = 1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for i in range(n_times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    time.sleep(delay)
            raise last_err
        return wrapper
    return decorator

@retry(n_times=5, delay=0.5)
def call_api():
    ...
```

### 1.4 类装饰器与 `__call__`

当装饰器需要维护状态时，类装饰器更合适：

```python
class CallCounter:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

@CallCounter
def greet(name):
    return f"Hello, {name}!"

greet("Ada"); greet("Bob")
print(greet.count)
```

### 1.5 标准库里的实用装饰器

| 装饰器 | 用途 |
|---|---|
| `@functools.lru_cache(maxsize=128)` | 自动记忆化，算法中常用于缓存子结果 |
| `@functools.cached_property` | 只计算一次的属性（懒加载） |
| `@functools.singledispatch` | 基于第一个参数类型分派的多态 |
| `@dataclasses.dataclass` | 自动生成 `__init__` / `__repr__` / `__eq__` |
| `@staticmethod` / `@classmethod` | 类方法 |
| `@property` | 属性访问器 |

### 1.6 AIGC 场景实战

**给训练函数加计时 + 显存监控：**

```python
import functools, time
import torch

def gpu_profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{func.__name__}: {dt:.2f}s, peak_mem={mem:.2f}GB")
        return out
    return wrapper

@gpu_profile
def train_one_epoch(model, loader):
    ...
```

---

## 2. 生成器（Generator）

### 2.1 从列表到生成器

```python
def read_squares_list(n):
    result = []
    for i in range(n):
        result.append(i * i)
    return result

def read_squares_gen(n):
    for i in range(n):
        yield i * i

for x in read_squares_gen(10**9):
    if x > 100:
        break
```

生成器是**惰性求值**：只在被请求下一个值时才计算，内存占用 O(1)。

### 2.2 大数据集流式处理（AIGC 强相关）

训练时读取 TB 级文本语料：

```python
from pathlib import Path

def iter_corpus(data_dir: Path):
    for file in sorted(data_dir.glob("*.jsonl")):
        with file.open(encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

for sample in iter_corpus(Path("./data")):
    process(sample)
```

HuggingFace `datasets` 的 `IterableDataset`、PyTorch `DataLoader` 本质都是生成器协议。

### 2.3 生成器表达式

```python
prompt_lens = (len(p) for p in prompts)
total = sum(prompt_lens)

avg_len = sum(len(p) for p in prompts) / len(prompts)
```

### 2.4 `yield from` 与生成器组合

```python
def flatten(nested):
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

list(flatten([1, [2, [3, 4]], 5]))
```

### 2.5 生成器协议与 `Iterator`

```python
class CountDown:
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        return self
    def __next__(self):
        if self.n <= 0:
            raise StopIteration
        self.n -= 1
        return self.n

for i in CountDown(3):
    print(i)
```

`PyTorch` `DataLoader` 内部就实现了 `__iter__` / `__next__`。

---

## 3. 上下文管理器（Context Manager）

### 3.1 `with` 的本质

```python
with open("x.txt") as f:
    data = f.read()
```

等价于：

```python
f = open("x.txt")
try:
    data = f.read()
finally:
    f.close()
```

核心是实现 `__enter__` / `__exit__` 协议。

### 3.2 自定义上下文管理器：两种写法

**类写法：**

```python
class Timer:
    def __init__(self, label="block"):
        self.label = label
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        dt = time.perf_counter() - self.t0
        print(f"{self.label}: {dt:.3f}s")

with Timer("forward"):
    out = model(x)
```

**`contextlib.contextmanager` 写法（更推荐）：**

```python
from contextlib import contextmanager

@contextmanager
def timer(label="block"):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"{label}: {dt:.3f}s")

with timer("forward"):
    out = model(x)
```

### 3.3 AIGC 场景实战：全局状态切换

PyTorch 用上下文管理器优雅地控制 autograd、精度、设备：

```python
with torch.no_grad():
    output = model(x)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(x)
    loss = loss_fn(output, y)
```

我们也可以自己写：

```python
@contextmanager
def eval_mode(model):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)

with eval_mode(model):
    predictions = model(x)
```

### 3.4 异步上下文管理器（Python 3.11+）

```python
from contextlib import asynccontextmanager
import httpx

@asynccontextmanager
async def http_client():
    client = httpx.AsyncClient(timeout=30)
    try:
        yield client
    finally:
        await client.aclose()

async def main():
    async with http_client() as client:
        resp = await client.get("https://api.openai.com/v1/models")
```

---

## 4. 进阶工具：`functools`

| 工具 | 用途示例 |
|---|---|
| `functools.partial` | 固化部分参数：`partial(torch.randn, device="cuda")` |
| `functools.reduce` | 累积运算：`reduce(lambda a,b: a*b, shape)` |
| `functools.lru_cache` | 记忆化（注意：不能用在返回 Tensor 的函数上，会内存泄露） |
| `functools.cached_property` | 懒加载属性 |
| `functools.singledispatch` | 基于类型分派 |

---

## 小结

| 特性 | 一句话记忆 | 最常见的 AIGC 场景 |
|---|---|---|
| 装饰器 | "函数的加工厂" | `@torch.no_grad()`、`@app.post(...)` |
| 生成器 | "惰性求值的数据流" | `DataLoader` 迭代、大语料流式读取 |
| 上下文管理器 | "确保清理的 try-finally 糖" | `with torch.autocast()`、`with open(...)` |

下一节我们讨论 `async/await`——现代 AIGC 推理服务（vLLM、SGLang、FastAPI）的基石。
