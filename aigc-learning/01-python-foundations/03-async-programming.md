# 03 · 异步编程：`asyncio` / `async-await`

> AIGC 时代，异步编程变得极其重要：
> - LLM 服务（vLLM/FastAPI）需要高并发处理请求
> - Agent 框架（LangGraph）需要并行调用多个 LLM / Tool
> - RAG 需要并发检索多个向量库
> - 训练数据清洗需要并发调用外部 API

---

## 1. 并发模型速览

| 模型 | 适用场景 | 代表 |
|---|---|---|
| 多进程（`multiprocessing`） | CPU 密集型（绕开 GIL） | 数据预处理、特征提取 |
| 多线程（`threading`） | I/O 密集型（简单场景） | 本地文件 I/O |
| 异步（`asyncio`） | I/O 密集型（高并发） | 网络请求、LLM 调用 |
| GPU 并行 | 算力密集型 | 深度学习训练推理 |

**关键记忆：asyncio 是单线程的并发**——用协程切换避免阻塞等待。

---

## 2. 协程的基本形态

### 2.1 `async def` 与 `await`

```python
import asyncio

async def fetch(url: str) -> str:
    print(f"start fetch {url}")
    await asyncio.sleep(1)
    print(f"done fetch {url}")
    return f"<content of {url}>"

result = asyncio.run(fetch("https://example.com"))
print(result)
```

**陷阱**：`async def` 定义的函数**直接调用不会执行**，只会返回一个协程对象：

```python
coro = fetch("x")
print(coro)
```

必须用 `asyncio.run(coro)` 或在另一个协程里 `await coro`。

### 2.2 并发执行：`asyncio.gather`

```python
async def main():
    urls = ["u1", "u2", "u3"]
    results = await asyncio.gather(*(fetch(u) for u in urls))
    return results

asyncio.run(main())
```

三个请求**同时发出**，总耗时 ≈ 最慢的那一个，而不是累加。

### 2.3 `create_task` vs `gather`

```python
async def main():
    task1 = asyncio.create_task(fetch("u1"))
    task2 = asyncio.create_task(fetch("u2"))

    print("do other stuff...")
    await asyncio.sleep(0.3)

    r1 = await task1
    r2 = await task2
```

- `asyncio.create_task(coro)`：立刻把协程放入事件循环（开始执行）。
- `asyncio.gather(*coros)`：集中提交一批协程并等待全部完成。

---

## 3. AIGC 实战：并发调用 LLM API

### 3.1 并发调用多个 prompt

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def ask(prompt: str) -> str:
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

async def batch_ask(prompts: list[str]) -> list[str]:
    return await asyncio.gather(*(ask(p) for p in prompts))

prompts = ["Explain LoRA", "What is FlashAttention?", "How does DPO work?"]
answers = asyncio.run(batch_ask(prompts))
```

**相比串行调用加速 N 倍**（N 为并发度），这是 Agent / 评测系统的常用模式。

### 3.2 限制并发数：`Semaphore`

防止把服务打挂：

```python
async def ask_with_limit(sem: asyncio.Semaphore, prompt: str) -> str:
    async with sem:
        return await ask(prompt)

async def batch_ask(prompts: list[str], concurrency: int = 10) -> list[str]:
    sem = asyncio.Semaphore(concurrency)
    return await asyncio.gather(*(ask_with_limit(sem, p) for p in prompts))
```

### 3.3 带重试与超时

```python
async def ask_safe(prompt: str, timeout: float = 30, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            return await asyncio.wait_for(ask(prompt), timeout=timeout)
        except (asyncio.TimeoutError, Exception) as e:
            if attempt == retries - 1:
                return None
            await asyncio.sleep(2**attempt)
```

---

## 4. 异步迭代器与生成器

### 4.1 流式 LLM 输出

```python
async def stream_llm(prompt: str):
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

async def main():
    async for token in stream_llm("用一句话介绍 diffusion model"):
        print(token, end="", flush=True)

asyncio.run(main())
```

`async for` + `yield` = **异步生成器**，vLLM/SGLang 的流式 API 就是这种模式。

### 4.2 异步上下文管理器

```python
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        r = await client.get("https://huggingface.co")
        print(r.status_code)
```

---

## 5. 常见坑

### 5.1 不能在协程里调用阻塞函数

```python
async def bad():
    time.sleep(5)  # 会阻塞整个事件循环！

async def good():
    await asyncio.sleep(5)
```

如果必须调用阻塞函数（比如 `torch.load`、`cv2.imread`），用：

```python
loop = asyncio.get_running_loop()
result = await loop.run_in_executor(None, blocking_fn, *args)
```

### 5.2 忘记 `await`

```python
async def main():
    fetch("x")
```

Python 只会警告 "coroutine was never awaited"，不会执行。这是最常见的 bug。

### 5.3 在顶层用 `asyncio.run` 但嵌套到 Jupyter 里会报错

Jupyter 自带事件循环，直接：
```python
answers = await batch_ask(prompts)
```
或 `import nest_asyncio; nest_asyncio.apply()`。

---

## 6. 何时用 asyncio，何时不用？

| 用 asyncio | 别用 asyncio |
|---|---|
| 大量网络 I/O | CPU 密集计算（没有加速） |
| 高并发 API 服务 | 简单串行脚本 |
| 需要并发调多个 LLM | PyTorch 模型推理（用 batch 和 stream） |
| 异步数据库查询 | 进程级并行（用 `multiprocessing`） |

---

## 7. 进阶：`asyncio.Queue` 实现生产者-消费者

```python
async def producer(queue: asyncio.Queue, data):
    for x in data:
        await queue.put(x)
    await queue.put(None)

async def consumer(queue: asyncio.Queue):
    while True:
        x = await queue.get()
        if x is None:
            break
        await process(x)

async def main():
    q = asyncio.Queue(maxsize=100)
    await asyncio.gather(producer(q, data), consumer(q))
```

这是 **vLLM 的 scheduler 思想**——一个异步队列 + 多个消费者。

---

## 小结

- `async def` 定义协程；必须在事件循环里运行（`asyncio.run` 或 `await`）。
- 并发关键词：`asyncio.gather` / `create_task` / `Semaphore`。
- 流式输出用 `async for` + 异步生成器。
- 阻塞调用要丢到 `run_in_executor`。
- AIGC 并发调 API、LLM 流式输出、Agent 并行工具调用——全都离不开 asyncio。

下一节进入**类型注解**，它是写好工程代码的最后一块拼图。
