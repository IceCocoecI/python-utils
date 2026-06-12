# 00 · Python 工程理论框架

> 本文是模块 01 的理论底座。
> 后续 Python 语法、装饰器、生成器、async、类型注解、测试和 profiling 都服务于同一个目标：
> 把一次性脚本变成可复用、可调试、可协作的工程代码。

---

## 1. 本模块真正解决什么问题？

AIGC 项目经常从一个 notebook 或训练脚本开始，但很快会变成完整系统：

```text
数据处理 → 训练脚本 → 实验配置 → 推理服务 → 评估脚本 → 部署工具
```

如果 Python 基础不稳，常见问题会反复出现：

| 问题 | 表现 |
|---|---|
| 状态混乱 | 全局变量、隐式路径、随机种子不可控 |
| 接口不清 | 函数参数随意传字典，字段缺失运行时才炸 |
| 复用困难 | 代码只能在一个脚本里跑，换数据/模型就要复制粘贴 |
| 调试困难 | 全靠 `print`，异常信息不具体，日志没有上下文 |
| 性能不可见 | 不知道时间花在哪里、内存被谁占了 |
| 协作困难 | 没有测试、类型注解、格式化和依赖锁定 |

本模块的目标不是“学更多语法”，而是建立工程判断：

```text
用清晰的数据结构表达状态
用明确的函数签名表达接口
用测试和类型检查守住边界
用日志和 profiling 观察系统
```

---

## 2. Python 代码的四层结构

可以把 Python 工程代码分成四层：

| 层次 | 关注点 | 典型机制 |
|---|---|---|
| 数据层 | 如何表达配置、样本、请求、响应 | `dataclass`、`TypedDict`、Pydantic |
| 行为层 | 如何组织计算逻辑 | 函数、类、协议、装饰器 |
| 流程层 | 如何串起 I/O、并发、资源管理 | 生成器、上下文管理器、async |
| 工程层 | 如何验证、调试、发布、协作 | pytest、logging、profiling、ruff、mypy |

写 AIGC 代码时，很多 bug 都来自层次混乱。
例如把配置字典直接传遍全项目，本质是数据层没有建模；在函数里偷偷改全局状态，本质是行为层和流程层边界不清。

---

## 3. 数据结构是工程边界

### 3.1 `dict` 适合临时数据，不适合长期接口

临时脚本里这样写很方便：

```python
cfg = {"lr": 1e-4, "batch_size": 32, "epochs": 3}
```

但当配置跨文件、跨函数、跨团队流动时，裸 `dict` 会带来问题：

- 字段名打错运行时才发现；
- 默认值散落在多个地方；
- IDE 无法提示；
- 很难知道哪些字段是必需的。

更稳的做法：

| 场景 | 推荐 |
|---|---|
| 简单不可变配置 | `dataclass(frozen=True)` |
| 字典形态但需要静态约束 | `TypedDict` |
| API 请求/响应和运行时校验 | Pydantic |
| 大型训练配置 | Hydra / OmegaConf，详见模块 04 |

### 3.2 对象不是为了“面向对象”，而是为了封装不变量

一个类值得存在，通常因为它维护了某些不变量：

- tokenizer 和 vocab 必须匹配；
- dataset 的 `__len__` 和 `__getitem__` 必须一致；
- model wrapper 必须保证 device、dtype、eval/train 状态正确；
- client 必须复用连接、控制超时、处理重试。

如果一个类只是把几个无状态函数包起来，函数往往更清楚。

---

## 4. Python 数据模型：框架互操作的底层协议

Python 很多“魔法”其实是协议。

| 协议 | 方法 | AIGC 场景 |
|---|---|---|
| 可迭代 | `__iter__` | 数据流、日志流、token 流 |
| 序列 | `__len__`、`__getitem__` | PyTorch Dataset |
| 可调用 | `__call__` | transform、callback、model wrapper |
| 上下文管理 | `__enter__`、`__exit__` | 文件、Profiler、临时状态 |
| 异步上下文 | `__aenter__`、`__aexit__` | HTTP client、连接池 |

理解这些协议后，很多框架行为会变得直观：

- PyTorch `Dataset` 为什么实现 `__len__` 和 `__getitem__` 就能被 `DataLoader` 使用。
- `nn.Module` 为什么能像函数一样调用。
- `torch.no_grad()` 为什么能用 `with` 管理全局梯度状态。
- LLM 流式输出为什么可以建模成 iterator 或 async iterator。

---

## 5. 生成器是数据管线的基础抽象

生成器的核心价值是**惰性计算**：

```text
需要一个样本 → 读取一个样本 → 处理一个样本 → 产出一个样本
```

它适合：

- 大文件逐行读取；
- JSONL 训练语料流式处理；
- 日志、事件、token 流；
- 数据清洗 pipeline；
- 低内存预处理。

判断标准：

| 问题 | 倾向 |
|---|---|
| 数据能全部放内存，需要随机访问 | list / array / Dataset |
| 数据很大，只顺序消费一次 | generator |
| 需要异步等待外部 I/O | async generator |

不要为了“高级”而到处用生成器。需要多次遍历、排序、随机采样时，生成器反而会增加复杂度。

---

## 6. 装饰器是横切逻辑

装饰器适合处理“和业务逻辑正交”的需求：

- 计时；
- 缓存；
- 重试；
- 权限校验；
- tracing；
- 输入输出日志；
- 参数校验。

它不适合隐藏核心业务流程。一个装饰器如果改变了函数语义，却不体现在函数名和类型签名里，会让调试变难。

工程原则：

- 自定义装饰器必须用 `functools.wraps`。
- 需要保留类型签名时使用 `ParamSpec`。
- 复杂重试逻辑优先使用成熟库或显式函数封装。

---

## 7. 上下文管理器管理资源和临时状态

`with` 的意义是把“进入”和“退出”绑定在一起：

```text
进入资源 / 临时状态
  执行业务逻辑
无论成功或异常，都执行清理
```

AIGC 常见场景：

- 打开和关闭文件；
- 关闭 HTTP client；
- 记录 profiler 区间；
- 临时切换 `model.eval()`；
- 临时关闭梯度；
- 申请和释放 GPU 资源。

判断标准：只要有“必须恢复”的状态，就应该考虑上下文管理器。

---

## 8. Async 解决 I/O 并发，不解决 CPU/GPU 计算

`asyncio` 的价值是让一个线程同时等待多个 I/O：

| 适合 async | 不适合 async |
|---|---|
| 并发调用 LLM API | 大矩阵计算 |
| SSE / WebSocket 流式输出 | 图像解码 CPU 热点 |
| HTTP 请求、数据库查询 | PyTorch GPU kernel 本身 |
| 多工具调用等待结果 | Python 密集循环 |

LLM 应用服务通常需要 async，因为它们大量时间在等待：

```text
用户请求 → 检索服务 → reranker → LLM API / 推理后端 → 流式返回
```

但训练脚本通常不需要 async；训练并行主要靠 DataLoader workers、CUDA stream、分布式通信和多进程。

---

## 9. 类型注解是接口文档，也是自动检查

类型注解的价值不是让 Python 变成 Java，而是让接口更明确：

```python
def build_prompt(question: str, contexts: list[str]) -> str:
    ...
```

它能帮助你提前发现：

- 参数传错；
- 返回值不一致；
- 可选值没有处理 `None`；
- dict 字段拼错；
- 装饰器破坏函数签名。

在 AIGC 项目中，尤其值得加类型的地方：

- 配置对象；
- 数据样本；
- API 请求/响应；
- 工具调用 schema；
- 模型 wrapper 的输入输出；
- callback / hook / protocol。

---

## 10. 错误处理和日志是可观测性

异常处理的目标不是“别让程序崩”，而是让失败可诊断。

差的异常：

```text
ValueError: bad input
```

好的异常应该包含：

- 哪个文件、样本、请求出错；
- 关键参数是什么；
- 期望格式是什么；
- 当前拿到的值是什么；
- 是否可以重试。

日志也一样。训练、推理、RAG、Agent 都应该记录关键上下文：

- run id / request id；
- config 摘要；
- 输入长度和输出长度；
- 检索到的文档 id；
- 工具调用参数和结果；
- latency、token 数、显存峰值。

---

## 11. 测试不是只给业务系统用

算法代码同样需要测试，只是测试重点不同：

| 测试类型 | AIGC 场景 |
|---|---|
| shape 测试 | attention、embedding、batch collation |
| 数值等价测试 | naive attention vs optimized attention |
| 边界测试 | 空输入、超长文本、不同 batch size |
| 随机性测试 | 固定 seed 后结果可复现 |
| schema 测试 | SFT 数据、tool call JSON、API response |
| smoke test | 脚本在 CPU 小数据上能跑通 |

一个好的学习仓库也应该让示例在低成本环境里跑通。真实模型可以作为可选路径，小模型/合成数据应该作为默认路径。

---

## 12. Profiling 是工程判断，不是最后一步

不要凭感觉优化。Python 项目常见瓶颈位置：

| 瓶颈 | 工具 |
|---|---|
| Python 函数耗时 | `cProfile`、`line_profiler` |
| 数据加载 | PyTorch profiler、日志打点 |
| GPU kernel | `torch.profiler`、Nsight |
| 内存占用 | `tracemalloc`、PyTorch memory stats |
| API 延迟 | tracing、request id、p95 latency |

优化流程：

```text
先保证正确
  ↓
测量瓶颈
  ↓
只改一个变量
  ↓
再次测量
  ↓
保留能证明收益的数据
```

---

## 13. 从理论映射到本模块文档

| 理论问题 | 对应文档 |
|---|---|
| Pythonic 数据结构、路径、日志、异常、对象模型怎么写？ | [01-modern-python-basics](./01-modern-python-basics.md) |
| 装饰器、生成器、上下文管理器分别适合解决什么问题？ | [02-advanced-features](./02-advanced-features.md) |
| async/await 适合哪些 LLM 服务场景？ | [03-async-programming](./03-async-programming.md) |
| 类型注解如何约束配置、样本、请求、工具 schema？ | [04-type-hints](./04-type-hints.md) |
| 测试、调试、profiling、ruff/mypy 如何组成工程闭环？ | [05-engineering-best-practices](./05-engineering-best-practices.md) |

---

## 14. 工程判断清单

- [ ] 这个状态应该是局部变量、配置对象，还是类的字段？
- [ ] 这个接口是否需要 `dataclass`、`TypedDict` 或 Pydantic？
- [ ] 这个类是否维护了真实不变量，还是只是函数集合？
- [ ] 这段数据处理是否应该流式化？
- [ ] 这个装饰器是否隐藏了业务语义？
- [ ] 是否有需要 `with` 保证释放或恢复的资源？
- [ ] 这个并发问题是 I/O 等待，还是 CPU/GPU 计算？
- [ ] 关键函数是否有类型注解？
- [ ] 异常信息是否足够定位数据、请求或配置？
- [ ] 是否有最小 smoke test 证明脚本能跑通？
- [ ] 优化前是否已经 profiling？

