# 模块 01：现代 Python 编程基础与进阶

> AIGC 算法工程师写代码的第一原则：**用 Pythonic 的方式写清楚、可维护、可调试的代码。**
> 本模块的目标是让你从"能用 Python 跑通脚本"进化到"能用 Python 写出工程级代码"。

---

## 为什么这一步至关重要？

很多算法同学有过这样的经历：能训出 SOTA 模型，但代码一团乱麻，难以复用、难以上线。
顶级 AIGC 开源项目（`transformers`、`diffusers`、`vllm`、`nanochat`）之所以能流行，
一方面是算法强，另一方面是**工程水平高**——代码结构清晰、类型注解齐备、错误处理健壮。

本模块覆盖的能力，正是区分"会跑脚本"和"工程师"的分水岭。

---

## 学习内容

| # | 文档 | 核心话题 |
|---|---|---|
| 01 | [modern-python-basics](./01-modern-python-basics.md) | 数据结构、魔法方法、惯用法、itertools、match-case、常见陷阱 |
| 02 | [advanced-features](./02-advanced-features.md) | 装饰器、生成器、上下文管理器、functools |
| 03 | [async-programming](./03-async-programming.md) | asyncio、async/await、并发模型 |
| 04 | [type-hints](./04-type-hints.md) | typing、ParamSpec、Protocol、Pydantic |
| 05 | [engineering-best-practices](./05-engineering-best-practices.md) | 项目结构、pytest、调试、profiling、ruff/mypy |

---

## 示例代码（`examples/`）

| 文件 | 说明 |
|---|---|
| `decorators_demo.py` | 装饰器工厂、`functools.wraps`、类装饰器、`@lru_cache` 实战 |
| `generators_demo.py` | 生成器、`yield from`、数据流水线范式 |
| `async_demo.py` | `asyncio.gather`、并发请求、异步上下文管理器 |
| `type_hints_demo.py` | `ParamSpec`、`Protocol`、`TypedDict`、Pydantic v2 |

运行方式：

```bash
cd 01-python-foundations/examples
python decorators_demo.py
```

---

## 推荐配套阅读

- [Python 官方 typing 文档](https://docs.python.org/3/library/typing.html)
- [Real Python - Advanced Features](https://realpython.com/tutorials/advanced/)
- [Fluent Python, 2nd Edition](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/)
- [Pydantic v2 官方文档](https://docs.pydantic.dev/latest/)

---

## 自检清单

学完本模块，你应该能自信地回答以下问题：

- [ ] 为什么 `@functools.wraps` 是自定义装饰器的标配？
- [ ] 生成器 vs 列表，在什么场景下内存差异会是数量级的？
- [ ] `async def` 定义的函数直接调用会得到什么？为什么不是执行结果？
- [ ] `asyncio.gather` 和 `asyncio.create_task` 的区别？
- [ ] `List[int]` 和 `list[int]` 有什么区别？Python 3.10+ 推荐哪个？
- [ ] `Protocol` 相比 `ABC`（抽象基类）的优势是什么？
- [ ] 什么是 "duck typing"？Python 的 `Protocol` 如何把它做成可静态检查的？
- [ ] `__getitem__` + `__len__` 为什么能让一个类支持 `for x in ds`？
- [ ] 可变默认参数 `def f(x=[])` 为什么危险？怎么修？
- [ ] `pyproject.toml` 相比 `setup.py` 的优势是什么？
- [ ] 如何用 `torch.profiler` 定位训练的 GPU 瓶颈？
