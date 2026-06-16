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
| 00 | [python-engineering-theory](./00-python-engineering-theory.md) | Python 工程心智模型：数据结构、协议、生成器、async、类型、测试、profiling |
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

## 理论与实践怎么组织

本模块建议按三层学习：

| 层次 | 要回答的问题 | 对应材料 |
|---|---|---|
| 理论层 | Python 工程代码如何用数据结构、协议、流程控制、类型和测试守住边界？ | `00-python-engineering-theory.md` |
| 语言机制层 | Python 的对象模型、迭代器、协程、类型系统分别解决什么问题？ | `01` ~ `04` 文档 |
| 工程规范层 | 如何把脚本写成可测试、可调试、可维护的工程代码？ | `05-engineering-best-practices.md` |
| 模板层 | 如何把装饰器、生成器、async、类型注解落到可运行小例子？ | `examples/` |

学习顺序建议：

1. 先读 `00`，建立 Python 工程代码的整体心智模型。
2. 再读 `01` 和 `02`，掌握 Pythonic 写法和语言机制。
3. 读 `04`，用类型注解约束工程接口。
4. 最后读 `05` 并跑 examples，把测试、调试、profiling 接入日常代码。

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
