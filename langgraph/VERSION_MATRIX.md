# 版本矩阵

## 学习基线

| 组件 | 当前 Conda `langgraph` 环境 | 课程约束 | 用途 |
|---|---:|---:|---|
| Python | 3.11.14 | `>=3.10,<3.14` | 类型注解、标准库测试 |
| langgraph | 1.0.6 | `>=1.0.6,<1.1` | 图运行时 |
| langgraph-checkpoint | 4.0.0 | 由 langgraph 解析 | checkpoint 接口与内存实现 |
| langgraph-prebuilt | 1.0.6 | 由 langgraph 解析 | `ToolNode`、`tools_condition` |
| langchain-core | 1.2.7 | `>=1.2.7,<2.0` | 消息与工具协议 |
| typing-extensions | 4.15.0 | `>=4.15,<5` | `TypedDict` 等类型支持 |

本仓库没有把 `langchain`、`langchain-openai`、`pytest` 设为必需项。当前环境未安装它们，核心课程也不需要它们。

## 兼容策略

1. 使用 `StateGraph(..., context_schema=...)`，不使用已经弃用的 `config_schema`。
2. 使用 `InMemorySaver`；`MemorySaver` 虽是别名，但新代码统一采用前者。
3. 使用 `Runtime[Context]` 读取运行时上下文和 Store。
4. 低层工具循环使用 `ToolNode + tools_condition`。
5. 不使用 `langgraph.prebuilt.create_react_agent`。它在 1.x 已弃用；需要高层预构建 Agent 时，应根据对应 LangChain 版本查阅 `langchain.agents.create_agent`。
6. 恢复中断使用 `Command(resume=value)`，不会把完整 state 当新输入再次从 `START` 执行。

## 学习约束与生产锁定

`requirements.txt` 给出的是学习兼容范围，不是生产 lockfile。生产项目应：

- 使用 `uv.lock`、Poetry lock 或带哈希的 requirements 锁定完整依赖树；
- 升级前先运行状态、路由、持久化、中断恢复和副作用幂等测试；
- 检查 checkpoint 序列化格式和后端迁移说明；
- 分开验证 LangGraph、LangChain、模型 provider 三层升级，不要一次同时升级全部组件。

## 检查当前环境

```bash
conda run -n langgraph python -c "import importlib.metadata as m; print(m.version('langgraph'))"
conda list -n langgraph | grep -E '^(langgraph|langchain-core|typing-extensions)'
```

若版本超出 1.x，请先阅读 release notes，再决定升级代码或新建隔离环境，不要直接修改示例去兼容多个不相干的大版本。
