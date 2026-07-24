# 实践速通路线（5 天）

适合已经熟悉 Python 类型注解、LangChain message/tool 协议和基本 Agent 概念的人。不要跳过 checkpoint、Store 与 interrupt，它们才是 LangGraph 相对普通链式代码的核心价值。

| 天 | 阅读与示例 | 必做改动 | 当日交付 |
|---:|---|---|---|
| 1 | 01-03 | 写一个带循环上限的条件图 | state + topology + route tests |
| 2 | 04-06 | 用 user/thread 两个维度验证隔离 | memory namespace + isolation tests |
| 3 | 07-08 | 工具循环后加入高风险审批 | tool + resume test |
| 4 | 09-10 | Stream 一个 Send fan-out 子图 | event sample + reducer test |
| 5 | 综合 Lab | 完成一个 P1 改造 | 代码、测试、失败复盘 |

每天开始先运行对应 `--self-test`，结束时用自己的话回答章节自检题。若不能解释为什么恢复不能再次提交完整 state，应回到第 08 章。
