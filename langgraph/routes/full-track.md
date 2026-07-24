# 完整学习路线（4 周）

每天建议 60-90 分钟。每个阶段都要经历“读、跑、改、测、复盘”。

## 第 1 周：图与状态

| 天 | 内容 | 动手任务 | 验收 |
|---:|---|---|---|
| 1 | 01 首个图 | 跑线性图，画 state 更新表 | 能解释 compile 与 invoke |
| 2-3 | 02 State/Reducer | 改写 reducer，制造一次并发冲突 | 能解释覆盖与累积 |
| 4-5 | 03 控制流 | 增加一条路由和循环上限 | 所有分支有测试 |
| 6 | 回顾 | 不看代码重写最小图 | `--self-test` 通过 |
| 7 | 验收 | 完成基础阶段题 | 得分 >= 80% |

本周交付：一张状态字段表、一张图拓扑、一个自定义 reducer、三条路由测试。

## 第 2 周：上下文与记忆

| 天 | 内容 | 动手任务 | 验收 |
|---:|---|---|---|
| 8-9 | 04 Runtime Context | 增加 tenant 和 feature flag | 两个用户输出隔离 |
| 10-11 | 05 Persistence | 对比相同/不同 thread | 能读取 history 和分叉 |
| 12-13 | 06 Store | 保存、更新、删除用户偏好 | 跨 thread 共享且跨 user 隔离 |
| 14 | 复盘 | 画五类上下文边界图 | 不混淆 State/Context/Store |

本周交付：线程隔离测试、长期记忆 namespace 设计、数据删除策略说明。

## 第 3 周：Agent 与可恢复执行

| 天 | 内容 | 动手任务 | 验收 |
|---:|---|---|---|
| 15-16 | 07 Tools | 新增一个有 schema 的工具 | tool call id 与结果正确 |
| 17-18 | 08 HITL | 审批中修改 state 后恢复 | START 不重复执行 |
| 19-20 | 09 Streaming/可靠性 | 注入两次失败，观察 retry | 副作用有幂等说明 |
| 21 | 复盘 | 对比 Graph/Functional API | 能给出选型理由 |

本周交付：工具协议测试、中断恢复测试、失败重放分析。

## 第 4 周：组合与项目

| 天 | 内容 | 动手任务 | 验收 |
|---:|---|---|---|
| 22-23 | 10 子图/多 Agent | 对比 direct/wrapper，修改一条 handoff 和 Send fan-out | 能解释父目标、状态映射且 reducer 无冲突 |
| 24 | Lab 架构 | 先读 TODO，自行画图 | state ownership 清楚 |
| 25-27 | 综合 Lab | 跑五条业务路径，完成两项改造 | 全部测试通过 |
| 28 | 总验收 | 讲解设计、失败模式和生产差距 | 形成项目报告 |

最终运行：

```bash
conda run -n langgraph python langgraph/scripts/smoke_test.py --warnings-as-errors
conda run -n langgraph python -m unittest discover -s langgraph/tests -v
conda run -n langgraph python langgraph/scripts/check_links.py
```
