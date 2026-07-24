# 实践验收清单

## P0：图与状态

- [ ] 从零写一个至少三节点的 StateGraph。
- [ ] 为输入、内部状态、输出定义不同 schema。
- [ ] 写一个自定义 reducer，并证明它如何处理重复更新。
- [ ] 制造并修复一次并行写冲突。
- [ ] 为每条条件边写测试。
- [ ] 给循环加入硬上限和失败出口。

## P1：上下文与记忆

- [ ] 通过 Runtime Context 注入用户和依赖。
- [ ] 用相同 thread id 延续状态，用不同 thread 验证隔离。
- [ ] 读取 state history，并从历史 checkpoint 创建分支。
- [ ] 用 update_state 修正状态后继续执行。
- [ ] 同一用户跨 thread 共享 Store 记忆。
- [ ] 不同用户的 Store 数据完全隔离。

## P2：Agent 与人工介入

- [ ] 构造有效 tool call，并关联 ToolMessage id。
- [ ] 模型节点在收到工具结果后结束循环。
- [ ] 高风险调用在执行前 interrupt。
- [ ] resume 后 START 节点没有重复执行。
- [ ] 审批可以批准、拒绝或修改参数。
- [ ] 重复恢复不会重复外部操作。

## P3：并行与可靠性

- [ ] 用 `Send` 按运行时列表 fan-out。
- [ ] 聚合并行结果，不依赖完成顺序。
- [ ] 将子流程封装为原生 subgraph。
- [ ] 输出 updates 和 custom stream 事件。
- [ ] 注入瞬时错误并验证 retry 次数。
- [ ] 为外部副作用设计业务幂等键。

## P4：综合项目

- [ ] 综合 Lab 的 FAQ、billing、refund approve/decline 全部运行。
- [ ] 完成至少两个 TODO 改造。
- [ ] 测试成功、失败、恢复和隔离路径。
- [ ] 写出 toy 与生产实现的差距。
- [ ] 能用 10 分钟讲清 state ownership 和故障恢复。
