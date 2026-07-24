# 理论自测题

## 基础

1. 节点为什么应该返回部分更新，而不是原地修改 state？
2. 一个字段没有 reducer 时采用什么语义？
3. 两个并行节点写同一字段为什么可能报错？
4. `add_messages` 与 `operator.add` 的差异是什么？
5. conditional edge 与 `Command(goto=...)` 各适合什么场景？
6. 循环除了模型主动停止外，还需要什么硬边界？

## 上下文与记忆

1. State、Runtime Context、checkpoint、Store、模型上下文窗口分别是什么生命周期？
2. `user_id` 与 `thread_id` 为什么不能等同？
3. 为什么 checkpointer 能形成短期记忆，却不是长期用户记忆？
4. Store namespace 至少要包含哪些隔离维度？
5. 为什么长期记忆需要来源、更新时间、删除和敏感信息策略？
6. 时间旅行为什么是“分叉”而不是外部世界的真正回滚？

## Agent 与可靠性

1. ToolNode 做了什么，又没有替你做什么？
2. 为什么 interrupt 前的副作用必须可重放？
3. 正确 resume 为什么要使用相同 thread id 和 `Command(resume=...)`？
4. checkpoint 为什么不能提供支付事务的 exactly-once？
5. RetryPolicy 适合重试哪些错误，不适合哪些错误？
6. 并行 `Send` 结果为什么必须定义 reducer，为什么不能依赖完成顺序？
7. 子图共享累积型 channel 时可能出现什么问题？
8. 在 LangGraph 1.0.6 中，直接注册 compiled graph 与 wrapper 内调用 `subgraph.invoke()` 是否都能产生嵌套 namespace？真正的选型差异是什么？
9. `Command(graph=Command.PARENT, goto=...)` 的目标在哪一层解析，`update` 又由哪一层的 reducer 合并？
10. handoff 共享 `messages` 时，为什么应该返回新增消息而不是复制完整历史？
11. 单 Agent + tools、router、supervisor、handoff 分别适合什么边界？

## 生产

1. 如何测试一个图，而不依赖真实模型的随机自然语言？
2. 图升级后旧 checkpoint 的 schema 怎么处理？
3. 哪些数据需要从 trace 中脱敏？
4. 中断任务积压需要哪些指标和过期策略？
5. 什么情况下普通 Python 函数比 LangGraph 更合适？
