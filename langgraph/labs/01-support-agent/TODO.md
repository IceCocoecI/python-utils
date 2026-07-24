# Lab 01 改造任务

不要直接在唯一一份参考实现上做大改。建议创建自己的 Lab 目录，复用测试思路。

## P0：读懂与复现

- [ ] 运行 FAQ、billing、低额 refund、高额 refund 批准/拒绝五条路径。
- [ ] 画出 state 字段的写入节点和 reducer。
- [ ] 从 stream 输出指出 interrupt 前最后一个成功节点。
- [ ] 用 `get_state_history()` 找到暂停 checkpoint。

## P1：单模块改造

- [ ] 增加 `technical` intent 和一个独立子图。
- [ ] 为账单调查增加运行时可配置的 checks 列表。
- [ ] 给长期案件记忆增加 `created_at`、来源和删除函数。
- [ ] 支持人工审批时修改退款金额，而不只返回布尔值。
- [ ] 给 router、reducer 和金额解析分别写纯函数测试。

## P2：综合改造

- [ ] 将 FAQ 规则替换成离线 toy retriever，保留来源字段。
- [ ] 增加一个会失败两次再成功的 I/O 节点，并配置 RetryPolicy。
- [ ] 将重复退款测试扩展为“checkpoint 前崩溃后恢复”。
- [ ] 增加 supervisor，把 billing 和 refund 专家封装成子图。
- [ ] 增加消息摘要字段，区分 checkpoint 消息历史与模型输入窗口。

## P3：生产设计，不要求连接真实服务

- [ ] 写出 Postgres checkpointer/store 的迁移与回滚方案。
- [ ] 为审批 payload、消息和 trace 设计脱敏规则。
- [ ] 设计退款 API 的幂等键、outbox 和补偿流程。
- [ ] 定义 latency、interrupt backlog、retry、tool error 指标。
- [ ] 写一份 threat model：越权工具调用、prompt injection、跨租户记忆泄漏。

## 完成标准

每个改造必须包含：需求说明、状态/schema 变化、运行命令、至少一个成功测试、至少一个失败或边界测试、已知局限。
