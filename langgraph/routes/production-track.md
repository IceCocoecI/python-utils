# 生产工程专项路线

这条路线不教“如何接一个模型 API”，重点是长时间运行、有外部副作用的图怎样保持可恢复、可观测和可治理。

## 阶段 1：状态契约

阅读 02、03、04、10。

交付：

- state 字段 ownership、序列化格式、reducer 和 schema migration 表；
- Context 中的 tenant/auth/dependency 边界；
- 子图输入输出契约；
- 循环上限与预算策略。

## 阶段 2：持久化与记忆

阅读 05、06、08。

交付：

- checkpointer 后端选型、连接池、备份、恢复和迁移方案；
- thread id 和业务 id 映射；
- Store namespace、TTL、删除、加密、租户隔离；
- interrupt backlog、过期审批和重复 resume 策略。

## 阶段 3：副作用与可靠性

阅读 07、09 和综合 Lab 的幂等章节。

交付：

- 每个外部工具的权限、超时、retry、幂等键和补偿动作；
- checkpoint 与外部事务之间的故障矩阵；
- 限流、熔断和人工降级；
- 确定性离线测试与少量真实集成测试分层。

## 阶段 4：观测与发布

交付：

- request/thread/run/node/tool/operation 关联 id；
- latency、TTFT、node error、retry、interrupt age、token/cost 指标；
- PII 和 prompt/tool payload 脱敏；
- shadow/canary、图版本兼容和旧 checkpoint 处理方案；
- 回滚不依赖“删除 checkpoint”的发布预案。

上线门槛：恢复测试、重复副作用测试、跨租户隔离测试、负载测试和故障演练全部通过。
