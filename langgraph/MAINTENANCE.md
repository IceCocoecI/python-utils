# 维护约定

## 新增或修改章节

每章至少包含：

1. README：问题、心智模型、API、代码解读、常见坑、练习、自检。
2. 两个离线示例：`main()`、命令行入口、`--self-test`。
3. 版本边界：不使用 1.x 已弃用 API。
4. 至少一个失败模式，不只展示 happy path。
5. 根 README 或路线中的导航链接。

## 改动后验证

```bash
conda run -n langgraph python langgraph/scripts/smoke_test.py --warnings-as-errors
conda run -n langgraph python -m unittest discover -s langgraph/tests -v
conda run -n langgraph python langgraph/scripts/check_links.py
```

## 依赖原则

- 核心课程保持离线、确定性。
- provider/数据库依赖必须可选，不能在核心模块顶层 import。
- 学习 requirements 给兼容范围；生产用 lockfile。
- 升级 LangGraph 时，先运行 checkpoint、interrupt、tool、subgraph 和 Send 行为测试。

## 文档原则

- 理论文档讲机制、取舍和失败模式。
- 示例展示最小可观察机制。
- Lab 组合机制并提供参考实现和测试。
- 不把内存 mock、虚构 URL 或规则节点描述成“生产级”。
- 本地链接使用相对路径并通过检查脚本。
