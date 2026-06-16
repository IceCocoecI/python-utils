# AIGC Learning Routes

> `01` 到 `10` 是知识库；`routes/` 是学习路线图。
> 路线文档只负责告诉你“先学什么、跳过什么、做到什么程度”，正文仍然回到各模块阅读和实践。

---

## 路线选择

| 路线 | 适合人群 | 目标 |
|---|---|---|
| [practice-track](./practice-track.md) | 想尽快做出可运行项目 | 从基础脚本推进到微调、部署、RAG 和 profiling |
| [theory-track](./theory-track.md) | 想系统补原理、读论文、面试架构题 | 建立数据、模型、训练、推理、架构、GPU 的理论框架 |
| [full-linear-track](./full-linear-track.md) | 想完整按顺序学完全部模块 | 从基础到深水区的全量路线 |
| [role-llm-app-engineer](./role-llm-app-engineer.md) | LLM 应用 / RAG / Agent 方向 | 能构建可评估、可部署的 LLM 应用 |
| [role-finetuning-engineer](./role-finetuning-engineer.md) | 微调 / SFT / 对齐方向 | 能完成数据构造、LoRA/QLoRA、SFT、DPO 小闭环 |
| [role-inference-engineer](./role-inference-engineer.md) | 推理部署 / Serving / 性能方向 | 能理解 vLLM/SGLang、容量规划、服务化和 profiling |
| [role-research-foundation](./role-research-foundation.md) | 模型架构 / 论文 / 算法基础方向 | 能看懂主流 AIGC 模型架构和关键论文 |

---

## 推荐用法

1. 先选一条主路线，最多再选一条辅助路线。
2. 每学完一个阶段，回到 [assessments](../assessments/README.md) 做自检。
3. 实践优先的人按 [labs](../labs/README.md) 做项目，不要只读文档。
4. 理论优先的人也要跑最小示例，至少保证概念能落到代码。

---

## 环境约定

本仓库默认使用你的 `aigc` conda 环境运行示例：

```bash
conda run -n aigc python <script.py>
```

如果你已经在 shell 中执行过：

```bash
conda activate aigc
```

也可以直接运行：

```bash
python <script.py>
```
