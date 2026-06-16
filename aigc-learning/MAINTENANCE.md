# Maintenance Guide

> 维护目标：让学习路线、示例代码、链接和环境说明长期保持可用。

---

## 更新原则

- 优先保持 toy 示例 CPU 可跑，不把真实模型下载作为默认路径。
- 新增外部依赖前，先判断是否能用标准库、NumPy 或已有依赖解决。
- 真实模型、GPU 推理引擎、Triton/CUDA 内容放在可选路径。
- 路线文档只做导航，正文内容放在模块文档或 Lab 文档。
- 修改路径后必须跑链接检查。

---

## 每次改动前检查

```bash
git status --short
```

确认没有误改运行产物：

```bash
find aigc-learning -type f \( -name '*.pyc' -o -name '*.pt' -o -name '*.png' -o -name '*.jpg' \)
```

---

## 每次改动后验证

链接检查：

```bash
conda run -n aigc python aigc-learning/scripts/check_links.py
```

核心 smoke tests：

```bash
conda run -n aigc python aigc-learning/08-llm-applications/examples/toy_rag.py --self-test
conda run -n aigc python aigc-learning/10-cuda-and-triton/examples/torch_profiler_demo.py --steps 3
```

可选：模块 07 smoke test：

```bash
conda run -n aigc python aigc-learning/07-inference-and-deployment/examples/smoke_test.py
```

---

## 文档更新策略

### 路线文档

更新场景：

- 新增模块。
- 学习顺序发生变化。
- 某条岗位路线需要调整优先级。

检查点：

- [ ] 是否链接到正确模块。
- [ ] 是否给出明确产出。
- [ ] 是否避免复制正文。

### Lab 文档

更新场景：

- 新增示例脚本。
- 任务步骤变更。
- 验收标准变化。

检查点：

- [ ] 是否有起步命令。
- [ ] 是否能在 `aigc` 环境运行。
- [ ] 是否有 TODO 和报告模板入口。

### 示例代码

更新场景：

- 新增 runnable demo。
- 依赖或 API 变化。
- 修复运行失败。

检查点：

- [ ] 默认不下载外部模型。
- [ ] 默认 CPU 可跑，GPU 可选。
- [ ] 有 `--self-test` 或足够明确的成功输出。
- [ ] 输出写到 `examples/outputs/` 或被 `.gitignore` 忽略。

---

## 版本更新节奏

建议每月检查一次：

- PyTorch 和 CUDA 推荐安装命令。
- Transformers / Diffusers / Accelerate / PEFT 的最低版本。
- vLLM / SGLang 兼容性。
- LangChain / LlamaIndex / LangGraph API 变化。
- Triton 与 PyTorch 版本绑定关系。

更新记录可以写在提交信息或单独 changelog 中。

