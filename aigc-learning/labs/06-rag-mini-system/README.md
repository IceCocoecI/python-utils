# Lab 06：RAG Mini System

> 目标：搭建一个本地知识库问答系统，覆盖文档切分、embedding、索引、检索、生成和评估。

---

## 前置知识

- [../../08-llm-applications/01-rag-fundamentals.md](../../08-llm-applications/01-rag-fundamentals.md)
- [../../08-llm-applications/02-vector-databases.md](../../08-llm-applications/02-vector-databases.md)
- [../../08-llm-applications/03-orchestration-frameworks.md](../../08-llm-applications/03-orchestration-frameworks.md)
- [../../08-llm-applications/04-agent-engineering.md](../../08-llm-applications/04-agent-engineering.md)
- [../../07-inference-and-deployment/03-serving-frameworks.md](../../07-inference-and-deployment/03-serving-frameworks.md)

---

## 建议数据

优先使用本仓库已有 Markdown 文档作为知识库，例如：

```text
aigc-learning/01-python-foundations/*.md
aigc-learning/02-deep-learning-libraries/*.md
aigc-learning/08-llm-applications/*.md
```

这样不需要额外下载数据，也便于人工判断答案是否正确。

---

## 起步脚本

先跑一个不依赖外部模型的 toy RAG，确认流程：

```bash
conda run -n aigc python aigc-learning/08-llm-applications/examples/toy_rag.py --self-test
```

然后再把 toy embedding 替换成真实 embedding 模型，把 toy synthesizer 替换成真实 LLM。

---

## 任务

1. 实现文档读取：遍历若干 `.md` 文件，保留文件路径和标题作为 metadata。
2. 实现文本切分：比较至少两组 `chunk_size` 和 `overlap`。
3. 生成 embedding：
   - 可用本地 embedding 模型。
   - 如果没有模型，可以先用 toy embedding 或 TF-IDF/BM25 思路验证流程。
4. 建立索引：FAISS、Chroma 或轻量内存索引均可。
5. 实现检索：输入问题，返回 top-k chunks 和 metadata。
6. 实现生成：把检索结果拼进 prompt，调用本地或远程 LLM。
7. 准备 10 个测试问题，记录命中情况和答案质量。
8. 写出失败样例和改进方向。

---

## 验收标准

- [ ] 每条回答能追溯到引用的 chunk 或文件。
- [ ] 能解释 chunk size 和 overlap 的取舍。
- [ ] 能说明 dense、sparse、hybrid retrieval 的差异。
- [ ] 能识别至少 3 个失败样例。
- [ ] 能给出改进方案：rerank、query rewrite、metadata filter、prompt 约束、评估集。

---

## 延伸挑战

- 增加 reranker。
- 增加 hybrid retrieval。
- 增加 RAG 评估指标。
- 增加 prompt injection 防护规则。
- 把系统包装成 FastAPI 服务。
