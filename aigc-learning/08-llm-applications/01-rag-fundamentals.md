# 01 · RAG 基础

> RAG（Retrieval-Augmented Generation）是 LLM 应用的第一公民。
> 它让模型在回答时"先查资料再说话"，一举解决幻觉、知识截止和私有数据三大难题。
> 掌握 RAG 是 LLM 应用开发的必修课。

---

## 1. 为什么需要 RAG？

LLM 有三个根本限制：

| 限制 | 表现 | RAG 如何解决 |
|---|---|---|
| **幻觉** | 一本正经地编造不存在的事实 | 检索真实文档作为上下文，模型基于证据回答 |
| **知识截止** | 训练数据有截止日期 | 实时检索最新文档 |
| **私有数据** | 无法访问企业内部知识库 | 将私有文档索引后供模型检索 |

### 1.1 RAG vs 微调

```
场景                          推荐方案
─────────────────────────────────────────
需要最新/私有知识               → RAG
需要改变模型风格/格式           → 微调（SFT）
两者都需要                     → 微调 + RAG
知识库频繁更新                  → RAG（重建索引即可，无需重训）
```

---

## 2. RAG 流水线全景

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 文档加载  │───→│ 文本切分  │───→│ Embedding │───→│ 向量存储  │
│ Loading   │    │ Splitting │    │           │    │ Indexing  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                       │
                     ┌───────────────────────────────────┘
                     ↓
┌──────────┐    ┌──────────┐    ┌──────────┐
│ 用户提问  │───→│  检索     │───→│  生成     │
│ Query     │    │ Retrieval │    │ Generation│
└──────────┘    └──────────┘    └──────────┘
```

离线阶段（Indexing）：加载 → 切分 → 嵌入 → 存储
在线阶段（Querying）：用户提问 → 检索 → 拼接上下文 → LLM 生成

---

## 3. 文档加载

不同格式需要不同的解析器：

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
    CSVLoader,
    DirectoryLoader,
)

pdf_docs = PyPDFLoader("report.pdf").load()
html_docs = UnstructuredHTMLLoader("page.html").load()
md_docs = UnstructuredMarkdownLoader("notes.md").load()

all_docs = DirectoryLoader(
    "data/", glob="**/*.pdf", loader_cls=PyPDFLoader
).load()

print(f"加载了 {len(all_docs)} 个文档")
print(all_docs[0].page_content[:200])
print(all_docs[0].metadata)
```

常用加载器对比：

| 格式 | 推荐加载器 | 注意事项 |
|---|---|---|
| PDF | `PyPDFLoader` / `PyMuPDFLoader` | 表格和公式难解析，考虑 OCR |
| HTML | `UnstructuredHTMLLoader` | 需要去除导航栏等噪声 |
| Markdown | `UnstructuredMarkdownLoader` | 保留标题结构有利于切分 |
| 代码 | `TextLoader` + 语言解析器 | 按函数/类切分更合理 |
| 数据库 | 自定义 loader | SQL 查询后转文档 |

---

## 4. 文本切分

切分是 RAG 效果的关键——**切得太大检索不精准，切得太小丢失上下文**。

### 4.1 切分策略

```python
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
    RecursiveCharacterTextSplitter as CodeSplitter,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", ".", " ", ""],
)
chunks = splitter.split_documents(all_docs)
print(f"切分成 {len(chunks)} 个 chunk")
```

### 4.2 各策略对比

| 策略 | 原理 | 适用场景 |
|---|---|---|
| `CharacterTextSplitter` | 按固定字符数切 | 最简单，效果一般 |
| `RecursiveCharacterTextSplitter` | 按优先级分隔符递归切 | **通用首选** |
| `MarkdownHeaderTextSplitter` | 按 Markdown 标题切 | 技术文档 |
| `Language` 感知切分 | 按代码语法切（函数、类） | 代码库 |
| 语义切分 | 按 embedding 相似度切 | 效果最好，成本最高 |

### 4.3 Chunk Size 和 Overlap 的选择

```
chunk_size   效果
──────────────────────────────────
100-200      信息碎片化，缺失上下文
300-500      多数场景的甜区
500-1000     上下文更完整，但检索噪声增加
1000+        大块文本，检索精度下降

overlap 通常设为 chunk_size 的 10%–20%
```

实践建议：
- 先用 500 + 50 overlap 作为 baseline
- 根据评估指标调整
- 对结构化文档（Markdown/代码），优先用结构感知切分

---

## 5. Embedding 模型

Embedding 模型把文本转为向量，让语义相近的文本在向量空间中靠近。

### 5.1 常用模型

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
embeddings = model.encode(["你好世界", "Hello World"])
print(embeddings.shape)  # (2, 1024)

import numpy as np
similarity = np.dot(embeddings[0], embeddings[1]) / (
    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
)
print(f"余弦相似度: {similarity:.4f}")
```

### 5.2 模型对比

| 模型 | 维度 | 多语言 | 特点 |
|---|---|---|---|
| `text-embedding-3-small` (OpenAI) | 1536 | ✅ | 性价比高，API 调用 |
| `text-embedding-3-large` (OpenAI) | 3072 | ✅ | 精度最高的 API 方案 |
| `BAAI/bge-large-zh-v1.5` | 1024 | 中文优化 | 开源中文首选 |
| `BAAI/bge-m3` | 1024 | ✅ | 多语言 + dense/sparse/colbert 三合一 |
| `jinaai/jina-embeddings-v3` | 1024 | ✅ | 可变维度，任务感知 |
| `Cohere embed-v3` | 1024 | ✅ | 搜索/分类/聚类分模式 |

选择原则：
- 中文场景：优先 BGE 系列
- 多语言：BGE-M3 或 Jina v3
- 不想部署：OpenAI embedding API
- 参考 [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) 排名

### 5.3 使用 OpenAI Embedding

```python
from openai import OpenAI

client = OpenAI()
response = client.embeddings.create(
    input=["你好世界", "Hello World"],
    model="text-embedding-3-small",
)
vectors = [item.embedding for item in response.data]
print(f"维度: {len(vectors[0])}")
```

---

## 6. 检索策略

### 6.1 Dense Retrieval（稠密检索）

用 embedding 向量做近似最近邻搜索，这是 RAG 的默认方式。

### 6.2 Sparse Retrieval（BM25）

传统关键词检索，在精确术语匹配上仍然很强：

```python
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(chunks, k=5)
results = bm25.invoke("模型量化方法")
```

### 6.3 Hybrid Retrieval（混合检索）

组合 dense + sparse，取两者之长：

```python
from langchain.retrievers import EnsembleRetriever

dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
sparse_retriever = BM25Retriever.from_documents(chunks, k=5)

hybrid = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.6, 0.4],
)
results = hybrid.invoke("什么是 LoRA？")
```

| 方法 | 擅长 | 弱点 |
|---|---|---|
| Dense | 语义理解，同义词 | 精确术语、罕见词 |
| Sparse (BM25) | 精确匹配，专有名词 | 不理解语义 |
| Hybrid | 兼顾两者 | 需要调权重 |

---

## 7. Reranking（重排序）

检索返回 top-k 后，用 cross-encoder 重新打分，显著提升精度：

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

query = "什么是 RAG？"
passages = [doc.page_content for doc in results]

pairs = [[query, p] for p in passages]
scores = reranker.predict(pairs)

ranked = sorted(zip(scores, passages), reverse=True)
for score, text in ranked[:3]:
    print(f"[{score:.4f}] {text[:80]}...")
```

Reranker 对比：

| 模型 | 特点 |
|---|---|
| `BAAI/bge-reranker-v2-m3` | 开源多语言，效果好 |
| `Cohere Rerank` | API 服务，开箱即用 |
| `ms-marco-MiniLM-L-12-v2` | 轻量级英文 reranker |

典型流程：检索 top-20 → rerank → 取 top-5 → 送入 LLM。

---

## 8. Prompt 工程：给 LLM 注入检索结果

```python
RAG_PROMPT = """基于以下参考资料回答用户问题。
如果参考资料中没有相关信息，请明确说"我没有找到相关信息"，不要编造。
在回答中引用来源。

参考资料：
{context}

用户问题：{question}

回答："""

context = "\n\n".join([
    f"[来源 {i+1}] {doc.page_content}"
    for i, doc in enumerate(top_docs)
])
prompt = RAG_PROMPT.format(context=context, question=user_query)
```

关键原则：
- 明确指示模型基于给定上下文回答
- 要求模型在不确定时坦白承认
- 鼓励引用来源编号
- 控制上下文长度，不要超过模型的 context window

---

## 9. RAG 评估

### 9.1 Ragas 框架

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

eval_data = Dataset.from_dict({
    "question": ["什么是 RAG？"],
    "answer": ["RAG 是检索增强生成..."],
    "contexts": [["RAG 全称 Retrieval-Augmented Generation..."]],
    "ground_truth": ["RAG 是一种结合检索和生成的技术..."],
})

result = evaluate(eval_data, metrics=[
    faithfulness,        # 回答是否忠于检索到的上下文
    answer_relevancy,    # 回答是否与问题相关
    context_precision,   # 检索到的上下文是否精准
    context_recall,      # 检索是否覆盖了所需信息
])
print(result)
```

### 9.2 评估指标解读

| 指标 | 衡量什么 | 分数低说明 |
|---|---|---|
| Faithfulness | 回答是否基于上下文（不编造） | 模型在幻觉 |
| Answer Relevancy | 回答是否回应了问题 | 答非所问 |
| Context Precision | 检索结果中有用的比例 | 检索太多噪声 |
| Context Recall | 所需信息被检索到的比例 | 检索遗漏关键信息 |

---

## 10. 高级 RAG 技巧

### 10.1 查询变换

用户的原始查询往往不够精确，可以先改写：

```python
from langchain.retrievers import MultiQueryRetriever

multi_query = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm,
)
results = multi_query.invoke("RAG 效果不好怎么办？")
```

### 10.2 HyDE（Hypothetical Document Embeddings）

先让 LLM 生成一个"假想答案"，再用它的 embedding 去检索：

```python
from langchain.chains import HypotheticalDocumentEmbedder

hyde = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=embeddings,
    prompt_key="web_search",
)
hyde_embedding = hyde.embed_query("量子计算的应用场景")
```

### 10.3 Parent-Child Chunks

切分时保留父子关系：用小 chunk 做检索（精准），返回大 chunk 做上下文（完整）：

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=InMemoryStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(docs)
results = retriever.invoke("什么是 attention？")
```

---

## 11. 完整 RAG 流水线示例

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 加载文档
loader = DirectoryLoader("docs/", glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()

# 2. 切分
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. 嵌入 + 索引
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 4. 构建 RAG chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
    "基于以下上下文回答问题。如果不知道就说不知道。\n\n"
    "上下文：{context}\n\n问题：{question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. 查询
answer = chain.invoke("什么是注意力机制？")
print(answer)
```

---

## 12. 常见陷阱

| 陷阱 | 后果 | 解决方案 |
|---|---|---|
| chunk 太小（<100 字） | 信息碎片化，上下文不完整 | 至少 300–500 字 |
| chunk 太大（>2000 字） | 检索噪声大，精度下降 | 控制在 500–1000 字 |
| overlap = 0 | 跨 chunk 的信息丢失 | 设 10%–20% overlap |
| embedding 模型和检索语言不匹配 | 中文文档用英文模型效果差 | 选择匹配语言的模型 |
| 上下文塞满 context window | 超出窗口的内容被截断 | 限制 top-k，用 reranker 精选 |
| 不做评估直接上线 | 不知道效果好不好 | 用 Ragas 建立评估集 |
| 只用 dense retrieval | 精确术语检索差 | 加 BM25 做 hybrid |
| PDF 解析质量差 | 切分后全是乱码碎片 | 换更好的解析器（如 `unstructured`） |
| 忽略 metadata | 无法做筛选和溯源 | 保留来源、页码等 metadata |

---

## 13. 本章小结

```
RAG = Retrieval + Augmented + Generation

核心原则：
1. 切分决定检索质量 → 选对切分策略
2. Embedding 决定语义匹配 → 选对模型
3. Retrieval 决定上下文 → dense + sparse + rerank
4. Prompt 决定生成质量 → 清晰指令 + 引用来源
5. 评估决定迭代方向 → Ragas 四大指标
```

下一章我们深入向量数据库——RAG 系统的"心脏"。
