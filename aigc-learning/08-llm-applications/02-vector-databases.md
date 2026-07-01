# 02 · 向量数据库

> 向量数据库是 RAG 系统的心脏——没有高效的语义检索，一切都是空谈。
> 本节从原理到实战，覆盖 FAISS、Milvus、Chroma、Qdrant、pgvector 五大方案。
> 生产选型不只看“能存多少向量”，还要看过滤、更新、隔离、备份、观测和运维成本。

---

## 1. 什么是向量数据库？

传统数据库按精确值检索（`WHERE name = 'Alice'`），向量数据库按**语义相似度**检索——
"找到和这段话意思最接近的 5 条记录"。

```
传统检索：  关键词匹配 → "苹果" 只能找到包含"苹果"的文档
语义检索：  向量相似度 → "苹果" 能找到包含"iPhone""水果""Apple"的文档
```

核心能力：
- 存储高维向量（通常 768–3072 维）
- 毫秒级近似最近邻（ANN）搜索
- 支持 metadata 过滤
- 支持增删改查

### 1.1 向量数据库在 RAG 里真正负责什么？

| 能力 | 为什么重要 |
|---|---|
| ANN 搜索 | 大规模向量下低延迟召回 |
| metadata 过滤 | 租户、权限、语言、时间、文档类型 |
| upsert/delete | 文档更新、删除、合规要求 |
| 多向量/混合检索 | 同一文档可有 dense、sparse、title、body 多路表示 |
| 分片/副本 | 容量、可用性和吞吐 |
| 备份恢复 | 索引损坏或误删后可恢复 |
| 观测指标 | 查询延迟、召回、过滤命中、索引构建状态 |

教学 demo 可以只存向量，生产系统必须把 payload/metadata 当成同等重要的数据。

---

## 2. 向量相似度度量

### 2.1 三种常用距离

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

# L2 距离（欧氏距离）——越小越相似
l2 = np.linalg.norm(a - b)

# 余弦相似度——越大越相似（-1 到 1）
cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 内积（Inner Product）——越大越相似
ip = np.dot(a, b)

print(f"L2: {l2:.4f}, Cosine: {cosine:.4f}, IP: {ip:.4f}")
```

### 2.2 如何选择

| 度量 | 何时使用 | 注意 |
|---|---|---|
| Cosine | 文本 embedding（最常用） | 大多数 embedding 模型已归一化 |
| L2 | 图像特征、未归一化向量 | 对向量长度敏感 |
| Inner Product | 已归一化的向量 | 归一化后等价于 cosine |

> **经验法则**：如果 embedding 模型输出已归一化（如 OpenAI、BGE），用 cosine 或 IP 都行。

### 2.3 距离度量必须和模型约定一致

入库前要固定三件事：

```text
embedding_model + normalize_strategy + distance_metric
```

任何一项变化都应视为新索引版本。否则你会遇到“代码没报错，但检索质量突然变差”的隐性故障。

---

## 3. FAISS：Meta 的向量检索库

FAISS 不是数据库，是一个**高性能向量检索库**——纯内存、无持久化、无 API 服务。
适合中小规模（百万级以下）或嵌入到应用中的场景。

### 3.1 基础用法

```python
import faiss
import numpy as np

d = 1024       # 向量维度
n = 100000     # 向量数量
nq = 5         # 查询数量

np.random.seed(42)
xb = np.random.randn(n, d).astype("float32")   # 数据库向量
xq = np.random.randn(nq, d).astype("float32")  # 查询向量

# IndexFlatL2：暴力搜索，精确但慢
index = faiss.IndexFlatL2(d)
index.add(xb)
print(f"索引中有 {index.ntotal} 条向量")

D, I = index.search(xq, k=5)  # D: 距离, I: 索引
print(f"最近邻索引: {I[0]}")
print(f"对应距离:   {D[0]}")
```

### 3.2 常用索引类型

```python
# IVF: 倒排索引，先聚类再在簇内搜索
nlist = 100  # 聚类数
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
index_ivf.train(xb)
index_ivf.add(xb)
index_ivf.nprobe = 10  # 搜索时访问的簇数（精度/速度平衡）

# HNSW: 基于图的索引，精度高、速度快
index_hnsw = faiss.IndexHNSWFlat(d, 32)  # 32 = M 参数
index_hnsw.add(xb)

# PQ: 乘积量化，大幅压缩内存
m = 8  # 子空间数
index_pq = faiss.IndexPQ(d, m, 8)  # 每子空间 8 bits
index_pq.train(xb)
index_pq.add(xb)
```

### 3.3 索引类型对比

| 索引 | 时间复杂度 | 内存 | 精度 | 适用场景 |
|---|---|---|---|---|
| `IndexFlatL2` | O(n·d) | 最大 | 100% | 小数据集（<10 万） |
| `IndexIVFFlat` | O(nprobe·n/nlist·d) | 中 | 高 | 中等数据集 |
| `IndexHNSWFlat` | O(log n) | 大 | 极高 | 追求精度和速度 |
| `IndexPQ` | O(n·m) | **最小** | 中 | 内存受限的大数据集 |
| `IndexIVFPQ` | 更快 | 最小 | 中 | 亿级数据集 |

### 3.4 GPU 加速

```python
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # 0 = GPU 编号
D, I = gpu_index.search(xq, k=5)
```

### 3.5 保存与加载

```python
faiss.write_index(index, "my_index.faiss")
index = faiss.read_index("my_index.faiss")
```

---

## 4. Milvus：分布式向量数据库

Milvus 是目前最成熟的开源向量数据库，支持亿级规模、分布式部署、混合检索。

### 4.1 架构概览

```
┌──────────────────────────────────────────┐
│               Milvus Cluster              │
│  ┌──────┐  ┌──────┐  ┌──────┐           │
│  │Proxy │  │Query │  │Index │           │
│  │Node  │  │Node  │  │Node  │           │
│  └──────┘  └──────┘  └──────┘           │
│          ↕            ↕                   │
│    ┌──────────────────────┐              │
│    │    Object Storage    │  (MinIO/S3)  │
│    └──────────────────────┘              │
│    ┌──────────────────────┐              │
│    │    Meta Storage      │  (etcd)      │
│    └──────────────────────┘              │
└──────────────────────────────────────────┘
```

### 4.2 快速上手

```bash
pip install pymilvus
```

```python
from pymilvus import MilvusClient

client = MilvusClient("milvus_demo.db")  # 轻量模式，数据存本地文件

client.create_collection(
    collection_name="docs",
    dimension=1024,
    metric_type="COSINE",
)

import numpy as np
vectors = np.random.randn(1000, 1024).tolist()
data = [
    {"id": i, "vector": vectors[i], "text": f"文档 {i}", "category": "tech"}
    for i in range(1000)
]
client.insert(collection_name="docs", data=data)

results = client.search(
    collection_name="docs",
    data=[vectors[0]],
    limit=5,
    output_fields=["text", "category"],
)
for hit in results[0]:
    print(f"ID: {hit['id']}, Distance: {hit['distance']:.4f}, Text: {hit['entity']['text']}")
```

### 4.3 混合检索（Dense + Sparse）

```python
from pymilvus import AnnSearchRequest, RRFRanker

dense_req = AnnSearchRequest(
    data=[dense_vector],
    anns_field="dense_vector",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=20,
)
sparse_req = AnnSearchRequest(
    data=[sparse_vector],
    anns_field="sparse_vector",
    param={"metric_type": "IP"},
    limit=20,
)

results = client.hybrid_search(
    collection_name="docs",
    reqs=[dense_req, sparse_req],
    ranker=RRFRanker(),  # Reciprocal Rank Fusion
    limit=5,
)
```

---

## 5. Chroma：轻量级嵌入式向量数据库

Chroma 的哲学是"开箱即用"——不需要任何服务进程，直接嵌入应用。

```python
import chromadb

client = chromadb.Client()  # 内存模式
# client = chromadb.PersistentClient(path="./chroma_data")  # 持久化

collection = client.create_collection(
    name="my_docs",
    metadata={"hnsw:space": "cosine"},
)

collection.add(
    documents=["RAG 是检索增强生成", "LLM 可以理解自然语言", "向量数据库存储 embedding"],
    ids=["doc1", "doc2", "doc3"],
    metadatas=[
        {"source": "wiki", "year": 2024},
        {"source": "paper", "year": 2023},
        {"source": "blog", "year": 2024},
    ],
)

results = collection.query(
    query_texts=["什么是 RAG？"],
    n_results=2,
    where={"year": {"$gte": 2024}},  # metadata 过滤
)
print(results["documents"])
print(results["distances"])
```

Chroma 内置了 embedding 功能（默认用 `all-MiniLM-L6-v2`），你传入文本它自动 embed。

---

## 6. Qdrant：功能丰富的向量数据库

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

client = QdrantClient(":memory:")  # 内存模式

client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

points = [
    PointStruct(
        id=i,
        vector=np.random.randn(1024).tolist(),
        payload={"text": f"文档 {i}", "category": "tech", "year": 2024},
    )
    for i in range(100)
]
client.upsert(collection_name="docs", points=points)

results = client.query_points(
    collection_name="docs",
    query=np.random.randn(1024).tolist(),
    limit=5,
    query_filter=Filter(
        must=[FieldCondition(key="category", match=MatchValue(value="tech"))]
    ),
)
for point in results.points:
    print(f"ID: {point.id}, Score: {point.score:.4f}")
```

Qdrant 的亮点：
- 强大的 payload 过滤（嵌套 JSON、地理位置、范围查询）
- 支持命名向量（同一条记录多个向量字段）
- Rust 编写，性能优秀

---

## 7. pgvector：PostgreSQL 的向量扩展

如果你已有 PostgreSQL，pgvector 让你无需引入新系统：

```sql
-- 启用扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 建表
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1024),
    metadata JSONB
);

-- 建索引
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 插入
INSERT INTO documents (content, embedding, metadata)
VALUES ('RAG 是检索增强生成', '[0.1, 0.2, ...]', '{"source": "wiki"}');

-- 查询（余弦距离）
SELECT content, 1 - (embedding <=> '[0.1, 0.2, ...]') AS similarity
FROM documents
WHERE metadata->>'source' = 'wiki'
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 5;
```

```python
import psycopg2

conn = psycopg2.connect("postgresql://user:pass@localhost/mydb")
cur = conn.cursor()
cur.execute(
    "SELECT content, 1 - (embedding <=> %s::vector) AS similarity "
    "FROM documents ORDER BY embedding <=> %s::vector LIMIT 5",
    (str(query_vector.tolist()), str(query_vector.tolist())),
)
for row in cur.fetchall():
    print(f"相似度: {row[1]:.4f}, 内容: {row[0][:80]}")
```

---

## 8. 方案对比

| 特性 | FAISS | Milvus | Chroma | Qdrant | pgvector |
|---|---|---|---|---|---|
| 类型 | 库 | 数据库 | 嵌入式 DB | 数据库 | PG 扩展 |
| 部署 | 嵌入进程 | 独立集群 | 嵌入/CS | 独立服务 | PG 内 |
| 规模 | 亿级 | 百亿级 | 百万级 | 亿级 | 千万级 |
| 持久化 | 手动 | ✅ | ✅ | ✅ | ✅ |
| 分布式 | ❌ | ✅ | ❌ | ✅ | PG 主从 |
| Metadata 过滤 | ❌ | ✅ | ✅ | ✅（强） | ✅（SQL） |
| 混合检索 | ❌ | ✅ | ❌ | ❌ | ❌ |
| GPU 支持 | ✅ | ✅ | ❌ | ❌ | ❌ |
| 语言 | C++/Python | Go/C++ | Python | Rust | C |
| 适合场景 | 原型/嵌入 | 生产大规模 | 快速原型 | 中大规模 | 已有 PG |

> 表格是学习用概览。各项目能力更新很快，具体是否支持 hybrid、多向量、量化、分布式、云服务和过滤语法，以当前官方文档为准。

### 如何选择？

```
你的数据量是多少？
├── < 100 万 & 快速原型 → Chroma
├── < 100 万 & 需要嵌入应用 → FAISS
├── < 1000 万 & 已有 PostgreSQL → pgvector
├── < 1000 万 & 需要强过滤 → Qdrant
└── > 1000 万 / 需要分布式 → Milvus
```

### 8.1 生产选型矩阵

| 约束 | 优先选择 |
|---|---|
| 单机 demo、课程实验 | Chroma / FAISS |
| Python 进程内检索、无需服务 | FAISS |
| 已有 PostgreSQL、数据量中小、团队熟 SQL | pgvector |
| 需要强 payload 过滤、简单运维 | Qdrant |
| 需要大规模分布式、混合检索、企业级扩展 | Milvus |
| 多租户强隔离 | 独立 collection / database / index，避免只靠后置过滤 |
| 频繁更新删除 | 关注 upsert/delete、compaction、索引重建成本 |
| 合规审计 | 关注备份、删除证明、访问日志和数据驻留 |

不要只按向量数量选库。RAG 线上问题很多来自过滤、更新和权限，而不是 ANN 本身。

---

## 9. 索引类型详解

### 9.1 Flat（暴力搜索）

逐一计算距离，100% 精确但 O(n) 复杂度。适合小数据集或作为 baseline。

### 9.2 IVF（Inverted File Index）

```
训练阶段：用 K-Means 把向量聚成 nlist 个簇
查询阶段：先找最近的 nprobe 个簇，只在这些簇内搜索

nprobe ↑ → 精度 ↑，速度 ↓
nprobe = nlist → 等价于暴力搜索
```

### 9.3 HNSW（Hierarchical Navigable Small World）

```
              Layer 2:    A ─────── B
                          │         │
              Layer 1:    A ── C ── B ── D
                          │   │    │   │
              Layer 0:    A ─ C ─ E ─ B ─ D ─ F ─ G

多层图结构，从上层"跳跃"到下层精搜
M: 每层连接数（大 → 精度高，内存大）
efConstruction: 建图时的搜索宽度
efSearch: 查询时的搜索宽度
```

- 优点：查询速度极快，精度极高
- 缺点：内存占用最大，建图慢

### 9.4 PQ（Product Quantization）

把高维向量切成 m 个子向量，每个子向量用码本压缩成 1 字节。
1024 维 float32 向量（4096 字节）→ 压缩到 m 字节（如 64 字节），压缩 64 倍。

### 9.5 ScaNN（Google）

结合各向量量化技术，Google 内部大规模使用。开源实现：`google-research/scann`。

---

## 10. 实战：用 FAISS 构建语义搜索系统

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

documents = [
    "RAG 是检索增强生成技术，结合了信息检索和文本生成",
    "向量数据库用于存储和检索高维向量",
    "FAISS 是 Meta 开发的高效向量检索库",
    "LangChain 是一个 LLM 应用开发框架",
    "Transformer 架构基于自注意力机制",
    "LoRA 是一种参数高效的微调方法",
    "vLLM 是高性能 LLM 推理引擎",
    "Embedding 模型将文本转换为稠密向量表示",
]

embeddings = model.encode(documents, normalize_embeddings=True)
d = embeddings.shape[1]

index = faiss.IndexFlatIP(d)  # 归一化后用内积等价于余弦
index.add(embeddings.astype("float32"))

query = "什么技术可以让大模型查找资料？"
query_vec = model.encode([query], normalize_embeddings=True).astype("float32")

D, I = index.search(query_vec, k=3)
print("查询:", query)
for i, (dist, idx) in enumerate(zip(D[0], I[0])):
    print(f"  Top-{i+1} [{dist:.4f}]: {documents[idx]}")
```

输出示例：
```
查询: 什么技术可以让大模型查找资料？
  Top-1 [0.7823]: RAG 是检索增强生成技术，结合了信息检索和文本生成
  Top-2 [0.6541]: 向量数据库用于存储和检索高维向量
  Top-3 [0.6102]: Embedding 模型将文本转换为稠密向量表示
```

---

## 11. 实际工程考量

### 11.1 维度选择

| 维度 | 内存（100 万条） | 适用 |
|---|---|---|
| 384 | ~1.5 GB | 轻量级场景 |
| 768 | ~3.0 GB | 通用 |
| 1024 | ~4.0 GB | 高精度 |
| 3072 | ~12.0 GB | OpenAI large |

### 11.2 Scaling 策略

```
数据量增长 → 内存不够 → 怎么办？
├── PQ 压缩（牺牲少量精度）
├── IVF 分区（减少搜索范围）
├── 磁盘索引（FAISS OnDisk / Milvus DiskANN）
├── 分片（Milvus / Qdrant 自动分片）
└── 降维（PCA / Matryoshka embedding）
```

### 11.3 索引生命周期

生产向量库要管理索引版本：

```text
raw_document_version
parser_version
chunker_version
embedding_model_version
index_version
created_at
```

常见流程：

1. 新文档进入 staging index。
2. 跑离线评估集，对比旧索引。
3. 通过后切换线上 alias。
4. 保留旧索引用于回滚。
5. 删除过期索引并记录审计。

不要直接在生产索引上“边改边试”，否则很难解释质量波动。

---

## 12. 常见陷阱

| 陷阱 | 后果 | 解决方案 |
|---|---|---|
| 向量未归一化就用 IP 距离 | 检索结果偏向长向量 | 入库前 L2 归一化，或用 cosine |
| 索引类型和数据量不匹配 | HNSW 用于亿级数据 OOM | 大数据用 IVF-PQ |
| nprobe 设太小 | IVF 召回率低 | 从 nlist/10 开始调 |
| 不做持久化 | 进程重启索引丢失 | FAISS 手动保存，或用真正的数据库 |
| 混用不同 embedding 模型 | 索引和查询的向量空间不一致 | 全链路用同一个模型 |
| 忽略 metadata 过滤 | 检索出不相关的文档 | 用 Milvus/Qdrant 的过滤功能 |
| 一次性加载所有数据到内存 | 启动慢、OOM | 分批加载，或用内存映射 |
| 只做后置权限过滤 | 召回质量差，甚至泄漏风险 | 检索前过滤或租户隔离 |
| 无备份和回滚 | 索引损坏后只能重建 | snapshot、alias、蓝绿切换 |
| 不监控空结果率 | 用户看到“没找到”但没人发现 | 监控 empty retrieval rate |
| 删除文档不删向量 | 合规和隐私风险 | delete + compaction + 审计 |

---

## 13. 实践任务：压测一个向量库方案

任选 FAISS、Qdrant、Milvus、Chroma 或 pgvector，记录：

```text
向量数量：
向量维度：
距离度量：
索引类型：
metadata 字段：
过滤条件：
top-k：
QPS：
p50/p95 latency：
内存/磁盘占用：
更新和删除耗时：
备份恢复方式：
```

至少做 4 组实验：

| 实验 | 观察点 |
|---|---|
| Flat vs HNSW/IVF | 召回、延迟、内存 |
| 有过滤 vs 无过滤 | 延迟和召回变化 |
| top-k 5/20/100 | 延迟、rerank 成本 |
| 10 万 / 100 万向量 | 扩展趋势 |

---

## 14. 本章小结

```
选库决策树：
  快速原型 → Chroma
  嵌入式 / 离线 → FAISS
  已有 PG → pgvector
  生产级 / 需过滤 → Qdrant 或 Milvus
  亿级分布式 → Milvus

核心概念：
  1. 距离度量：cosine（文本）、L2（图像）、IP（归一化后）
  2. 索引选择：Flat（精确）→ IVF（中等）→ HNSW（快速）→ PQ（压缩）
  3. 工程要点：归一化、持久化、分片、metadata 过滤
```

下一章我们学习如何用编排框架把向量检索、LLM 调用、工具串成完整工作流。
