import logging
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.llms.dashscope import DashScope
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import get_response_synthesizer
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Settings.embed_model = HuggingFaceEmbedding(model_name='all-MiniLM-L6-v2')
MODEL_NAME_ENUM  = {
    "qwen-plus": "qwen-plus",
    "qwen-turbo": "qwen-turbo",
    "qwen-max": "qwen-max"

  }
# 设置通义千问的 LLM 配置
Settings.llm = DashScope(
    api_key="sk-a9d33f951f7f4dc1a6901f984d4db27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model=MODEL_NAME_ENUM.get("qwen-max"),
)

# 加载文档
logger.info("Loading documents from directory.")
documents = SimpleDirectoryReader("./data").load_data()

# 初始化 ChromaDB 客户端
logger.info("Initializing ChromaDB client.")
db = chromadb.PersistentClient(path="./chroma_db")

# 创建集合
logger.info("Creating or getting Chroma collection.")
chroma_collection = db.get_or_create_collection(
    name="quickstart",
    metadata={"hnsw:space": "cosine", "embedding_dimensions": 384}  # 匹配嵌入模型维度
)

# 将 Chroma 作为向量存储
logger.info("Setting up Chroma as vector store.")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 创建索引
logger.info("Creating vector store index from documents.")
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# 配置检索器
logger.info("Configuring retriever.")
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# 配置响应合成器
logger.info("Configuring response synthesizer.")
response_synthesizer = get_response_synthesizer()

# 组装查询引擎
logger.info("Assembling query engine.")
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.3)],
)

# 查询
query_text = "根据文档内容，帮我整理福州平潭3天旅游攻略"
logger.info(f"Querying with text: {query_text}")
response = query_engine.query(query_text)
logger.info(f"Response: {response}")