"""
StateGraph 构建模块

定义和编译多模态内容生成 Agent 的工作流图。
"""

from langgraph.graph import StateGraph, START, END

from multimodal_agent.logging_config import get_logger
from multimodal_agent.state import AgentState
from multimodal_agent.nodes.router import router_node, route_to_worker
from multimodal_agent.nodes.text_worker import text_worker_node
from multimodal_agent.nodes.image_worker import image_worker_node
from multimodal_agent.nodes.video_worker import video_worker_node
from multimodal_agent.nodes.error_handler import error_handler_node

logger = get_logger(__name__)


def create_multimodal_agent() -> StateGraph:
    """
    创建多模态内容生成 Agent
    
    构建包含以下节点的 StateGraph：
    - router: 意图识别和路由决策
    - text_worker: 文本生成
    - image_worker: 图片生成
    - video_worker: 视频生成
    - error_handler: 错误处理
    
    工作流程：
    ```
    START
      │
      ▼
    router (意图识别)
      │
      ├── task_type=text ──────► text_worker ───┐
      │                                         │
      ├── task_type=image ─────► image_worker ──┤
      │                                         │
      ├── task_type=video ─────► video_worker ──┤
      │                                         │
      └── task_type=unknown ───► error_handler ─┤
                                                │
                                                ▼
                                               END
    ```
    
    Returns:
        StateGraph: 编译后的工作流图
    """
    logger.info("building_multimodal_agent_graph")
    
    # 创建状态图构建器
    builder = StateGraph(AgentState)
    
    # ============================================
    # Step 1: 添加节点
    # ============================================
    
    # Router 节点 - 负责意图识别
    builder.add_node("router", router_node)
    
    # Worker 节点 - 负责具体内容生成
    builder.add_node("text_worker", text_worker_node)
    builder.add_node("image_worker", image_worker_node)
    builder.add_node("video_worker", video_worker_node)
    
    # Error Handler 节点 - 处理错误情况
    builder.add_node("error_handler", error_handler_node)
    
    logger.debug("nodes_added", nodes=["router", "text_worker", "image_worker", "video_worker", "error_handler"])
    
    # ============================================
    # Step 2: 添加边（连接节点）
    # ============================================
    
    # 入口：从 START 到 Router
    builder.add_edge(START, "router")
    
    # 条件边：从 Router 根据任务类型路由到不同的 Worker
    builder.add_conditional_edges(
        source="router",
        path=route_to_worker,
        path_map={
            "text_worker": "text_worker",
            "image_worker": "image_worker",
            "video_worker": "video_worker",
            "error_handler": "error_handler",
        },
    )
    
    # Worker 完成后结束
    builder.add_edge("text_worker", END)
    builder.add_edge("image_worker", END)
    builder.add_edge("video_worker", END)
    builder.add_edge("error_handler", END)
    
    logger.debug("edges_added")
    
    # ============================================
    # Step 3: 编译图
    # ============================================
    
    compiled_graph = builder.compile()
    
    logger.info("multimodal_agent_graph_compiled")
    
    return compiled_graph


def visualize_graph(graph: StateGraph, output_path: str | None = None) -> bytes | None:
    """
    可视化工作流图
    
    Args:
        graph: 编译后的图
        output_path: 可选的输出文件路径
        
    Returns:
        bytes | None: PNG 图片数据，如果可视化失败则返回 None
    """
    try:
        # 生成 Mermaid 格式的图
        mermaid_png = graph.get_graph(xray=True).draw_mermaid_png()
        
        if output_path:
            with open(output_path, "wb") as f:
                f.write(mermaid_png)
            logger.info("graph_visualization_saved", path=output_path)
        
        return mermaid_png
        
    except Exception as e:
        logger.warning(
            "graph_visualization_failed",
            error=str(e),
        )
        return None


def print_graph_structure(graph: StateGraph) -> str:
    """
    打印图结构（文本格式）
    
    Args:
        graph: 编译后的图
        
    Returns:
        str: 图结构的文本表示
    """
    structure = """
╔══════════════════════════════════════════════════════════════╗
║         Multimodal Content Generation Agent                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║    ┌─────────┐                                               ║
║    │  START  │                                               ║
║    └────┬────┘                                               ║
║         │                                                    ║
║         ▼                                                    ║
║    ┌─────────────────────────────────────┐                   ║
║    │            ROUTER                   │                   ║
║    │     (Intent Recognition)            │                   ║
║    │  - Analyze user input               │                   ║
║    │  - Classify task type               │                   ║
║    │  - Route to appropriate worker      │                   ║
║    └───┬──────────┬──────────┬──────┬───┘                   ║
║        │          │          │      │                        ║
║        │ TEXT     │ IMAGE    │VIDEO │ UNKNOWN               ║
║        ▼          ▼          ▼      ▼                        ║
║   ┌─────────┐┌─────────┐┌─────────┐┌─────────┐              ║
║   │  TEXT   ││  IMAGE  ││  VIDEO  ││  ERROR  │              ║
║   │ WORKER  ││ WORKER  ││ WORKER  ││ HANDLER │              ║
║   └────┬────┘└────┬────┘└────┬────┘└────┬────┘              ║
║        │          │          │          │                    ║
║        └──────────┴──────────┴──────────┘                    ║
║                       │                                      ║
║                       ▼                                      ║
║                  ┌─────────┐                                 ║
║                  │   END   │                                 ║
║                  └─────────┘                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    return structure
