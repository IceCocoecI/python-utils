"""
漫剧工作流图

支持两种模式：
1. 自动模式 (auto): 一键式生成，无需用户干预
2. 交互模式 (interactive): 在关键节点暂停，等待用户选择

使用 LangGraph 的 interrupt 和 checkpointer 实现 Human-in-the-loop。
"""

from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from multimodal_agent.logging_config import get_logger
from multimodal_agent.comic_workflow.state import (
    ComicState,
    WorkflowStatus,
    InteractionPoint,
)
from multimodal_agent.comic_workflow.nodes.script_generator import script_generator_node
from multimodal_agent.comic_workflow.nodes.scene_parser import scene_parser_node
from multimodal_agent.comic_workflow.nodes.image_generator import image_generator_node
from multimodal_agent.comic_workflow.nodes.audio_generator import audio_generator_node
from multimodal_agent.comic_workflow.nodes.video_composer import video_composer_node
from multimodal_agent.comic_workflow.nodes.interaction_handler import (
    image_selection_node,
)

logger = get_logger(__name__)


def check_error(state: ComicState) -> Literal["continue", "end"]:
    """检查是否有错误"""
    if state.get("status") == WorkflowStatus.FAILED:
        return "end"
    if state.get("error"):
        return "end"
    return "continue"


def route_after_images(state: ComicState) -> Literal["wait_selection", "audio", "end"]:
    """图片生成后的路由"""
    # 先检查错误
    if state.get("status") == WorkflowStatus.FAILED:
        return "end"
    if state.get("error"):
        return "end"
    
    # 检查是否等待用户输入
    if state.get("status") == WorkflowStatus.WAITING_INPUT:
        return "wait_selection"
    
    return "audio"


def create_comic_workflow() -> StateGraph:
    """
    创建漫剧工作流图（不带 checkpointer）
    
    工作流程:
    ```
    START
      │
      ▼
    script_generator ──► scene_parser ──► image_generator
                                              │
                         ┌────────────────────┤
                         │                    │
              (interactive)                (auto)
                         │                    │
                         ▼                    │
                  [INTERRUPT]                 │
                  wait_selection              │
                         │                    │
                         ▼                    │
                  image_selection ◄───────────┘
                         │
                         ▼
                  audio_generator
                         │
                         ▼
                  video_composer
                         │
                         ▼
                        END
    ```
    
    Returns:
        StateGraph: 编译后的工作流图
    """
    logger.info("building_comic_workflow_graph")
    
    builder = StateGraph(ComicState)
    
    # ===== 添加节点 =====
    
    # 剧本生成
    builder.add_node("script_generator", script_generator_node)
    
    # 场景解析
    builder.add_node("scene_parser", scene_parser_node)
    
    # 图片生成（会根据交互模式设置状态）
    builder.add_node("image_generator", image_generator_node)
    
    # 图片选择处理
    builder.add_node("image_selection", image_selection_node)
    
    # 音频生成
    builder.add_node("audio_generator", audio_generator_node)
    
    # 视频合成
    builder.add_node("video_composer", video_composer_node)
    
    # ===== 添加边 =====
    
    # 开始 -> 剧本生成
    builder.add_edge(START, "script_generator")
    
    # 剧本生成 -> 检查错误 -> 场景解析
    builder.add_conditional_edges(
        "script_generator",
        check_error,
        {"continue": "scene_parser", "end": END},
    )
    
    # 场景解析 -> 检查错误 -> 图片生成
    builder.add_conditional_edges(
        "scene_parser",
        check_error,
        {"continue": "image_generator", "end": END},
    )
    
    # 图片生成 -> 根据交互模式路由（包含错误检查）
    builder.add_conditional_edges(
        "image_generator",
        route_after_images,
        {
            "wait_selection": "image_selection",  # 交互模式
            "audio": "audio_generator",           # 自动模式
            "end": END,                           # 错误时结束
        },
    )
    
    # 图片选择 -> 音频生成
    builder.add_edge("image_selection", "audio_generator")
    
    # 音频生成 -> 检查错误 -> 视频合成
    builder.add_conditional_edges(
        "audio_generator",
        check_error,
        {"continue": "video_composer", "end": END},
    )
    
    # 视频合成 -> 结束
    builder.add_edge("video_composer", END)
    
    logger.info("comic_workflow_graph_built")
    
    return builder.compile()


def create_comic_workflow_with_checkpointer():
    """
    创建带 checkpointer 的漫剧工作流图
    
    支持：
    - 状态持久化
    - 工作流暂停和恢复
    - Human-in-the-loop 交互
    
    Returns:
        tuple: (编译后的图, checkpointer)
    """
    logger.info("building_comic_workflow_with_checkpointer")
    
    builder = StateGraph(ComicState)
    
    # ===== 添加节点（同上）=====
    builder.add_node("script_generator", script_generator_node)
    builder.add_node("scene_parser", scene_parser_node)
    builder.add_node("image_generator", image_generator_node)
    builder.add_node("image_selection", image_selection_node)
    builder.add_node("audio_generator", audio_generator_node)
    builder.add_node("video_composer", video_composer_node)
    
    # ===== 添加边（同上）=====
    builder.add_edge(START, "script_generator")
    
    builder.add_conditional_edges(
        "script_generator",
        check_error,
        {"continue": "scene_parser", "end": END},
    )
    
    builder.add_conditional_edges(
        "scene_parser",
        check_error,
        {"continue": "image_generator", "end": END},
    )
    
    builder.add_conditional_edges(
        "image_generator",
        route_after_images,
        {
            "wait_selection": "image_selection",
            "audio": "audio_generator",
            "end": END,
        },
    )
    
    builder.add_edge("image_selection", "audio_generator")
    
    builder.add_conditional_edges(
        "audio_generator",
        check_error,
        {"continue": "video_composer", "end": END},
    )
    
    builder.add_edge("video_composer", END)
    
    # 创建 checkpointer
    checkpointer = MemorySaver()
    
    # 编译时指定需要中断的节点
    # interrupt_before 表示在执行该节点之前暂停
    compiled = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["image_selection"],  # 在图片选择前暂停
    )
    
    logger.info("comic_workflow_with_checkpointer_built")
    
    return compiled, checkpointer


def print_comic_workflow_structure() -> str:
    """
    打印工作流结构
    
    Returns:
        str: 工作流结构的文本表示
    """
    return """
╔══════════════════════════════════════════════════════════════════╗
║                    Comic Drama Workflow                          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║    ┌─────────┐                                                   ║
║    │  START  │                                                   ║
║    └────┬────┘                                                   ║
║         │                                                        ║
║         ▼                                                        ║
║    ┌──────────────────┐                                          ║
║    │ script_generator │  根据用户提示生成剧本                     ║
║    └────────┬─────────┘                                          ║
║             │                                                    ║
║             ▼                                                    ║
║    ┌──────────────────┐                                          ║
║    │  scene_parser    │  解析剧本为场景列表                       ║
║    └────────┬─────────┘                                          ║
║             │                                                    ║
║             ▼                                                    ║
║    ┌──────────────────┐                                          ║
║    │ image_generator  │  为每个场景生成多张候选图片               ║
║    └────────┬─────────┘                                          ║
║             │                                                    ║
║    ┌────────┴────────┐                                           ║
║    │                 │                                           ║
║    ▼ (interactive)   ▼ (auto)                                    ║
║ ┌─────────────┐      │                                           ║
║ │ [INTERRUPT] │      │                                           ║
║ │ 等待用户选择 │      │                                           ║
║ └──────┬──────┘      │                                           ║
║        ▼             │                                           ║
║ ┌─────────────────┐  │                                           ║
║ │ image_selection │◄─┘  处理用户选择                              ║
║ └────────┬────────┘                                              ║
║          │                                                       ║
║          ▼                                                       ║
║    ┌──────────────────┐                                          ║
║    │ audio_generator  │  为每个场景生成配音                       ║
║    └────────┬─────────┘                                          ║
║             │                                                    ║
║             ▼                                                    ║
║    ┌──────────────────┐                                          ║
║    │ video_composer   │  合成最终视频                             ║
║    └────────┬─────────┘                                          ║
║             │                                                    ║
║             ▼                                                    ║
║    ┌─────────┐                                                   ║
║    │   END   │                                                   ║
║    └─────────┘                                                   ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Mode: auto - 一键生成，自动选择第一张图片                        ║
║  Mode: interactive - 在图片生成后暂停，等待用户选择               ║
╚══════════════════════════════════════════════════════════════════╝
"""
