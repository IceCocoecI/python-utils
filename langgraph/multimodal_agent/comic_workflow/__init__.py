"""
漫剧工作流模块

支持一键式生成和 Human-in-the-loop 交互模式。
"""

from multimodal_agent.comic_workflow.state import (
    ComicState,
    Scene,
    SceneImage,
    SceneAudio,
    WorkflowStatus,
    InteractionPoint,
    ProviderConfig,
    create_initial_comic_state,
)
from multimodal_agent.comic_workflow.graph import (
    create_comic_workflow,
    create_comic_workflow_with_checkpointer,
)
from multimodal_agent.comic_workflow.runner import ComicWorkflowRunner

__all__ = [
    # 状态定义
    "ComicState",
    "Scene",
    "SceneImage", 
    "SceneAudio",
    "WorkflowStatus",
    "InteractionPoint",
    "ProviderConfig",
    "create_initial_comic_state",
    # 工作流
    "create_comic_workflow",
    "create_comic_workflow_with_checkpointer",
    # 运行器
    "ComicWorkflowRunner",
]
