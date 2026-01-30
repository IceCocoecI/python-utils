"""
交互处理节点

处理 Human-in-the-loop 交互，如图片选择、审核等。
"""

from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage

from multimodal_agent.logging_config import get_logger
from multimodal_agent.comic_workflow.state import (
    ComicState,
    WorkflowStatus,
    InteractionPoint,
)

logger = get_logger(__name__)


def image_selection_node(state: ComicState) -> dict:
    """
    图片选择节点
    
    处理用户的图片选择输入。
    
    注意：在 LangGraph 中，这个节点会在 interrupt 之后被调用。
    用户的选择可能已经在 runner._process_user_input 中处理，
    或者需要在这里处理（如果直接调用 graph）。
    
    Args:
        state: 当前状态（可能已包含用户选择）
        
    Returns:
        dict: 状态更新
    """
    scenes = state.get("scenes", [])
    
    logger.info(
        "processing_image_selection",
        workflow_id=state["workflow_id"],
        scene_count=len(scenes),
    )
    
    # 检查是否已经有选择（由 runner._process_user_input 处理）
    already_selected = all(s.selected_image is not None for s in scenes if s.image_candidates)
    
    if already_selected:
        logger.info(
            "image_selection_already_done",
            workflow_id=state["workflow_id"],
        )
        # 选择已完成，只需更新状态
        ai_message = AIMessage(
            content=f"图片选择完成！已为 {len(scenes)} 个场景选择图片。"
        )
        return {
            "status": WorkflowStatus.RUNNING,
            "current_interaction": None,
            "pending_selection": None,
            "messages": [ai_message],
        }
    
    # 如果没有预先选择，使用默认（第一张）
    logger.info(
        "using_default_selection",
        workflow_id=state["workflow_id"],
    )
    
    updated_scenes = []
    selected_count = 0
    for scene in scenes:
        # 如果还没有选择，默认选第一张
        if not scene.selected_image and scene.image_candidates:
            scene.image_candidates[0].is_selected = True
            scene.selected_image = scene.image_candidates[0]
            selected_count += 1
        updated_scenes.append(scene)
    
    logger.info(
        "image_selection_completed",
        workflow_id=state["workflow_id"],
        default_selections=selected_count,
    )
    
    ai_message = AIMessage(
        content=f"图片选择完成！已为 {len(updated_scenes)} 个场景选择图片。"
    )
    
    return {
        "scenes": updated_scenes,
        "status": WorkflowStatus.RUNNING,
        "current_interaction": None,
        "pending_selection": None,
        "messages": [ai_message],
    }


def should_wait_for_selection(state: ComicState) -> Literal["wait", "continue"]:
    """
    决策函数：是否需要等待用户选择
    
    Args:
        state: 当前状态
        
    Returns:
        str: "wait" 或 "continue"
    """
    if state.get("status") == WorkflowStatus.WAITING_INPUT:
        return "wait"
    return "continue"


def apply_user_selection(state: ComicState, selections: dict[str, str]) -> dict:
    """
    应用用户选择
    
    这是一个工具函数，用于在外部应用用户选择后更新状态。
    
    Args:
        state: 当前状态
        selections: 选择映射 {scene_id: image_id}
        
    Returns:
        dict: 状态更新
    """
    scenes = state.get("scenes", [])
    
    updated_scenes = []
    for scene in scenes:
        selected_id = selections.get(scene.scene_id)
        
        if selected_id:
            for img in scene.image_candidates:
                if img.image_id == selected_id:
                    img.is_selected = True
                    scene.selected_image = img
                    logger.info(
                        "image_selected",
                        scene_id=scene.scene_id,
                        image_id=selected_id,
                    )
                    break
        
        updated_scenes.append(scene)
    
    return {
        "scenes": updated_scenes,
        "status": WorkflowStatus.RUNNING,
        "current_interaction": None,
        "pending_selection": None,
    }
