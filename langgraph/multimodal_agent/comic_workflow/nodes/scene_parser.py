"""
场景解析节点

将剧本解析为场景列表。
"""

import re
import uuid
from langchain_core.messages import AIMessage

from multimodal_agent.logging_config import get_logger
from multimodal_agent.comic_workflow.state import ComicState, Scene, WorkflowStatus

logger = get_logger(__name__)


def parse_script_to_scenes(script: str) -> list[Scene]:
    """
    解析剧本为场景列表
    
    Args:
        script: 剧本文本
        
    Returns:
        list[Scene]: 场景列表
    """
    scenes = []
    
    # 使用正则表达式分割场景
    scene_pattern = r'【场景(\d+)】\s*([\s\S]*?)(?=【场景\d+】|$)'
    matches = re.findall(scene_pattern, script)
    
    for i, (scene_num, content) in enumerate(matches):
        # 解析场景内容
        description = ""
        dialogue = ""
        narration = ""
        
        # 提取场景描述
        desc_match = re.search(r'场景描述[：:]\s*(.+?)(?=对白|旁白|$)', content, re.DOTALL)
        if desc_match:
            description = desc_match.group(1).strip()
        
        # 提取对白
        dialogue_match = re.search(r'对白[：:]\s*([\s\S]*?)(?=旁白|$)', content)
        if dialogue_match:
            dialogue = dialogue_match.group(1).strip()
        
        # 提取旁白
        narration_match = re.search(r'旁白[：:]\s*(.+?)$', content, re.DOTALL)
        if narration_match:
            narration = narration_match.group(1).strip()
        
        # 如果没有匹配到格式化内容，使用整个内容作为描述
        if not description and not dialogue and not narration:
            description = content.strip()
        
        scene = Scene(
            scene_id=str(uuid.uuid4()),
            scene_number=int(scene_num),
            description=description or f"场景 {scene_num}",
            dialogue=dialogue if dialogue else None,
            narration=narration if narration else None,
        )
        scenes.append(scene)
    
    # 如果没有匹配到任何场景，创建一个默认场景
    if not scenes:
        scenes.append(Scene(
            scene_id=str(uuid.uuid4()),
            scene_number=1,
            description=script[:500] if script else "默认场景",
            dialogue=None,
            narration=None,
        ))
    
    return scenes


def scene_parser_node(state: ComicState) -> dict:
    """
    场景解析节点
    
    Args:
        state: 当前状态
        
    Returns:
        dict: 状态更新
    """
    script = state.get("script")
    
    if not script:
        logger.error(
            "scene_parsing_failed_no_script",
            workflow_id=state["workflow_id"],
        )
        return {
            "status": WorkflowStatus.FAILED,
            "error": "没有可用的剧本",
        }
    
    logger.info(
        "scene_parsing_started",
        workflow_id=state["workflow_id"],
        script_length=len(script),
    )
    
    try:
        scenes = parse_script_to_scenes(script)
        
        logger.info(
            "scene_parsing_completed",
            workflow_id=state["workflow_id"],
            scene_count=len(scenes),
        )
        
        ai_message = AIMessage(
            content=f"场景解析完成！共 {len(scenes)} 个场景。\n\n"
                    + "\n".join([f"- 场景{s.scene_number}: {s.description[:50]}..." for s in scenes])
        )
        
        return {
            "scenes": scenes,
            "messages": [ai_message],
        }
        
    except Exception as e:
        logger.error(
            "scene_parsing_failed",
            workflow_id=state["workflow_id"],
            error=str(e),
        )
        
        return {
            "status": WorkflowStatus.FAILED,
            "error": f"场景解析失败: {str(e)}",
        }
