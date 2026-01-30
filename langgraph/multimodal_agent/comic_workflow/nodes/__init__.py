"""
漫剧工作流节点
"""

from multimodal_agent.comic_workflow.nodes.script_generator import script_generator_node
from multimodal_agent.comic_workflow.nodes.scene_parser import scene_parser_node
from multimodal_agent.comic_workflow.nodes.image_generator import image_generator_node
from multimodal_agent.comic_workflow.nodes.audio_generator import audio_generator_node
from multimodal_agent.comic_workflow.nodes.video_composer import video_composer_node
from multimodal_agent.comic_workflow.nodes.interaction_handler import (
    image_selection_node,
    should_wait_for_selection,
)

__all__ = [
    "script_generator_node",
    "scene_parser_node",
    "image_generator_node",
    "audio_generator_node",
    "video_composer_node",
    "image_selection_node",
    "should_wait_for_selection",
]
