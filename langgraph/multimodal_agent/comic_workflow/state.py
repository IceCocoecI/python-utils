"""
漫剧工作流状态定义

支持 Human-in-the-loop 的复杂状态管理。
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal
import operator

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class WorkflowStatus(str, Enum):
    """工作流状态"""
    
    PENDING = "pending"           # 等待开始
    RUNNING = "running"           # 运行中
    WAITING_INPUT = "waiting_input"  # 等待用户输入
    COMPLETED = "completed"       # 完成
    FAILED = "failed"             # 失败
    CANCELLED = "cancelled"       # 取消


class InteractionPoint(str, Enum):
    """交互点类型"""
    
    SCRIPT_REVIEW = "script_review"      # 剧本审核
    SCENE_SELECTION = "scene_selection"  # 分镜选择
    IMAGE_SELECTION = "image_selection"  # 图片选择
    AUDIO_REVIEW = "audio_review"        # 配音审核
    FINAL_REVIEW = "final_review"        # 最终审核


class SceneImage(BaseModel):
    """场景图片"""
    
    image_id: str = Field(description="图片 ID")
    url: str = Field(description="图片 URL")
    prompt: str = Field(description="生成提示")
    is_selected: bool = Field(default=False, description="是否被选中")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SceneAudio(BaseModel):
    """场景音频"""
    
    audio_id: str = Field(description="音频 ID")
    url: str = Field(description="音频 URL")
    text: str = Field(description="配音文本")
    voice: str = Field(description="使用的声音")
    duration: float = Field(description="时长（秒）")


class Scene(BaseModel):
    """漫剧场景/分镜"""
    
    scene_id: str = Field(description="场景 ID")
    scene_number: int = Field(description="场景序号")
    description: str = Field(description="场景描述")
    dialogue: str | None = Field(default=None, description="对白")
    narration: str | None = Field(default=None, description="旁白")
    
    # 图片候选（多张供选择）
    image_candidates: list[SceneImage] = Field(
        default_factory=list,
        description="图片候选列表"
    )
    selected_image: SceneImage | None = Field(
        default=None,
        description="选中的图片"
    )
    
    # 音频
    audio: SceneAudio | None = Field(default=None, description="配音")
    
    # 生成的视频片段
    video_url: str | None = Field(default=None, description="视频片段 URL")


class ProviderConfig(BaseModel):
    """Provider 配置"""
    
    text_provider: str = Field(default="mock", description="文本生成 Provider")
    text_model: str | None = Field(default=None, description="文本模型")
    
    image_provider: str = Field(default="mock", description="图片生成 Provider")
    image_model: str | None = Field(default=None, description="图片模型")
    
    video_provider: str = Field(default="mock", description="视频生成 Provider")
    video_model: str | None = Field(default=None, description="视频模型")
    
    audio_provider: str = Field(default="mock", description="音频生成 Provider")
    audio_model: str | None = Field(default=None, description="音频模型")


class ComicState(TypedDict):
    """
    漫剧工作流状态
    
    支持 Human-in-the-loop 的完整状态定义。
    """
    
    # ===== 基础信息 =====
    workflow_id: str                    # 工作流 ID
    status: WorkflowStatus              # 工作流状态
    created_at: str                     # 创建时间
    
    # ===== 用户输入 =====
    user_prompt: str                    # 用户原始提示
    style_preferences: dict[str, Any]   # 风格偏好
    
    # ===== Provider 配置 =====
    provider_config: ProviderConfig     # Provider 配置
    
    # ===== 生成内容 =====
    script: str | None                  # 生成的剧本
    scenes: list[Scene]                 # 场景列表
    
    # ===== 输出 =====
    final_video_url: str | None         # 最终视频 URL
    
    # ===== 交互控制 =====
    interaction_mode: Literal["auto", "interactive"]  # 交互模式
    current_interaction: InteractionPoint | None      # 当前交互点
    pending_selection: dict[str, Any] | None          # 待选择的内容
    
    # ===== 对话历史 =====
    messages: Annotated[list[AnyMessage], operator.add]
    
    # ===== 错误信息 =====
    error: str | None


def create_initial_comic_state(
    user_prompt: str,
    workflow_id: str | None = None,
    interaction_mode: Literal["auto", "interactive"] = "interactive",
    provider_config: ProviderConfig | None = None,
    style_preferences: dict[str, Any] | None = None,
) -> ComicState:
    """
    创建初始漫剧工作流状态
    
    Args:
        user_prompt: 用户提示
        workflow_id: 工作流 ID（可选，自动生成）
        interaction_mode: 交互模式
        provider_config: Provider 配置
        style_preferences: 风格偏好
        
    Returns:
        ComicState: 初始状态
    """
    import uuid
    
    return ComicState(
        workflow_id=workflow_id or str(uuid.uuid4()),
        status=WorkflowStatus.PENDING,
        created_at=datetime.now().isoformat(),
        user_prompt=user_prompt,
        style_preferences=style_preferences or {},
        provider_config=provider_config or ProviderConfig(),
        script=None,
        scenes=[],
        final_video_url=None,
        interaction_mode=interaction_mode,
        current_interaction=None,
        pending_selection=None,
        messages=[],
        error=None,
    )
