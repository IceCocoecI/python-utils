"""
状态管理模块

定义 Agent 的状态结构，使用 Pydantic 进行严格的数据验证。
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import operator


class TaskType(str, Enum):
    """任务类型枚举"""
    
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    UNKNOWN = "unknown"


class ContentResult(BaseModel):
    """生成内容结果"""
    
    task_type: TaskType = Field(
        description="任务类型"
    )
    content: str | None = Field(
        default=None,
        description="生成的内容（文本）或 URL（图片/视频）"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="额外的元数据"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="创建时间"
    )
    
    class Config:
        use_enum_values = True


class ErrorInfo(BaseModel):
    """错误信息"""
    
    error_type: str = Field(
        description="错误类型"
    )
    message: str = Field(
        description="错误消息"
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="错误详情"
    )
    occurred_at: datetime = Field(
        default_factory=datetime.now,
        description="发生时间"
    )
    node_name: str = Field(
        default="",
        description="发生错误的节点名称"
    )


class AgentState(TypedDict):
    """
    Agent 状态定义
    
    使用 TypedDict 以兼容 LangGraph 的状态管理机制。
    
    Attributes:
        messages: 对话历史，使用 operator.add 进行累加
        user_input: 用户原始输入
        task_type: 识别的任务类型
        generated_content: 生成的内容结果
        error_message: 错误信息（如果有）
        is_completed: 是否完成
    """
    
    # 对话历史 - 使用 Annotated 标注累加方式
    messages: Annotated[list[AnyMessage], operator.add]
    
    # 用户原始输入
    user_input: str
    
    # 任务类型
    task_type: TaskType | None
    
    # 生成结果
    generated_content: ContentResult | None
    
    # 错误信息
    error_message: ErrorInfo | None
    
    # 完成标记
    is_completed: bool


def create_initial_state(user_input: str) -> AgentState:
    """
    创建初始状态
    
    Args:
        user_input: 用户输入
        
    Returns:
        AgentState: 初始化的 Agent 状态
    """
    return AgentState(
        messages=[],
        user_input=user_input,
        task_type=None,
        generated_content=None,
        error_message=None,
        is_completed=False,
    )


# 路由决策返回类型
RouteDecision = Literal["text_worker", "image_worker", "video_worker", "error_handler"]
