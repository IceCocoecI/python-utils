"""
Router 节点 (Supervisor)

负责分析用户意图，决定将任务路由到哪个 Worker。
使用 LLM 进行意图识别。
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from multimodal_agent.config import get_config
from multimodal_agent.logging_config import get_logger
from multimodal_agent.state import AgentState, TaskType, ErrorInfo

logger = get_logger(__name__)


class IntentClassification(BaseModel):
    """意图分类结果"""
    
    task_type: Literal["text", "image", "video", "unknown"] = Field(
        description="识别的任务类型：text=文本生成, image=图片生成, video=视频生成, unknown=无法识别"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="置信度分数"
    )
    reasoning: str = Field(
        description="分类理由"
    )


# 系统提示词
ROUTER_SYSTEM_PROMPT = """你是一个意图分类助手，负责分析用户的输入并判断他们想要生成什么类型的内容。

你需要将用户请求分类为以下类型之一：
1. **text** - 文本生成任务：
   - 写文章、写故事、写代码
   - 总结、翻译、改写
   - 回答问题、解释概念
   - 任何需要生成文字内容的请求

2. **image** - 图片生成任务：
   - 生成图片、画图、绘画
   - 创建插图、设计图像
   - 关键词：画、图片、图像、绘制、设计、logo、海报

3. **video** - 视频生成任务：
   - 生成视频、制作动画
   - 创建短片、制作视频内容
   - 关键词：视频、动画、影片、短视频

4. **unknown** - 无法识别：
   - 如果请求不清楚或不属于以上任何类型

请仔细分析用户的请求，输出你的判断。"""


def _create_router_llm() -> ChatOpenAI:
    """
    创建用于路由的 LLM 实例
    
    Returns:
        ChatOpenAI: 配置好的 LLM 实例
    """
    config = get_config()
    
    return ChatOpenAI(
        model=config.openai.model_name,
        api_key=config.openai.api_key.get_secret_value(),
        base_url=config.openai.api_base,
        temperature=0,  # 使用确定性输出
    )


def _classify_intent_with_llm(user_input: str) -> IntentClassification:
    """
    使用 LLM 进行意图分类
    
    Args:
        user_input: 用户输入
        
    Returns:
        IntentClassification: 分类结果
    """
    llm = _create_router_llm()
    
    # 使用结构化输出
    structured_llm = llm.with_structured_output(IntentClassification)
    
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=f"请分析以下用户请求：\n\n{user_input}"),
    ]
    
    result = structured_llm.invoke(messages)
    return result


def _classify_intent_fallback(user_input: str) -> IntentClassification:
    """
    备用的基于规则的意图分类
    
    当 LLM 不可用时使用此方法。
    
    Args:
        user_input: 用户输入
        
    Returns:
        IntentClassification: 分类结果
    """
    user_input_lower = user_input.lower()
    
    # 视频关键词
    video_keywords = ["视频", "video", "动画", "animation", "影片", "短片", "clip"]
    if any(kw in user_input_lower for kw in video_keywords):
        return IntentClassification(
            task_type="video",
            confidence=0.8,
            reasoning="检测到视频相关关键词",
        )
    
    # 图片关键词
    image_keywords = [
        "图片", "图像", "image", "picture", "画", "draw", "绘制",
        "设计", "design", "logo", "海报", "poster", "插图", "illustration"
    ]
    if any(kw in user_input_lower for kw in image_keywords):
        return IntentClassification(
            task_type="image",
            confidence=0.8,
            reasoning="检测到图片相关关键词",
        )
    
    # 文本关键词
    text_keywords = [
        "写", "write", "文章", "article", "故事", "story", "代码", "code",
        "总结", "summarize", "翻译", "translate", "解释", "explain",
        "回答", "answer", "描述", "describe", "生成文本", "文字"
    ]
    if any(kw in user_input_lower for kw in text_keywords):
        return IntentClassification(
            task_type="text",
            confidence=0.8,
            reasoning="检测到文本相关关键词",
        )
    
    # 默认为文本生成
    return IntentClassification(
        task_type="text",
        confidence=0.5,
        reasoning="未检测到明确关键词，默认为文本生成任务",
    )


def router_node(state: AgentState) -> dict:
    """
    Router 节点函数
    
    分析用户意图并设置任务类型。
    
    Args:
        state: 当前 Agent 状态
        
    Returns:
        dict: 状态更新
    """
    user_input = state["user_input"]
    
    logger.info(
        "router_processing",
        user_input_length=len(user_input),
    )
    
    try:
        # 尝试使用 LLM 进行分类
        try:
            classification = _classify_intent_with_llm(user_input)
            logger.info(
                "intent_classified_by_llm",
                task_type=classification.task_type,
                confidence=classification.confidence,
                reasoning=classification.reasoning,
            )
        except Exception as llm_error:
            # LLM 失败时使用备用方案
            logger.warning(
                "llm_classification_failed_using_fallback",
                error=str(llm_error),
            )
            classification = _classify_intent_fallback(user_input)
            logger.info(
                "intent_classified_by_fallback",
                task_type=classification.task_type,
                confidence=classification.confidence,
                reasoning=classification.reasoning,
            )
        
        # 映射到 TaskType 枚举
        task_type_map = {
            "text": TaskType.TEXT,
            "image": TaskType.IMAGE,
            "video": TaskType.VIDEO,
            "unknown": TaskType.UNKNOWN,
        }
        task_type = task_type_map.get(classification.task_type, TaskType.UNKNOWN)
        
        # 添加 AI 消息到对话历史
        ai_message = AIMessage(
            content=f"已识别任务类型：{task_type.value}（置信度：{classification.confidence:.2f}）\n理由：{classification.reasoning}"
        )
        
        return {
            "messages": [ai_message],
            "task_type": task_type,
        }
        
    except Exception as e:
        logger.error(
            "router_error",
            error=str(e),
        )
        
        return {
            "task_type": TaskType.UNKNOWN,
            "error_message": ErrorInfo(
                error_type="RouterError",
                message=f"意图识别失败: {str(e)}",
                node_name="router",
            ),
            "is_completed": True,
        }


def route_to_worker(state: AgentState) -> Literal["text_worker", "image_worker", "video_worker", "error_handler"]:
    """
    路由决策函数
    
    根据任务类型决定路由到哪个 Worker。
    
    Args:
        state: 当前 Agent 状态
        
    Returns:
        str: 目标节点名称
    """
    # 如果已有错误，路由到错误处理
    if state.get("error_message"):
        logger.info("routing_to_error_handler")
        return "error_handler"
    
    task_type = state.get("task_type")
    
    if task_type == TaskType.TEXT:
        logger.info("routing_to_text_worker")
        return "text_worker"
    elif task_type == TaskType.IMAGE:
        logger.info("routing_to_image_worker")
        return "image_worker"
    elif task_type == TaskType.VIDEO:
        logger.info("routing_to_video_worker")
        return "video_worker"
    else:
        logger.warning("unknown_task_type_routing_to_error")
        return "error_handler"
