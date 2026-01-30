"""
剧本生成节点

根据用户提示生成漫剧剧本。
"""

from langchain_core.messages import AIMessage

from multimodal_agent.logging_config import get_logger
from multimodal_agent.utils import run_async
from multimodal_agent.comic_workflow.state import ComicState, WorkflowStatus
from multimodal_agent.services.providers import get_provider
from multimodal_agent.services.providers.base import ProviderType
from multimodal_agent.services.providers.text.base import TextGenerationConfig

logger = get_logger(__name__)

SCRIPT_SYSTEM_PROMPT = """你是一位专业的漫剧编剧。根据用户的提示，创作一个适合漫剧形式的剧本。

剧本格式要求：
1. 每个场景用【场景X】标注
2. 包含场景描述、对白、旁白
3. 场景数量控制在 4-8 个
4. 每个场景的描述要适合转换为图片

输出格式示例：
【场景1】
场景描述：城市的清晨，阳光透过高楼照在街道上
对白：小明：新的一天开始了！
旁白：这是一个平凡的早晨，但注定不平凡的一天。

【场景2】
场景描述：咖啡馆内部，温馨的氛围
对白：咖啡师：早上好，还是老样子吗？
      小明：是的，谢谢！
旁白：熟悉的味道，熟悉的问候。
"""


def script_generator_node(state: ComicState) -> dict:
    """
    剧本生成节点
    
    Args:
        state: 当前状态
        
    Returns:
        dict: 状态更新
    """
    user_prompt = state["user_prompt"]
    provider_config = state["provider_config"]
    
    logger.info(
        "script_generation_started",
        workflow_id=state["workflow_id"],
        prompt_length=len(user_prompt),
    )
    
    try:
        # 获取文本生成 Provider
        text_provider = get_provider(
            ProviderType.TEXT,
            provider_config.text_provider,
        )
        
        # 生成配置
        config = TextGenerationConfig(
            model=provider_config.text_model,
            system_prompt=SCRIPT_SYSTEM_PROMPT,
            max_tokens=4096,
            temperature=0.8,
        )
        
        # 运行异步生成
        result = run_async(
            text_provider.generate(
                prompt=f"请根据以下主题创作一个漫剧剧本：\n\n{user_prompt}",
                config=config,
            )
        )
        
        script = result.content
        
        logger.info(
            "script_generation_completed",
            workflow_id=state["workflow_id"],
            script_length=len(script),
        )
        
        ai_message = AIMessage(
            content=f"剧本生成完成！\n\n{script[:500]}..."
        )
        
        return {
            "script": script,
            "status": WorkflowStatus.RUNNING,
            "messages": [ai_message],
        }
        
    except Exception as e:
        logger.error(
            "script_generation_failed",
            workflow_id=state["workflow_id"],
            error=str(e),
        )
        
        return {
            "status": WorkflowStatus.FAILED,
            "error": f"剧本生成失败: {str(e)}",
            "messages": [AIMessage(content=f"剧本生成失败: {str(e)}")],
        }
