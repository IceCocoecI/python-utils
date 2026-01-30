"""
漫剧工作流运行器

提供工作流的执行、暂停、恢复功能。
支持 Human-in-the-loop 交互。
"""

import uuid
from typing import Any, Generator

from langchain_core.messages import HumanMessage

from multimodal_agent.logging_config import get_logger
from multimodal_agent.comic_workflow.state import (
    ComicState,
    WorkflowStatus,
    InteractionPoint,
    ProviderConfig,
    create_initial_comic_state,
)
from multimodal_agent.comic_workflow.graph import (
    create_comic_workflow,
    create_comic_workflow_with_checkpointer,
)

logger = get_logger(__name__)


class ComicWorkflowRunner:
    """
    漫剧工作流运行器
    
    支持：
    - 自动模式：一键生成
    - 交互模式：暂停等待用户选择
    - 状态持久化和恢复
    """
    
    def __init__(self, use_checkpointer: bool = True):
        """
        初始化运行器
        
        Args:
            use_checkpointer: 是否使用 checkpointer（支持暂停恢复）
        """
        self.use_checkpointer = use_checkpointer
        
        if use_checkpointer:
            self.graph, self.checkpointer = create_comic_workflow_with_checkpointer()
        else:
            self.graph = create_comic_workflow()
            self.checkpointer = None
        
        self._current_thread_id: str | None = None
    
    def run_auto(
        self,
        user_prompt: str,
        provider_config: ProviderConfig | None = None,
        style_preferences: dict[str, Any] | None = None,
    ) -> ComicState:
        """
        自动模式运行：一键生成，无需用户干预
        
        Args:
            user_prompt: 用户提示
            provider_config: Provider 配置
            style_preferences: 风格偏好
            
        Returns:
            ComicState: 最终状态
        """
        workflow_id = str(uuid.uuid4())
        
        initial_state = create_initial_comic_state(
            user_prompt=user_prompt,
            workflow_id=workflow_id,
            interaction_mode="auto",
            provider_config=provider_config,
            style_preferences=style_preferences,
        )
        
        initial_state["messages"] = [HumanMessage(content=user_prompt)]
        
        logger.info(
            "comic_workflow_auto_started",
            workflow_id=workflow_id,
        )
        
        if self.use_checkpointer:
            config = {"configurable": {"thread_id": workflow_id}}
            result = self.graph.invoke(initial_state, config)
        else:
            result = self.graph.invoke(initial_state)
        
        logger.info(
            "comic_workflow_auto_completed",
            workflow_id=workflow_id,
            status=result.get("status"),
        )
        
        return result
    
    def run_interactive(
        self,
        user_prompt: str,
        provider_config: ProviderConfig | None = None,
        style_preferences: dict[str, Any] | None = None,
    ) -> Generator[ComicState, dict[str, Any] | None, ComicState]:
        """
        交互模式运行：使用 Generator 支持暂停和恢复
        
        Usage:
            runner = ComicWorkflowRunner()
            gen = runner.run_interactive("创作一个关于冒险的漫剧")
            
            # 第一次运行，直到暂停
            state = next(gen)
            
            if state["status"] == WorkflowStatus.WAITING_INPUT:
                # 显示候选图片，让用户选择
                selections = get_user_selections(state)
                
                # 发送选择并继续
                state = gen.send(selections)
            
            # 获取最终结果
            final_state = state
        
        Args:
            user_prompt: 用户提示
            provider_config: Provider 配置
            style_preferences: 风格偏好
            
        Yields:
            ComicState: 工作流状态（可能在交互点暂停）
            
        Returns:
            ComicState: 最终状态
        """
        if not self.use_checkpointer:
            raise RuntimeError("交互模式需要 checkpointer，请使用 use_checkpointer=True")
        
        workflow_id = str(uuid.uuid4())
        self._current_thread_id = workflow_id
        
        initial_state = create_initial_comic_state(
            user_prompt=user_prompt,
            workflow_id=workflow_id,
            interaction_mode="interactive",
            provider_config=provider_config,
            style_preferences=style_preferences,
        )
        
        initial_state["messages"] = [HumanMessage(content=user_prompt)]
        
        config = {"configurable": {"thread_id": workflow_id}}
        
        logger.info(
            "comic_workflow_interactive_started",
            workflow_id=workflow_id,
        )
        
        # 第一次运行
        result = None
        for event in self.graph.stream(initial_state, config, stream_mode="values"):
            result = event
        
        # 如果没有结果，返回初始状态
        if result is None:
            logger.error("comic_workflow_no_result", workflow_id=workflow_id)
            initial_state["status"] = WorkflowStatus.FAILED
            initial_state["error"] = "工作流未产生任何输出"
            yield initial_state
            return initial_state
        
        # 检查是否需要用户输入
        while result.get("status") == WorkflowStatus.WAITING_INPUT:
            logger.info(
                "comic_workflow_waiting_input",
                workflow_id=workflow_id,
                interaction_point=result.get("current_interaction"),
            )
            
            # 暂停，等待用户输入
            user_input = yield result
            
            if user_input is None:
                # 用户取消
                logger.info("comic_workflow_cancelled_by_user", workflow_id=workflow_id)
                result["status"] = WorkflowStatus.CANCELLED
                yield result
                return result
            
            # 应用用户选择
            logger.info(
                "comic_workflow_resuming",
                workflow_id=workflow_id,
                user_input_keys=list(user_input.keys()) if isinstance(user_input, dict) else None,
            )
            
            # 更新状态
            update = self._process_user_input(result, user_input)
            
            # 继续执行
            result = None
            for event in self.graph.stream(update, config, stream_mode="values"):
                result = event
            
            # 如果没有结果，中断
            if result is None:
                logger.error("comic_workflow_resume_no_result", workflow_id=workflow_id)
                update["status"] = WorkflowStatus.FAILED
                update["error"] = "恢复执行后未产生输出"
                yield update
                return update
        
        logger.info(
            "comic_workflow_interactive_completed",
            workflow_id=workflow_id,
            status=result.get("status") if result else None,
        )
        
        # 最终 yield 结果，确保调用者能获取到
        yield result
        return result
    
    def _process_user_input(
        self,
        current_state: ComicState,
        user_input: dict[str, Any],
    ) -> ComicState:
        """
        处理用户输入，更新状态
        
        Args:
            current_state: 当前状态
            user_input: 用户输入
            
        Returns:
            ComicState: 更新后的状态
        """
        interaction_type = current_state.get("current_interaction")
        
        if interaction_type == InteractionPoint.IMAGE_SELECTION:
            # 图片选择
            selections = user_input.get("selections", {})
            scenes = current_state.get("scenes", [])
            
            for scene in scenes:
                selected_id = selections.get(scene.scene_id)
                if selected_id:
                    for img in scene.image_candidates:
                        if img.image_id == selected_id:
                            img.is_selected = True
                            scene.selected_image = img
                            break
                
                # 如果没有选择，默认第一张
                if not scene.selected_image and scene.image_candidates:
                    scene.image_candidates[0].is_selected = True
                    scene.selected_image = scene.image_candidates[0]
            
            # 更新状态
            updated_state = dict(current_state)
            updated_state["scenes"] = scenes
            updated_state["status"] = WorkflowStatus.RUNNING
            updated_state["current_interaction"] = None
            updated_state["pending_selection"] = None
            
            return updated_state
        
        # 其他交互类型...
        return current_state
    
    def get_pending_selection(self, state: ComicState) -> dict[str, Any] | None:
        """
        获取待选择的内容
        
        Args:
            state: 当前状态
            
        Returns:
            dict | None: 待选择的内容
        """
        return state.get("pending_selection")
    
    def resume(
        self,
        thread_id: str,
        user_input: dict[str, Any],
    ) -> ComicState:
        """
        恢复暂停的工作流
        
        Args:
            thread_id: 工作流线程 ID
            user_input: 用户输入
            
        Returns:
            ComicState: 执行后的状态
        """
        if not self.use_checkpointer:
            raise RuntimeError("恢复功能需要 checkpointer")
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # 获取当前状态
        current_state = self.graph.get_state(config)
        
        if current_state is None:
            raise ValueError(f"找不到线程 {thread_id} 的状态")
        
        # 处理用户输入并更新状态
        updated_state = self._process_user_input(current_state.values, user_input)
        
        # 继续执行
        result = None
        for event in self.graph.stream(updated_state, config, stream_mode="values"):
            result = event
        
        return result


def run_comic_workflow_demo():
    """
    运行漫剧工作流演示
    """
    print("\n" + "=" * 60)
    print("漫剧工作流演示")
    print("=" * 60)
    
    # 使用 Mock Provider
    provider_config = ProviderConfig(
        text_provider="mock",
        image_provider="mock",
        video_provider="mock",
        audio_provider="mock",
    )
    
    # ===== 自动模式演示 =====
    print("\n--- 自动模式 ---")
    
    runner = ComicWorkflowRunner(use_checkpointer=False)
    
    result = runner.run_auto(
        user_prompt="创作一个关于一只小猫冒险的温馨故事",
        provider_config=provider_config,
        style_preferences={"art_style": "anime illustration"},
    )
    
    print(f"状态: {result['status']}")
    print(f"场景数: {len(result.get('scenes', []))}")
    if result.get("final_video_url"):
        print(f"视频 URL: {result['final_video_url']}")
    
    # ===== 交互模式演示 =====
    print("\n--- 交互模式演示 ---")
    
    runner_interactive = ComicWorkflowRunner(use_checkpointer=True)
    
    gen = runner_interactive.run_interactive(
        user_prompt="创作一个关于友谊的漫剧",
        provider_config=provider_config,
    )
    
    try:
        # 运行直到第一个暂停点
        state = next(gen)
        
        print(f"当前状态: {state['status']}")
        
        if state["status"] == WorkflowStatus.WAITING_INPUT:
            print("工作流暂停，等待用户选择图片...")
            
            # 获取待选择内容
            pending = runner_interactive.get_pending_selection(state)
            if pending:
                print(f"待选择场景数: {len(pending.get('scenes', []))}")
                
                # 模拟用户选择（选择每个场景的第一张图片）
                selections = {}
                for scene in pending.get("scenes", []):
                    if scene.get("candidates"):
                        selections[scene["scene_id"]] = scene["candidates"][0]["image_id"]
                
                print(f"用户选择: {selections}")
                
                # 发送选择并继续
                state = gen.send({"selections": selections})
        
        print(f"最终状态: {state['status']}")
        if state.get("final_video_url"):
            print(f"视频 URL: {state['final_video_url']}")
            
    except StopIteration as e:
        print(f"工作流完成: {e.value}")
    
    print("\n" + "=" * 60)
    print("演示完成!")


if __name__ == "__main__":
    from multimodal_agent.logging_config import setup_logging
    setup_logging()
    run_comic_workflow_demo()
