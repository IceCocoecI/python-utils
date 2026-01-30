"""
漫剧工作流命令行界面

提供交互式命令行体验。
"""

import sys
from pathlib import Path

# 添加父目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from typing import Any

from multimodal_agent.logging_config import setup_logging, get_logger
from multimodal_agent.comic_workflow.state import (
    ComicState,
    WorkflowStatus,
    ProviderConfig,
    InteractionPoint,
)
from multimodal_agent.comic_workflow.runner import ComicWorkflowRunner
from multimodal_agent.comic_workflow.graph import print_comic_workflow_structure
from multimodal_agent.services.providers import list_providers

setup_logging()
logger = get_logger(__name__)


def print_providers():
    """打印可用的 Provider"""
    print("\n可用的 Provider:")
    print("-" * 40)
    
    providers = list_providers()
    for provider_type, provider_list in providers.items():
        print(f"\n{provider_type}:")
        for p in provider_list:
            default_mark = " (默认)" if p.is_default else ""
            print(f"  - {p.name}{default_mark}: {p.description}")
            if p.models:
                print(f"    模型: {', '.join(p.models)}")


def print_result(state: ComicState):
    """打印执行结果"""
    print("\n" + "=" * 60)
    print("执行结果")
    print("=" * 60)
    
    print(f"\n状态: {state['status']}")
    
    if state.get("script"):
        print(f"\n剧本长度: {len(state['script'])} 字符")
    
    scenes = state.get("scenes", [])
    if scenes:
        print(f"\n场景数量: {len(scenes)}")
        for scene in scenes:
            print(f"\n  场景 {scene.scene_number}:")
            print(f"    描述: {scene.description[:50]}...")
            if scene.selected_image:
                print(f"    选中图片: {scene.selected_image.url}")
            if scene.audio:
                print(f"    配音时长: {scene.audio.duration:.1f}秒")
    
    if state.get("final_video_url"):
        print(f"\n最终视频: {state['final_video_url']}")
    
    if state.get("error"):
        print(f"\n错误: {state['error']}")
    
    print("\n" + "=" * 60)


def interactive_image_selection(state: ComicState) -> dict[str, str]:
    """
    交互式图片选择
    
    Args:
        state: 当前状态
        
    Returns:
        dict: 选择结果 {scene_id: image_id}
    """
    pending = state.get("pending_selection", {})
    scenes = pending.get("scenes", [])
    
    print("\n" + "=" * 60)
    print("请为每个场景选择一张图片")
    print("=" * 60)
    
    selections = {}
    
    for scene_info in scenes:
        scene_id = scene_info["scene_id"]
        scene_num = scene_info["scene_number"]
        description = scene_info["description"]
        candidates = scene_info["candidates"]
        
        print(f"\n场景 {scene_num}: {description}")
        print("-" * 40)
        
        for i, img in enumerate(candidates, 1):
            print(f"  {i}. {img['url']}")
        
        while True:
            try:
                choice = input(f"请选择 (1-{len(candidates)}, 默认1): ").strip()
                if not choice:
                    choice = "1"
                
                idx = int(choice) - 1
                if 0 <= idx < len(candidates):
                    selections[scene_id] = candidates[idx]["image_id"]
                    print(f"  ✓ 已选择图片 {choice}")
                    break
                else:
                    print(f"  请输入 1-{len(candidates)} 之间的数字")
            except ValueError:
                print("  请输入有效数字")
            except KeyboardInterrupt:
                print("\n已取消")
                return {}
    
    return selections


def run_interactive_mode():
    """交互模式"""
    print("\n" + "=" * 60)
    print("漫剧工作流 - 交互模式")
    print("=" * 60)
    
    # 获取用户输入
    print("\n请输入您想创作的漫剧主题:")
    user_prompt = input("> ").strip()
    
    if not user_prompt:
        print("输入为空，使用默认主题")
        user_prompt = "创作一个关于友谊和冒险的温馨故事"
    
    # 选择 Provider
    print("\n使用默认 Provider (mock)")
    provider_config = ProviderConfig(
        text_provider="mock",
        image_provider="mock",
        video_provider="mock",
        audio_provider="mock",
    )
    
    # 创建运行器
    runner = ComicWorkflowRunner(use_checkpointer=True)
    
    print("\n开始生成漫剧...")
    print("-" * 40)
    
    # 运行工作流
    gen = runner.run_interactive(
        user_prompt=user_prompt,
        provider_config=provider_config,
        style_preferences={"art_style": "anime illustration"},
    )
    
    try:
        state = next(gen)
        
        while state.get("status") == WorkflowStatus.WAITING_INPUT:
            # 图片选择交互
            if state.get("current_interaction") == InteractionPoint.IMAGE_SELECTION:
                selections = interactive_image_selection(state)
                
                if not selections:
                    print("已取消")
                    return
                
                state = gen.send({"selections": selections})
            else:
                # 其他交互点
                print(f"等待输入: {state.get('current_interaction')}")
                user_input = input("> ").strip()
                state = gen.send({"input": user_input})
        
        print_result(state)
        
    except StopIteration as e:
        if e.value:
            print_result(e.value)


def run_auto_mode():
    """自动模式"""
    print("\n" + "=" * 60)
    print("漫剧工作流 - 自动模式")
    print("=" * 60)
    
    # 获取用户输入
    print("\n请输入您想创作的漫剧主题:")
    user_prompt = input("> ").strip()
    
    if not user_prompt:
        user_prompt = "创作一个关于友谊和冒险的温馨故事"
    
    provider_config = ProviderConfig(
        text_provider="mock",
        image_provider="mock",
        video_provider="mock",
        audio_provider="mock",
    )
    
    runner = ComicWorkflowRunner(use_checkpointer=False)
    
    print("\n开始自动生成漫剧...")
    print("-" * 40)
    
    result = runner.run_auto(
        user_prompt=user_prompt,
        provider_config=provider_config,
    )
    
    print_result(result)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="漫剧工作流命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python cli.py --interactive    # 交互模式
  python cli.py --auto          # 自动模式
  python cli.py --providers     # 列出可用 Provider
  python cli.py --graph         # 显示工作流结构
        """
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="交互模式（支持图片选择）"
    )
    parser.add_argument(
        "-a", "--auto",
        action="store_true",
        help="自动模式（一键生成）"
    )
    parser.add_argument(
        "-p", "--providers",
        action="store_true",
        help="列出可用的 Provider"
    )
    parser.add_argument(
        "-g", "--graph",
        action="store_true",
        help="显示工作流结构"
    )
    
    args = parser.parse_args()
    
    # 确保导入 Provider（触发注册）
    from multimodal_agent.services.providers.text import mock_provider
    from multimodal_agent.services.providers.image import mock_provider as img_mock
    from multimodal_agent.services.providers.video import mock_provider as vid_mock
    from multimodal_agent.services.providers.audio import mock_provider as aud_mock
    
    if args.graph:
        print(print_comic_workflow_structure())
        return
    
    if args.providers:
        print_providers()
        return
    
    if args.auto:
        run_auto_mode()
        return
    
    if args.interactive:
        run_interactive_mode()
        return
    
    # 默认显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()
