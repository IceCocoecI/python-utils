"""
Agent 集成测试

测试多模态内容生成 Agent 的完整工作流。
"""

import sys
from pathlib import Path

# 添加父目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from langchain_core.messages import HumanMessage

from multimodal_agent.state import AgentState, TaskType, create_initial_state
from multimodal_agent.graph import create_multimodal_agent
from multimodal_agent.logging_config import setup_logging

# 初始化日志（测试时使用 console 格式）
setup_logging()


class TestMultimodalAgent:
    """多模态 Agent 测试类"""
    
    @pytest.fixture
    def agent(self):
        """创建 Agent 实例"""
        return create_multimodal_agent()
    
    def _create_state(self, user_input: str) -> AgentState:
        """创建测试状态"""
        state = create_initial_state(user_input)
        state["messages"] = [HumanMessage(content=user_input)]
        return state
    
    def test_text_generation_intent(self, agent):
        """测试文本生成意图识别"""
        test_inputs = [
            "请写一篇关于Python的文章",
            "帮我写一个故事",
            "生成一段代码",
            "总结这段文字",
        ]
        
        for user_input in test_inputs:
            state = self._create_state(user_input)
            result = agent.invoke(state)
            
            # 验证任务类型为文本
            assert result["task_type"] == TaskType.TEXT, \
                f"输入 '{user_input}' 应该被识别为文本任务"
            
            # 验证有生成内容
            assert result["generated_content"] is not None, \
                f"输入 '{user_input}' 应该有生成内容"
            
            # 验证完成标记
            assert result["is_completed"] is True
    
    def test_image_generation_intent(self, agent):
        """测试图片生成意图识别"""
        test_inputs = [
            "画一张风景画",
            "生成一张图片",
            "帮我设计一个logo",
            "绘制一张海报",
        ]
        
        for user_input in test_inputs:
            state = self._create_state(user_input)
            result = agent.invoke(state)
            
            # 验证任务类型为图片
            assert result["task_type"] == TaskType.IMAGE, \
                f"输入 '{user_input}' 应该被识别为图片任务"
            
            # 验证有生成内容（URL）
            assert result["generated_content"] is not None
            assert "cdn.example.com" in result["generated_content"].content
    
    def test_video_generation_intent(self, agent):
        """测试视频生成意图识别"""
        test_inputs = [
            "生成一个视频",
            "制作一段动画",
            "创建一个短视频",
        ]
        
        for user_input in test_inputs:
            state = self._create_state(user_input)
            result = agent.invoke(state)
            
            # 验证任务类型为视频
            assert result["task_type"] == TaskType.VIDEO, \
                f"输入 '{user_input}' 应该被识别为视频任务"
            
            # 验证有生成内容（URL）
            assert result["generated_content"] is not None
            assert result["generated_content"].content.endswith(".mp4")
    
    def test_messages_accumulation(self, agent):
        """测试消息累积"""
        state = self._create_state("写一篇文章")
        result = agent.invoke(state)
        
        # 应该至少有用户消息和 AI 响应
        assert len(result["messages"]) >= 2
        
        # 第一条应该是用户消息
        assert result["messages"][0].content == "写一篇文章"
    
    def test_error_handling(self, agent):
        """测试错误处理"""
        # 空输入应该能正常处理
        state = self._create_state("")
        result = agent.invoke(state)
        
        # 应该完成（可能有错误）
        assert result["is_completed"] is True
    
    def test_state_structure(self, agent):
        """测试状态结构完整性"""
        state = self._create_state("测试输入")
        result = agent.invoke(state)
        
        # 验证所有必需字段存在
        assert "messages" in result
        assert "user_input" in result
        assert "task_type" in result
        assert "is_completed" in result
        
        # 验证状态值类型
        assert isinstance(result["messages"], list)
        assert isinstance(result["is_completed"], bool)


class TestContentGeneration:
    """内容生成测试类"""
    
    @pytest.fixture
    def agent(self):
        return create_multimodal_agent()
    
    def test_text_content_structure(self, agent):
        """测试文本生成内容结构"""
        state = create_initial_state("写一篇短文")
        state["messages"] = [HumanMessage(content="写一篇短文")]
        result = agent.invoke(state)
        
        content = result["generated_content"]
        assert content is not None
        assert content.task_type == TaskType.TEXT
        assert content.content is not None
        assert len(content.content) > 0
        assert "request_id" in content.metadata
    
    def test_image_content_structure(self, agent):
        """测试图片生成内容结构"""
        state = create_initial_state("画一张图")
        state["messages"] = [HumanMessage(content="画一张图")]
        result = agent.invoke(state)
        
        content = result["generated_content"]
        assert content is not None
        assert content.task_type == TaskType.IMAGE
        assert "https://" in content.content
        assert "size" in content.metadata
        assert "style" in content.metadata
    
    def test_video_content_structure(self, agent):
        """测试视频生成内容结构"""
        state = create_initial_state("制作一个视频")
        state["messages"] = [HumanMessage(content="制作一个视频")]
        result = agent.invoke(state)
        
        content = result["generated_content"]
        assert content is not None
        assert content.task_type == TaskType.VIDEO
        assert content.content.endswith(".mp4")
        assert "duration" in content.metadata
        assert "resolution" in content.metadata


def run_quick_test():
    """快速测试函数"""
    print("\n" + "=" * 60)
    print("运行快速测试")
    print("=" * 60)
    
    agent = create_multimodal_agent()
    
    test_cases = [
        ("文本测试", "请写一段关于AI的介绍"),
        ("图片测试", "画一张山水画"),
        ("视频测试", "生成一个产品展示视频"),
    ]
    
    for name, user_input in test_cases:
        print(f"\n测试: {name}")
        print(f"输入: {user_input}")
        
        state = create_initial_state(user_input)
        state["messages"] = [HumanMessage(content=user_input)]
        
        result = agent.invoke(state)
        
        print(f"任务类型: {result['task_type']}")
        print(f"是否完成: {result['is_completed']}")
        if result.get("generated_content"):
            print(f"生成内容: {result['generated_content'].content[:100]}...")
        if result.get("error_message"):
            print(f"错误信息: {result['error_message'].message}")
        
        print("-" * 40)
    
    print("\n测试完成!")


if __name__ == "__main__":
    run_quick_test()
