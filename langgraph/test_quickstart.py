"""
测试 LangGraph QuickStart 示例

这个文件演示了如何运行 quickstart.py 中的代码。
注意：运行前需要安装必要的依赖包。
"""

import sys
import os

# 添加父目录到路径，以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("测试 LangGraph QuickStart 示例")
print("=" * 60)

# 由于原代码中使用了 init_chat_model，这需要相应的API密钥
# 这里我们创建一个简化的版本进行测试
print("注意：完整的示例需要以下依赖：")
print("  - langchain-core")
print("  - langchain-community")
print("  - langgraph")
print("  - 相应的模型API密钥（如Anthropic、OpenAI等）")

print("\n快速安装命令：")
print("  pip install langchain-core langchain-community langgraph")

print("\n" + "=" * 60)
print("由于缺少API密钥，我们创建一个简化的演示版本...")

# 创建一个简化的演示版本
def simplified_demo():
    """简化的LangGraph工作流演示"""
    print("\nLangGraph 工作流概念演示")
    print("-" * 40)
    
    print("1. 定义状态: MessagesState")
    print("   - messages: 存储对话历史")
    print("   - llm_calls: 跟踪LLM调用次数")
    
    print("\n2. 定义节点:")
    print("   - llm_call: 调用语言模型并决定下一步")
    print("   - tool_node: 执行工具调用")
    
    print("\n3. 工作流流程:")
    print("   开始")
    print("     ↓")
    print("   llm_call（LLM决策）")
    print("     ↓")
    print("   [是否需要工具？] → 是 → tool_node（执行工具）")
    print("     ↓                          ↓")
    print("     否                       返回llm_call")
    print("     ↓")
    print("     结束")
    
    print("\n4. 工具示例:")
    print("   - add(3, 4) = 7")
    print("   - multiply(5, 3) = 15")
    print("   - divide(10, 2) = 5.0")
    
    print("\n5. 实际应用场景:")
    print("   - 数学计算助手")
    print("   - 数据查询工具")
    print("   - 自动化工作流")
    print("   - 多步骤决策系统")
    
    print("\n" + "-" * 40)
    print("要运行完整示例，请执行:")
    print("  python langgraph/quickstart.py")
    
    print("\n注意：你需要：")
    print("  1. 安装依赖包")
    print("  2. 设置相应的API密钥")
    print("  3. 在quickstart.py中更新模型名称")

# 运行简化演示
simplified_demo()

print("\n" + "=" * 60)
print("参考文档：")
print("  - LangGraph官方文档: https://langchain-ai.github.io/langgraph/")
print("  - LangChain工具文档: https://python.langchain.com/docs/modules/tools/")
print("  - 状态管理: https://langchain-ai.github.io/langgraph/concepts/low_level/state/")