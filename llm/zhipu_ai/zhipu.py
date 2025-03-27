
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# 初始化 LLM 连接
llm = ChatOpenAI(
    temperature=0.8,
    model="GLM-4V-Plus-0111",
    openai_api_key="6f86a5ea17bfe124d68b537de32d9dc2.OyWpsv7MosxWZwTq",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    max_tokens=300
)

# 定义简化后的 Prompt 模板（无历史记录）
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# 构建处理链
runnable_sequence = prompt | llm

# 执行测试调用
if __name__ == "__main__":
    response = runnable_sequence.invoke({"question": "讲个冷笑话"})
    print(response)
    print("AI 回复：", response.content)