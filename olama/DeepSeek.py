import gradio as gr
import requests


# 定义 API 的请求函数
def chat_with_model(prompt):
    # API 的 URL
    api_url = "http://localhost:11434/api/generate"

    # 请求头和数据体
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-r1:7b",
        "prompt": prompt,
        "stream": False
    }

    try:
        # 调用 API
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()  # 如果有错误会抛出异常
        result = response.json()  # 假设返回的是 JSON 格式
        return result.get("response", "无法解析响应")  # 返回 API 的内容
    except requests.exceptions.RequestException as e:
        return f"请求失败: {e}"


# 定义 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# 🌟 DeepSeek模型聊天界面")
    gr.Markdown("这是一个基于 Gradio 的聊天界面，支持与大语言模型交互。")

    # 聊天历史记录
    chatbot = gr.Chatbot(label="聊天记录")
    with gr.Row():
        user_input = gr.Textbox(placeholder="请输入您的问题...", label="输入框", lines=1)
        submit_btn = gr.Button("发送")


    # 处理聊天逻辑
    # Initialize a dictionary to store chat histories by user IP
    user_histories = {}


    def user_chat(history, user_message, user_ip):
        if not user_message.strip():
            return history, "请输入有效的问题"

        # Initialize chat history for the user if not already present
        if user_ip not in user_histories:
            user_histories[user_ip] = []

        # Call the model to generate a reply
        model_reply = chat_with_model(user_message)

        # Update the user's chat history
        user_histories[user_ip].append(("用户", user_message))
        user_histories[user_ip].append(("模型", model_reply))

        # Return the updated history for the specific user
        return user_histories[user_ip], ""


    # 绑定事件
    submit_btn.click(user_chat, inputs=[chatbot, user_input], outputs=[chatbot, user_input])

# 运行 Gradio 应用，设置外部可访问
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)