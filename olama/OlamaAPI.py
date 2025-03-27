import requests
import json

# 定义请求的URL和数据
url = 'http://localhost:11434/api/generate'
headers = {'Content-Type': 'application/json'}
data = {
    "model": "llama3.2",
    "prompt": "Why is the sky blue?"
}

# 发送POST请求并启用流式传输
response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)

# 逐行读取流式响应
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))