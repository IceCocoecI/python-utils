# 04 · Demo 与前端展示

> 再强的模型，如果别人看不到效果，就等于不存在。
> 一个好的 Demo 能在 5 分钟内征服你的老板、投资人或论文评审。

---

## 本地可运行示例

当前 `aigc` 环境未安装 Gradio / Streamlit，因此本章示例提供了无 UI 依赖的 self-test，以及安装可选依赖后的启动方式：

```bash
cd aigc-learning/07-inference-and-deployment/examples

# 验证 toy chat 和流式输出逻辑
conda run -n aigc python demo_apps.py --mode self-test
```

安装可选依赖后启动 Gradio：

```bash
pip install gradio
conda run -n aigc python demo_apps.py --mode gradio
```

安装可选依赖后启动 Streamlit：

```bash
pip install streamlit
conda run -n aigc streamlit run demo_apps.py -- --mode streamlit
```

对应代码：[`examples/demo_apps.py`](./examples/demo_apps.py)

---

## 1. 为什么 Demo 重要？

| 角色 | 需要 Demo 的原因 |
|---|---|
| 算法工程师 | 快速验证模型效果，迭代 prompt/参数 |
| 团队 Leader | 向上汇报进展，展示阶段性成果 |
| 产品经理 | 理解模型能力边界，设计产品功能 |
| 外部用户 | 在 HuggingFace Spaces 上试用你的模型 |
| 论文作者 | 附一个在线 Demo 让评审直接体验 |

**核心原则**：Demo 的目标不是"功能完整"，而是**用最少的代码让模型效果可视化**。

---

## 2. Gradio：最快搭 ML Demo 的方式

### 2.1 为什么选 Gradio

- **HuggingFace 官方**：与 HF 生态深度集成。
- **极简 API**：3 行代码搭建 Demo。
- **一键分享**：`share=True` 生成公网 URL。
- **原生支持流式**：LLM 逐 token 输出、图像实时生成。
- **ChatInterface**：专为聊天场景设计的组件。

### 2.2 安装

```bash
pip install gradio
```

### 2.3 `Interface`：最简单的入门

```python
import gradio as gr


def greet(name: str, intensity: int) -> str:
    return "Hello, " + name + "!" * intensity


demo = gr.Interface(
    fn=greet,
    inputs=[
        gr.Textbox(label="Name"),
        gr.Slider(1, 10, value=3, step=1, label="Intensity"),
    ],
    outputs=gr.Textbox(label="Greeting"),
    title="Hello World Demo",
    description="一个简单的打招呼 Demo",
)

demo.launch()
```

> 运行后打开 `http://localhost:7860`，你会看到一个带输入框、滑块和输出框的 Web 界面。

### 2.4 `ChatInterface`：LLM 聊天

```python
import gradio as gr
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")


def chat(message: str, history: list[dict]) -> str:
    messages = history + [{"role": "user", "content": message}]

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content


demo = gr.ChatInterface(
    fn=chat,
    title="AI 聊天助手",
    description="基于 Qwen2.5 的对话 Demo",
    examples=["你好，介绍一下你自己", "用 Python 写一个快速排序", "解释什么是 Transformer"],
    type="messages",
)

demo.launch()
```

### 2.5 流式聊天（逐 token 输出）

```python
import gradio as gr
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")


def chat_stream(message: str, history: list[dict]):
    messages = history + [{"role": "user", "content": message}]

    stream = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
        stream=True,
    )

    partial = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            partial += chunk.choices[0].delta.content
            yield partial


demo = gr.ChatInterface(
    fn=chat_stream,
    title="AI 聊天助手（流式）",
    type="messages",
)

demo.launch()
```

> **关键**：`yield` 而不是 `return`。Gradio 会自动识别生成器函数并启用流式显示。

### 2.6 图像生成 Demo

```python
import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")


def generate_image(
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int,
):
    generator = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    return image


demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", lines=3, placeholder="描述你想生成的图像..."),
        gr.Textbox(label="Negative Prompt", lines=2, placeholder="不想出现的内容..."),
        gr.Slider(5, 50, value=25, step=1, label="Steps"),
        gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale"),
        gr.Number(label="Seed (-1 = random)", value=-1, precision=0),
    ],
    outputs=gr.Image(label="Generated Image", type="pil"),
    title="SDXL 文生图",
    allow_flagging="never",
)

demo.launch()
```

---

## 3. Gradio Blocks：自由布局

`Interface` 适合快速原型，`Blocks` 则给你完全的布局控制。

### 3.1 基本结构

```python
import gradio as gr

with gr.Blocks(title="My App", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 AI 创作工作台")

    with gr.Tabs():
        with gr.Tab("文本生成"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(label="Prompt", lines=4)
                    temperature = gr.Slider(0, 2, value=0.7, label="Temperature")
                    generate_btn = gr.Button("生成", variant="primary")
                with gr.Column(scale=3):
                    output = gr.Textbox(label="生成结果", lines=10)

        with gr.Tab("图像生成"):
            with gr.Row():
                img_prompt = gr.Textbox(label="Prompt")
                img_output = gr.Image(label="生成图像")
            img_btn = gr.Button("生成图像")

    generate_btn.click(fn=generate_text, inputs=[prompt, temperature], outputs=output)
    img_btn.click(fn=generate_image, inputs=img_prompt, outputs=img_output)

demo.launch()
```

### 3.2 关键布局组件

| 组件 | 作用 |
|---|---|
| `gr.Row()` | 水平排列 |
| `gr.Column(scale=N)` | 垂直排列，scale 控制宽度比例 |
| `gr.Tabs()` / `gr.Tab()` | 选项卡 |
| `gr.Accordion()` | 可折叠面板（放高级参数） |
| `gr.Group()` | 视觉分组 |
| `gr.Markdown()` | 渲染 Markdown 文本 |

### 3.3 事件绑定

```python
# 按钮点击
btn.click(fn=process, inputs=[input1, input2], outputs=output)

# 输入变化时触发
slider.change(fn=update, inputs=slider, outputs=display)

# 流式输出
btn.click(fn=stream_fn, inputs=prompt, outputs=output)

# 多输出
btn.click(fn=multi_output, inputs=prompt, outputs=[text_out, image_out, audio_out])
```

---

## 4. Gradio 高级特性

### 4.1 自定义 CSS / 主题

```python
custom_css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}

#main-title {
    text-align: center;
    color: #2563eb;
}

.output-box {
    font-family: 'Fira Code', monospace;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI 助手", elem_id="main-title")
    output = gr.Textbox(elem_classes=["output-box"])
```

### 4.2 内置主题

```python
# Gradio 提供多种内置主题
gr.themes.Default()
gr.themes.Soft()
gr.themes.Monochrome()
gr.themes.Glass()
gr.themes.Base()

# 自定义主题
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    font=gr.themes.GoogleFont("Inter"),
)

demo = gr.Blocks(theme=theme)
```

### 4.3 一键分享

```python
# 生成公网可访问的 URL（72 小时有效）
demo.launch(share=True)
# 输出类似：Running on public URL: https://xxxxx.gradio.live

# 指定端口
demo.launch(server_port=7860)

# 允许外部访问
demo.launch(server_name="0.0.0.0")
```

### 4.4 认证

```python
# 简单用户名密码认证
demo.launch(auth=("admin", "password123"))

# 多用户
demo.launch(auth=[("user1", "pass1"), ("user2", "pass2")])

# 自定义认证函数
def auth_fn(username, password):
    return username == "admin" and password == "secret"

demo.launch(auth=auth_fn)
```

---

## 5. Streamlit：数据应用 / 交互式 Demo

### 5.1 核心理念

Streamlit 的设计哲学是**"脚本即应用"**——写一个从上到下的 Python 脚本，Streamlit 自动把它变成 Web 应用。

### 5.2 安装与运行

```bash
pip install streamlit

# 运行（不是 python xxx.py，而是 streamlit run）
streamlit run app.py
```

### 5.3 基础示例

```python
# app.py
import streamlit as st

st.title("AI 文本分析")
st.write("输入文本，AI 帮你分析情感和关键词。")

text = st.text_area("输入文本", height=150, placeholder="在这里输入...")

col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
with col2:
    max_tokens = st.number_input("Max Tokens", 50, 2000, 512)

if st.button("分析", type="primary"):
    if not text:
        st.warning("请输入文本")
    else:
        with st.spinner("分析中..."):
            result = analyze(text, temperature, max_tokens)
        st.success("分析完成！")
        st.json(result)
```

### 5.4 聊天界面

```python
import streamlit as st
from openai import OpenAI

st.title("AI 聊天助手")

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("输入消息..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=st.session_state.messages,
            stream=True,
        )
        response = st.write_stream(
            chunk.choices[0].delta.content or ""
            for chunk in stream
            if chunk.choices[0].delta.content
        )

    st.session_state.messages.append({"role": "assistant", "content": response})
```

### 5.5 状态管理

```python
# Streamlit 每次交互都会重新运行整个脚本
# 用 session_state 保存跨次运行的状态

if "counter" not in st.session_state:
    st.session_state.counter = 0

if st.button("➕"):
    st.session_state.counter += 1

st.write(f"Count: {st.session_state.counter}")
```

### 5.6 缓存

```python
# @st.cache_resource：缓存全局资源（模型、数据库连接）
@st.cache_resource
def load_model():
    from transformers import pipeline
    return pipeline("text-generation", model="Qwen/Qwen2.5-7B-Instruct",
                    torch_dtype="float16", device_map="auto")


# @st.cache_data：缓存数据（函数返回值）
@st.cache_data(ttl=3600)
def load_data(url: str):
    import pandas as pd
    return pd.read_csv(url)
```

### 5.7 布局

```python
# 侧边栏
with st.sidebar:
    st.header("设置")
    model = st.selectbox("模型", ["Qwen2.5-7B", "Qwen2.5-14B"])
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)

# 列布局
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.text_input("输入")
with col2:
    st.metric("延迟", "120ms", "-30ms")
with col3:
    st.metric("吞吐", "150 tok/s", "+20%")

# 选项卡
tab1, tab2 = st.tabs(["生成", "历史"])
with tab1:
    st.write("生成内容")
with tab2:
    st.write("历史记录")

# 可展开区域
with st.expander("高级设置"):
    top_p = st.slider("Top P", 0.0, 1.0, 0.9)
    rep_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.1)
```

---

## 6. Streamlit vs Gradio

| 维度 | Gradio | Streamlit |
|---|---|---|
| **定位** | ML Demo | 数据应用 / Dashboard |
| **上手速度** | ★★★★★ | ★★★★ |
| **布局灵活性** | ★★★★ (Blocks) | ★★★★★ |
| **ML 组件** | 丰富（音频、图像、视频、3D） | 基础 |
| **流式支持** | 原生（yield） | `st.write_stream` |
| **聊天 UI** | `ChatInterface` | `st.chat_message` + `st.chat_input` |
| **状态管理** | 事件驱动 | `session_state`（脚本重跑） |
| **分享** | `share=True`（公网链接） | Streamlit Cloud |
| **HF 集成** | 原生（Spaces） | 支持但非原生 |
| **自定义 CSS** | 支持 | 有限 |
| **数据可视化** | 基础 | 丰富（原生图表、Plotly 集成） |
| **最适合** | ML 模型 Demo | 数据分析 Dashboard |

> **经验法则**：展示 ML 模型用 Gradio，做数据产品/仪表盘用 Streamlit。

---

## 7. 部署

### 7.1 Gradio → HuggingFace Spaces

```
# 1. 在 HuggingFace 创建 Space（选择 Gradio SDK）
# 2. 创建以下文件结构：

my-space/
├── app.py              # Gradio 代码
└── requirements.txt    # 依赖
```

```python
# app.py（HuggingFace Spaces 版本）
import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")


def chat(message, history):
    messages = history + [{"role": "user", "content": message}]
    partial = ""
    for token in client.chat_completion(messages, max_tokens=512, stream=True):
        partial += token.choices[0].delta.content or ""
        yield partial


demo = gr.ChatInterface(fn=chat, title="Qwen Chat", type="messages")
demo.launch()
```

```bash
# 推送到 HuggingFace
git init
git remote add origin https://huggingface.co/spaces/your-username/your-space
git add .
git commit -m "Initial commit"
git push origin main
```

### 7.2 Streamlit Cloud

```bash
# 1. 把代码推到 GitHub
# 2. 在 streamlit.io 连接 GitHub 仓库
# 3. 选择 app.py 作为入口
# 4. 自动部署
```

### 7.3 Docker 部署

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Gradio
EXPOSE 7860
CMD ["python", "app.py"]

# 或 Streamlit
# EXPOSE 8501
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 8. Open WebUI 和其他 LLM 前端

### 8.1 Open WebUI

Open WebUI（原 Ollama WebUI）是一个功能丰富的 ChatGPT 风格前端：

- 支持 Ollama、OpenAI 兼容 API（包括 vLLM、SGLang）
- 多模型切换、对话管理、RAG、图像生成
- 用户管理、权限控制
- 可本地部署

```bash
# Docker 一键部署（连接 vLLM）
docker run -d -p 3000:8080 \
    -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
    -e OPENAI_API_KEY=dummy \
    -v open-webui:/app/backend/data \
    --name open-webui \
    ghcr.io/open-webui/open-webui:main
```

### 8.2 其他前端工具

| 工具 | 定位 | 特点 |
|---|---|---|
| **Open WebUI** | 全功能 ChatGPT 替代 | 多模型、RAG、用户管理 |
| **Chatbot UI** | 简洁 ChatGPT 克隆 | 轻量、易部署 |
| **LobeChat** | 现代 AI 聊天 | 插件生态、多模态 |
| **Jan** | 桌面 AI 助手 | 本地运行、隐私友好 |

---

## 9. Chainlit：Agent UI

Chainlit 专为 LLM Agent / Chain 场景设计：

```python
# app.py
import chainlit as cl
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")


@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="")
    await msg.send()

    stream = await client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手。"},
            {"role": "user", "content": message.content},
        ],
        stream=True,
    )

    async for chunk in stream:
        if token := chunk.choices[0].delta.content:
            await msg.stream_token(token)

    await msg.update()
```

```bash
pip install chainlit
chainlit run app.py
```

**Chainlit 特色**：
- 原生显示 Agent 的思考过程（Steps）
- 支持文件上传和多模态
- 可嵌入到现有 Web 应用中

---

## 10. 完整示例：多模态 Gradio Blocks Demo

```python
"""
多模态 AI 工作台：文本生成 + 图像生成 + 图像理解。
"""

import gradio as gr
import torch
from openai import OpenAI
from diffusers import AutoPipelineForText2Image

# --- 模型加载 ---

llm_client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

img_pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
).to("cuda")


# --- 功能函数 ---

def chat_stream(message, history, system_prompt, temperature):
    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({"role": "user", "content": message})

    stream = llm_client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
        stream=True,
    )

    partial = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            partial += chunk.choices[0].delta.content
            yield partial


def generate_image(prompt, steps, seed):
    generator = None
    if seed >= 0:
        generator = torch.Generator("cuda").manual_seed(int(seed))

    image = img_pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=0.0,
        generator=generator,
    ).images[0]

    return image


# --- 界面构建 ---

custom_css = """
.gradio-container { max-width: 1100px !important; margin: auto !important; }
"""

with gr.Blocks(
    title="AI 工作台",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=custom_css,
) as demo:

    gr.Markdown(
        """
        # AI 创作工作台
        集成文本对话与图像生成能力。
        """
    )

    with gr.Tabs():
        # --- Tab 1: 聊天 ---
        with gr.Tab("💬 对话"):
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.ChatInterface(
                        fn=chat_stream,
                        additional_inputs=[
                            gr.Textbox(
                                value="你是一个有帮助的 AI 助手。",
                                label="系统提示词",
                            ),
                            gr.Slider(0, 2, value=0.7, label="Temperature"),
                        ],
                        type="messages",
                    )

        # --- Tab 2: 图像生成 ---
        with gr.Tab("🎨 图像生成"):
            with gr.Row():
                with gr.Column(scale=2):
                    img_prompt = gr.Textbox(
                        label="Prompt",
                        lines=3,
                        placeholder="Describe the image you want to generate...",
                    )
                    with gr.Row():
                        img_steps = gr.Slider(1, 8, value=4, step=1, label="Steps")
                        img_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                    img_btn = gr.Button("🎨 生成", variant="primary", size="lg")

                with gr.Column(scale=3):
                    img_output = gr.Image(label="生成结果", height=512)

            img_btn.click(
                fn=generate_image,
                inputs=[img_prompt, img_steps, img_seed],
                outputs=img_output,
            )

            gr.Examples(
                examples=[
                    ["a cute cat astronaut floating in space, digital art", 4, 42],
                    ["a serene Japanese garden in autumn, watercolor", 4, 123],
                    ["a cyberpunk city at night, neon lights, rain", 4, 7],
                ],
                inputs=[img_prompt, img_steps, img_seed],
            )

    gr.Markdown("---\n*Powered by Qwen2.5 + SDXL-Turbo*")

demo.launch(server_name="0.0.0.0", server_port=7860)
```

---

## 11. 让 Demo 更好看的技巧

### 11.1 Gradio 主题定制

```python
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="gray",
    font=gr.themes.GoogleFont("Noto Sans SC"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    block_background_fill="*neutral_50",
    button_primary_background_fill="*primary_500",
)
```

### 11.2 示例（Examples）

```python
# 预设示例让用户一键体验，大幅降低使用门槛
gr.Examples(
    examples=[
        ["a photo of a cat", 25, 7.5],
        ["a landscape painting", 30, 9.0],
    ],
    inputs=[prompt, steps, guidance],
    outputs=image,
    fn=generate,
    cache_examples=True,  # 预计算示例结果（加速体验）
)
```

### 11.3 进度条

```python
import gradio as gr


def long_task(prompt, progress=gr.Progress()):
    progress(0, desc="加载模型...")
    model = load_model()

    progress(0.3, desc="处理输入...")
    inputs = process(prompt)

    progress(0.5, desc="推理中...")
    result = model.generate(inputs)

    progress(0.9, desc="后处理...")
    output = postprocess(result)

    progress(1.0, desc="完成！")
    return output
```

### 11.4 Streamlit 美化

```python
# 页面配置
st.set_page_config(
    page_title="AI 工作台",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义 CSS
st.markdown("""
<style>
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
.stButton > button {
    width: 100%;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# 指标卡片
col1, col2, col3 = st.columns(3)
col1.metric("模型", "Qwen2.5-7B")
col2.metric("延迟", "120ms", "-15%")
col3.metric("吞吐", "150 tok/s", "+22%")
```

---

## 12. 常见坑

### 12.1 Gradio 版本不兼容

```python
# Gradio 版本迭代快，API 经常变化
# 特别是 3.x → 4.x → 5.x 变化很大

# 建议：固定版本
# pip install gradio==5.0.0

# ChatInterface 的 type 参数：
# Gradio 5.x 推荐 type="messages"（OpenAI 格式）
# 旧版 type="tuples"（已废弃）
```

### 12.2 模型加载在每次请求时重复

```python
# 错误：每次调用都加载模型
def generate(prompt):
    model = load_model()   # 每次请求都加载一遍！
    return model(prompt)

# Gradio 正确做法：在全局加载
model = load_model()

def generate(prompt):
    return model(prompt)

# Streamlit 正确做法：用 @st.cache_resource
@st.cache_resource
def load_model():
    return pipeline("text-generation", ...)
```

### 12.3 Gradio 流式函数返回累积文本

```python
# Gradio 流式要求 yield 累积的完整文本，而不是增量

# 错误：
def stream(prompt):
    for token in generate_tokens(prompt):
        yield token  # 只输出新 token，UI 会跳来跳去

# 正确：
def stream(prompt):
    full_text = ""
    for token in generate_tokens(prompt):
        full_text += token
        yield full_text  # 输出累积的完整文本
```

### 12.4 Streamlit 每次交互重新运行

```python
# Streamlit 的执行模型：每次用户操作都从头执行整个脚本
# 所有变量都会被重置！

# 错误：
counter = 0  # 每次都重置为 0
if st.button("Add"):
    counter += 1  # 永远是 1

# 正确：
if "counter" not in st.session_state:
    st.session_state.counter = 0
if st.button("Add"):
    st.session_state.counter += 1
```

### 12.5 HuggingFace Spaces 超时

```python
# HuggingFace Spaces 免费版有内存和超时限制
# 大模型通常放不下

# 解决方案：
# 1. 使用 HuggingFace Inference API（不用本地加载模型）
from huggingface_hub import InferenceClient
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")

# 2. 使用 ZeroGPU Spaces（免费 GPU，但有排队）
# 3. 升级到付费 Spaces（A10G / T4 / A100）
```

### 12.6 CORS 跨域问题

```python
# 如果前端和后端分离部署，可能遇到 CORS 问题

# FastAPI 解决：
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应指定具体域名
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gradio 默认允许所有来源
```

---

## 13. 小结

| 工具 | 一句话 |
|---|---|
| **Gradio Interface** | 3 行代码搭 Demo |
| **Gradio Blocks** | 自由布局，功能更强 |
| **Gradio ChatInterface** | 聊天场景专用，自带流式 |
| **Streamlit** | 数据应用 / Dashboard 首选 |
| **HuggingFace Spaces** | Gradio/Streamlit 一键上线 |
| **Open WebUI** | 全功能 ChatGPT 替代前端 |
| **Chainlit** | Agent / Chain 场景的专业 UI |

**一句话**：**快速 Demo 用 Gradio，数据产品用 Streamlit，团队内部用 Open WebUI。**
