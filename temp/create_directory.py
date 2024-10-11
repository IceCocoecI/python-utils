
import webbrowser
import markdown
from markdown.extensions.toc import TocExtension

catalog = {
    "1. StableDiffusion": [
        "【入门教程】",
        "  - StableDiffusion 小白教程——WebUI 初学者操作指南",
        "  - StableDiffusion 进阶教程——深度探索 18 问",
        "  - 提示词指南",
        "  - 采样器比较与分析",
        "  - 高清放大算法",
        "  - SegmentAnything 一键抠图与重绘",
        "  - 艺术家风格研究",
        "【进阶教程】",
        "  - StableDiffusion 扩散模型原理分析",
        "  - StableDiffusion Controlnet 原理探究",
        "  - Controlnet 应用",
        "  - 视频转动漫"
    ],
    "2. Lora 训练专题": [
        "Lora 模型训练底层原理",
        "  - 训练技巧与参数调整",
        "  - 实际案例展示"
    ],
    "3. ComfyUI 工作流系列": [
        "模特换装技术解析与实践",
        "  - 人物换脸的实现方法与效果优化",
        "  - 高清修复的技巧与应用",
        "  - 智能抠图的原理与操作指南"
    ],
    "4. AI 视频生成技术": [
        "AnimateDiff 技术原理与特点",
        "  - stable-diffusion-videos 的技术原理与特点",
        "  - 基于图片生成视频的方法与技巧",
        "  - 依据文字生成视频的流程与优化",
        "  - 其他 AI 视频生成工具介绍"
    ],
    "5. AI 音频技术": [
        "TTS（文字转语音）",
        "  - ASR（语音转文字）",
        "  - 声音克隆技术详解",
        "  - AI 音频技术应用——AI 配音",
        "  - AI 音频技术应用——AI 翻唱"
    ],
    "6. ChatGPT 与大语言模型": [
        "大模型的基础知识",
        "大语言模型的基础使用教程",
        "  - 工作文档润色实例",
        "  - 日常数据处理方法",
        "  - 如何让 GPT 帮你写代码？",
        "大语言模型的应用和二次开发",
        "  - 编程应用入门",
        "  - 插件开发与应用案例"
    ],
    "7. AI 工具分享": [
        "AI 办公工具推荐与使用技巧",
        "  - AI 写作工具推荐与使用技巧",
        "  - AI 绘画工具推荐与使用技巧",
        "  - AI 音频、视频领域的 AI 工具盘点"
    ],
    "8. 行业案例与副业变现": [
        "副业变现的途径与方法",
        "  - 信息差的挖掘与利用",
        "  - 实战经验分享与交流"
    ],
    "9. 交流与互动": [
        "用户经验分享与交流板块",
        "  - 定期的技术更新与趋势解读"
    ]

}


# 生成 Markdown 格式的目录
markdown_content = "# 目录\n"
for main_title, subsections in catalog.items():
    markdown_content += f'## {main_title}\n'
    for subsection in subsections:
        if subsection.startswith("  - "):
            markdown_content += f'{subsection}\n'
        else:
            markdown_content += f'   - {subsection}\n'

# 将 Markdown 转换为 HTML
html_content = markdown.markdown(markdown_content, extensions=[TocExtension()])

# 添加简约风格的 CSS 样式和目录内容
styled_html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body, h1, h2, ul, li, p {{
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
        }}
        body {{
            background-color: #f9f9f9;
            color: #333;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            background-color: #fff;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        h1, h2 {{
            margin-bottom: 20px;
            color: #333;
        }}
        ul {{
            list-style-type: none;
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
<div class="container">
    {html_content}
</div>
</body>
</html>
"""

# 保存为 HTML 文件，并指定编码为 utf-8
with open('catalog.html', 'w', encoding='utf-8') as f:
    f.write(styled_html_content)

# 在浏览器中打开生成的 HTML 文件
webbrowser.open('catalog.html')