# 多模态内容生成 Agent

基于 **LangGraph** 的生产级多模态内容生成智能体，支持文本、图片、视频、音频生成。

## 新功能

### 1. 多模型策略模式

每种类别（文本、图片、视频、音频）都支持多个 Provider 实现，可以方便地切换：

```python
from multimodal_agent.services import get_provider, ProviderType

# 使用默认 Provider
text_provider = get_provider(ProviderType.TEXT)

# 指定特定 Provider
image_provider = get_provider(ProviderType.IMAGE, "dalle")

# 列出所有可用 Provider
from multimodal_agent.services import list_providers
providers = list_providers()
```

### 2. 漫剧工作流 (Human-in-the-loop)

支持两种模式：
- **自动模式**: 一键式生成
- **交互模式**: 在关键节点暂停，等待用户选择

```python
from multimodal_agent.comic_workflow import ComicWorkflowRunner

# 自动模式
runner = ComicWorkflowRunner()
result = runner.run_auto("创作一个关于冒险的漫剧")

# 交互模式
gen = runner.run_interactive("创作一个关于友谊的漫剧")
state = next(gen)  # 运行到暂停点
# ... 用户选择图片 ...
state = gen.send({"selections": user_selections})  # 继续执行
```

## 项目结构

```
langgraph/
├── demo/                           # 原有示例代码
│
├── multimodal_agent/               # 多模态 Agent 核心
│   ├── config.py                   # 配置管理
│   ├── logging_config.py           # 日志配置
│   ├── state.py                    # AgentState 定义
│   ├── graph.py                    # 基础 StateGraph
│   ├── main.py                     # 基础入口
│   │
│   ├── services/                   # Service 层
│   │   ├── base.py                # 基础 HTTP 服务
│   │   ├── providers/             # Provider 策略模式 ⭐
│   │   │   ├── registry.py        # Provider 注册表
│   │   │   ├── text/              # 文本 Provider
│   │   │   │   ├── openai_provider.py
│   │   │   │   └── mock_provider.py
│   │   │   ├── image/             # 图片 Provider
│   │   │   │   ├── dalle_provider.py
│   │   │   │   └── mock_provider.py
│   │   │   ├── video/             # 视频 Provider
│   │   │   └── audio/             # 音频 Provider
│   │   └── ...
│   │
│   ├── nodes/                      # 基础 Graph 节点
│   │
│   └── comic_workflow/             # 漫剧工作流 ⭐
│       ├── state.py               # ComicState 状态
│       ├── graph.py               # 漫剧工作流图
│       ├── runner.py              # 运行器（支持 HITL）
│       ├── cli.py                 # 命令行界面
│       └── nodes/                 # 漫剧节点
│           ├── script_generator.py    # 剧本生成
│           ├── scene_parser.py        # 场景解析
│           ├── image_generator.py     # 多图生成
│           ├── interaction_handler.py # 交互处理
│           ├── audio_generator.py     # 配音生成
│           └── video_composer.py      # 视频合成
│
├── .env.example
├── requirements.txt
└── README.md
```

## 快速开始

### 安装依赖

```bash
cd langgraph
pip install -r requirements.txt
```

### 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，配置 API Keys
```

### 运行漫剧工作流

```bash
# 交互模式（支持图片选择）
python -m multimodal_agent.comic_workflow.cli --interactive

# 自动模式（一键生成）
python -m multimodal_agent.comic_workflow.cli --auto

# 查看工作流结构
python -m multimodal_agent.comic_workflow.cli --graph

# 列出可用 Provider
python -m multimodal_agent.comic_workflow.cli --providers
```

## 漫剧工作流详解

### 工作流程

```
START
  │
  ▼
script_generator ──► 根据用户提示生成剧本
  │
  ▼
scene_parser ──────► 解析剧本为场景列表
  │
  ▼
image_generator ───► 为每个场景生成多张候选图片
  │
  ├── (interactive) ──► [INTERRUPT] 等待用户选择
  │                           │
  │                           ▼
  │                     image_selection
  │                           │
  └── (auto) ─────────────────┘
  │
  ▼
audio_generator ───► 为每个场景生成配音
  │
  ▼
video_composer ────► 合成最终视频
  │
  ▼
END
```

### Human-in-the-loop 实现

使用 LangGraph 的 `interrupt_before` 和 `checkpointer`：

```python
# 创建带 checkpointer 的图
compiled = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["image_selection"],  # 在此节点前暂停
)

# 运行时可以暂停和恢复
config = {"configurable": {"thread_id": "unique_id"}}
result = compiled.invoke(initial_state, config)

# 如果暂停了，用户选择后继续
if result["status"] == "waiting_input":
    updated_state = apply_user_selection(result, user_choices)
    result = compiled.invoke(updated_state, config)
```

## Provider 系统

### 注册新的 Provider

```python
from multimodal_agent.services.providers import register_provider
from multimodal_agent.services.providers.base import ProviderType
from multimodal_agent.services.providers.text.base import TextProvider

@register_provider(ProviderType.TEXT, "my_provider", is_default=False)
class MyTextProvider(TextProvider):
    @property
    def info(self):
        return ProviderInfo(
            name="my_provider",
            provider_type=ProviderType.TEXT,
            models=["my-model-v1", "my-model-v2"],
        )
    
    async def generate(self, prompt, config=None, **kwargs):
        # 实现生成逻辑
        pass
```

### 切换默认 Provider

```python
from multimodal_agent.services.providers.registry import set_default_provider
from multimodal_agent.services.providers.base import ProviderType

set_default_provider(ProviderType.IMAGE, "midjourney")
```

## 配置 Provider

在 `ComicState` 或运行时指定：

```python
from multimodal_agent.comic_workflow.state import ProviderConfig

config = ProviderConfig(
    text_provider="openai",
    text_model="gpt-4o",
    image_provider="dalle",
    image_model="dall-e-3",
    video_provider="mock",
    audio_provider="mock",
)

runner.run_auto(prompt, provider_config=config)
```

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 流程编排 | LangGraph | 状态图 + Human-in-the-loop |
| 意图识别 | LangChain-OpenAI | Router 意图分类 |
| HTTP 客户端 | httpx | 异步 HTTP 请求 |
| 数据验证 | Pydantic | 状态和配置验证 |
| 状态持久化 | LangGraph Checkpointer | 工作流暂停/恢复 |
| 日志 | structlog | 结构化日志 |

## License

MIT
