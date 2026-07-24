# 环境与运行方式

## 1. 当前环境

本课程使用你现有的 Conda 环境：

```text
环境名: langgraph
路径:   /home/cz/anaconda3/envs/langgraph
Python: 3.11.14
核心包: langgraph 1.0.6
```

从仓库根目录运行：

```bash
cd /home/cz/software/pycharm-2025.3.4/pycharm-projects/python-utils
conda activate langgraph
```

快速检查：

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
python -c "import importlib.metadata as m; print(m.version('langgraph'))"
```

预期 Python 路径是 `/home/cz/anaconda3/envs/langgraph/bin/python`。

## 2. 为什么必须从仓库根目录运行

顶层学习目录也叫 `langgraph/`。它没有 `__init__.py`，因此不会主动成为一个普通 Python 包；从仓库根目录直接运行课程脚本时，Python 仍会加载 Conda 环境里的官方包。

推荐：

```bash
python langgraph/04-runtime-context/examples/request_context.py
```

不要在顶层添加以下文件：

```text
langgraph/__init__.py
```

否则 `import langgraph` 可能优先加载本地课程目录，出现 `No module named 'langgraph.graph'` 等遮蔽问题。

## 3. 安装与同步

当前环境已经满足课程要求，不需要安装额外依赖。若在另一台机器复现：

```bash
conda create -n langgraph python=3.11 -y
conda activate langgraph
python -m pip install -r langgraph/requirements.txt
```

`requirements.txt` 只包含离线课程的核心依赖。真实模型 provider 是可选扩展，不应为了学习图运行时而提前安装。

## 4. 能力矩阵

| 能力 | 是否需要网络 | 是否需要 API Key | 默认课程是否覆盖 |
|---|---:|---:|---:|
| StateGraph / reducer / routing | 否 | 否 | 是 |
| checkpoint / interrupt / Store | 否 | 否 | 是 |
| ToolNode 协议 | 否 | 否 | 是 |
| stream / subgraph / Send | 否 | 否 | 是 |
| 综合客服工作流 | 否 | 否 | 是 |
| OpenAI 或其他远程模型 | 是 | 是 | 否，可自行替换节点 |
| LangSmith 远程追踪 | 是 | 通常需要 | 否，课程使用本地 trace |
| 数据库型 checkpoint/store | 视后端而定 | 视后端而定 | 只讲接口和迁移边界 |

## 5. 验证层级

从快到慢执行：

```bash
# 1. 一个脚本
python langgraph/01-foundations/examples/hello_state_graph.py --self-test

# 2. 全部示例
python langgraph/scripts/smoke_test.py

# 升级依赖时，把弃用等警告视为错误
python langgraph/scripts/smoke_test.py --warnings-as-errors

# 3. 行为测试
python -m unittest discover -s langgraph/tests -v

# 4. 文档链接
python langgraph/scripts/check_links.py
```

## 6. IDE 配置

PyCharm 解释器选择：

```text
/home/cz/anaconda3/envs/langgraph/bin/python
```

Working directory 设置为：

```text
/home/cz/software/pycharm-2025.3.4/pycharm-projects/python-utils
```

不要把某个 `examples/` 子目录设置成全局 source root；示例故意保持局部、可独立运行，综合项目会在测试中显式加入自己的包路径。

## 7. 可选真实模型接入

接入真实模型时，建议另建文件或复制综合 Lab，而不是修改离线基线。基本替换边界是：

```text
deterministic planner node
            |
            v
real chat model with tools
```

保留原有状态 schema、tool validation、checkpoint、interrupt 和测试，用少量固定响应或 fake model 测协议，再用单独的集成测试访问真实服务。
