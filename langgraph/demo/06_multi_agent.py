"""
LangGraph 基础教程 06: 多智能体协作 (Multi-Agent Systems)
================================================

本示例演示 LangGraph 的多智能体协作能力：
1. Supervisor 模式 - 管理者协调多个专家
2. 协作模式 - 多个 Agent 相互配合
3. 子图模式 - 模块化的 Agent 组合
4. 分层架构 - 多级 Agent 体系

学习目标：
- 理解多 Agent 协作的设计模式
- 掌握 Supervisor 的实现方式
- 学会使用子图进行模块化

应用场景：
- 复杂任务分解
- 专业领域分工
- 团队协作模拟
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import json


# =============================================================================
# 第一部分：Supervisor 模式
# =============================================================================

class SupervisorState(TypedDict):
    """Supervisor 状态"""
    messages: Annotated[list, add_messages]
    task: str
    current_agent: str
    agent_outputs: dict
    final_answer: str


# 定义专家 Agent
class ExpertAgents:
    """专家 Agent 集合"""
    
    @staticmethod
    def researcher(task: str) -> str:
        """研究员：负责搜索和收集信息"""
        # 模拟研究结果
        research_results = {
            "python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建",
            "ai": "人工智能是计算机科学的一个分支，致力于创建智能机器",
            "default": f"关于 '{task}' 的研究：这是一个有趣的话题，需要进一步调查"
        }
        
        for key, value in research_results.items():
            if key in task.lower():
                return f"[研究员] {value}"
        return research_results["default"]
    
    @staticmethod
    def coder(task: str) -> str:
        """程序员：负责编写代码"""
        return f"[程序员] 这是关于 '{task}' 的代码示例：\n```python\n# TODO: 实现 {task}\npass\n```"
    
    @staticmethod
    def writer(task: str) -> str:
        """作家：负责撰写文档"""
        return f"[作家] 关于 '{task}' 的文档草稿：\n\n# {task}\n\n## 介绍\n这是一份关于该主题的详细说明..."
    
    @staticmethod
    def reviewer(content: str) -> str:
        """审核员：负责审核和改进"""
        return f"[审核员] 审核意见：内容质量良好，建议补充更多具体示例。"


def supervisor_node(state: SupervisorState) -> dict:
    """
    Supervisor 节点：决定下一步由哪个 Agent 执行
    
    这是多 Agent 系统的核心：
    1. 分析当前任务状态
    2. 决定需要哪个专家
    3. 或决定任务完成
    """
    task = state["task"]
    agent_outputs = state.get("agent_outputs", {})
    
    # 简单的任务分配逻辑
    # 实际应用中，这里会使用 LLM 来做决策
    
    # 如果还没有研究，先研究
    if "researcher" not in agent_outputs:
        return {"current_agent": "researcher"}
    
    # 如果任务涉及代码，调用程序员
    if "代码" in task or "code" in task.lower():
        if "coder" not in agent_outputs:
            return {"current_agent": "coder"}
    
    # 如果任务涉及文档，调用作家
    if "文档" in task or "document" in task.lower() or "write" in task.lower():
        if "writer" not in agent_outputs:
            return {"current_agent": "writer"}
    
    # 最后由审核员审核
    if "reviewer" not in agent_outputs and len(agent_outputs) > 0:
        return {"current_agent": "reviewer"}
    
    # 所有工作完成
    return {"current_agent": "FINISH"}


def agent_executor(state: SupervisorState) -> dict:
    """执行选定的 Agent"""
    current_agent = state["current_agent"]
    task = state["task"]
    agent_outputs = state.get("agent_outputs", {})
    
    # 调用对应的 Agent
    agents = {
        "researcher": ExpertAgents.researcher,
        "coder": ExpertAgents.coder,
        "writer": ExpertAgents.writer,
        "reviewer": lambda t: ExpertAgents.reviewer(str(agent_outputs))
    }
    
    if current_agent in agents:
        output = agents[current_agent](task)
        agent_outputs[current_agent] = output
        
        return {
            "agent_outputs": agent_outputs,
            "messages": [{"role": "assistant", "content": output}]
        }
    
    return {}


def synthesize_answer(state: SupervisorState) -> dict:
    """综合所有 Agent 的输出，生成最终答案"""
    outputs = state.get("agent_outputs", {})
    
    final_answer = "# 任务完成报告\n\n"
    for agent, output in outputs.items():
        final_answer += f"## {agent.title()} 的贡献\n{output}\n\n"
    
    return {"final_answer": final_answer}


def route_supervisor(state: SupervisorState) -> str:
    """Supervisor 路由函数"""
    current_agent = state.get("current_agent", "")
    
    if current_agent == "FINISH":
        return "synthesize"
    elif current_agent:
        return "execute"
    else:
        return "supervisor"


def build_supervisor_graph():
    """构建 Supervisor 多 Agent 图"""
    graph = StateGraph(SupervisorState)
    
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("execute", agent_executor)
    graph.add_node("synthesize", synthesize_answer)
    
    graph.add_edge(START, "supervisor")
    
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "execute": "execute",
            "synthesize": "synthesize",
            "supervisor": "supervisor"
        }
    )
    
    # 执行后回到 supervisor 决定下一步
    graph.add_edge("execute", "supervisor")
    graph.add_edge("synthesize", END)
    
    return graph.compile()


def demo_supervisor():
    """演示 Supervisor 模式"""
    print("=" * 60)
    print("演示 1: Supervisor 模式")
    print("=" * 60)
    
    app = build_supervisor_graph()
    
    print("\n任务: '写一份关于 Python 的代码文档'")
    print("-" * 40)
    
    for step_num, event in enumerate(app.stream({
        "messages": [],
        "task": "写一份关于 Python 的代码文档",
        "current_agent": "",
        "agent_outputs": {},
        "final_answer": ""
    })):
        for node, output in event.items():
            if node == "supervisor" and output.get("current_agent"):
                print(f"\n步骤 {step_num + 1}: Supervisor 分配任务给 -> {output['current_agent']}")
            elif node == "execute":
                agent = list(output.get("agent_outputs", {}).keys())[-1] if output.get("agent_outputs") else ""
                if agent:
                    print(f"  {agent} 完成工作")


# =============================================================================
# 第二部分：协作模式（链式协作）
# =============================================================================

class CollaborationState(TypedDict):
    """协作状态"""
    original_content: str
    draft: str
    reviewed_draft: str
    final_content: str
    collaboration_log: list[str]


def drafter_agent(state: CollaborationState) -> dict:
    """起草者：创建初稿"""
    content = state["original_content"]
    
    draft = f"""# 草稿
    
基于输入 "{content}"，我创建了以下内容：

1. 引言：介绍主题背景
2. 主体：详细阐述核心内容
3. 结论：总结要点

[草稿完成 - 等待审核]"""
    
    return {
        "draft": draft,
        "collaboration_log": state.get("collaboration_log", []) + ["起草者完成初稿"]
    }


def reviewer_agent(state: CollaborationState) -> dict:
    """审核者：审核并提出修改建议"""
    draft = state["draft"]
    
    reviewed = f"""{draft}

---
## 审核意见

1. 整体结构良好
2. 建议：增加具体示例
3. 建议：优化段落过渡
4. 状态：批准发布（需少量修改）

[审核完成 - 已批准]"""
    
    return {
        "reviewed_draft": reviewed,
        "collaboration_log": state["collaboration_log"] + ["审核者完成审核"]
    }


def editor_agent(state: CollaborationState) -> dict:
    """编辑者：根据审核意见进行最终编辑"""
    reviewed = state["reviewed_draft"]
    
    final = f"""# 最终版本

{state['original_content']}

## 正文

根据起草者的初稿和审核者的意见，最终版本如下：

1. 引言：[已优化]
2. 主体：[已添加示例]
3. 结论：[已完善过渡]

## 元信息
- 版本：1.0
- 状态：已发布
"""
    
    return {
        "final_content": final,
        "collaboration_log": state["collaboration_log"] + ["编辑者完成最终编辑"]
    }


def build_collaboration_graph():
    """构建协作图"""
    graph = StateGraph(CollaborationState)
    
    graph.add_node("drafter", drafter_agent)
    graph.add_node("reviewer", reviewer_agent)
    graph.add_node("editor", editor_agent)
    
    # 链式协作流程
    graph.add_edge(START, "drafter")
    graph.add_edge("drafter", "reviewer")
    graph.add_edge("reviewer", "editor")
    graph.add_edge("editor", END)
    
    return graph.compile()


def demo_collaboration():
    """演示协作模式"""
    print("\n" + "=" * 60)
    print("演示 2: 链式协作模式")
    print("=" * 60)
    
    app = build_collaboration_graph()
    
    print("\n协作任务: 创建技术文档")
    print("-" * 40)
    
    result = app.invoke({
        "original_content": "LangGraph 多智能体系统介绍",
        "draft": "",
        "reviewed_draft": "",
        "final_content": "",
        "collaboration_log": []
    })
    
    print("\n协作日志:")
    for log in result["collaboration_log"]:
        print(f"  ✓ {log}")


# =============================================================================
# 第三部分：子图模式（模块化）
# =============================================================================

class ResearchState(TypedDict):
    """研究子图状态"""
    topic: str
    sources: list[str]
    findings: str


class WritingState(TypedDict):
    """写作子图状态"""
    topic: str
    outline: str
    content: str


class MainState(TypedDict):
    """主图状态"""
    task: str
    research_result: str
    writing_result: str
    final_output: str


def build_research_subgraph():
    """构建研究子图"""
    
    def search_sources(state: ResearchState) -> dict:
        """搜索来源"""
        topic = state["topic"]
        sources = [
            f"来源1: 关于{topic}的学术论文",
            f"来源2: 关于{topic}的官方文档",
            f"来源3: 关于{topic}的技术博客"
        ]
        return {"sources": sources}
    
    def analyze_sources(state: ResearchState) -> dict:
        """分析来源"""
        findings = f"研究发现：关于'{state['topic']}'，我们从{len(state['sources'])}个来源收集了信息..."
        return {"findings": findings}
    
    graph = StateGraph(ResearchState)
    graph.add_node("search", search_sources)
    graph.add_node("analyze", analyze_sources)
    graph.add_edge(START, "search")
    graph.add_edge("search", "analyze")
    graph.add_edge("analyze", END)
    
    return graph.compile()


def build_writing_subgraph():
    """构建写作子图"""
    
    def create_outline(state: WritingState) -> dict:
        """创建大纲"""
        outline = f"""大纲: {state['topic']}
1. 引言
2. 背景
3. 主要内容
4. 结论"""
        return {"outline": outline}
    
    def write_content(state: WritingState) -> dict:
        """撰写内容"""
        content = f"基于大纲撰写的完整内容...\n{state['outline']}\n[内容已完成]"
        return {"content": content}
    
    graph = StateGraph(WritingState)
    graph.add_node("outline", create_outline)
    graph.add_node("write", write_content)
    graph.add_edge(START, "outline")
    graph.add_edge("outline", "write")
    graph.add_edge("write", END)
    
    return graph.compile()


# 创建子图实例
research_subgraph = build_research_subgraph()
writing_subgraph = build_writing_subgraph()


def research_node(state: MainState) -> dict:
    """调用研究子图"""
    result = research_subgraph.invoke({
        "topic": state["task"],
        "sources": [],
        "findings": ""
    })
    return {"research_result": result["findings"]}


def writing_node(state: MainState) -> dict:
    """调用写作子图"""
    result = writing_subgraph.invoke({
        "topic": state["task"],
        "outline": "",
        "content": ""
    })
    return {"writing_result": result["content"]}


def combine_results(state: MainState) -> dict:
    """合并结果"""
    final = f"""# 最终报告

## 研究部分
{state['research_result']}

## 写作部分
{state['writing_result']}
"""
    return {"final_output": final}


def build_main_graph_with_subgraphs():
    """构建使用子图的主图"""
    graph = StateGraph(MainState)
    
    graph.add_node("research", research_node)
    graph.add_node("writing", writing_node)
    graph.add_node("combine", combine_results)
    
    graph.add_edge(START, "research")
    graph.add_edge("research", "writing")
    graph.add_edge("writing", "combine")
    graph.add_edge("combine", END)
    
    return graph.compile()


def demo_subgraph():
    """演示子图模式"""
    print("\n" + "=" * 60)
    print("演示 3: 子图模式（模块化）")
    print("=" * 60)
    
    app = build_main_graph_with_subgraphs()
    
    print("\n任务: '人工智能发展历史'")
    print("-" * 40)
    
    for step_num, event in enumerate(app.stream({
        "task": "人工智能发展历史",
        "research_result": "",
        "writing_result": "",
        "final_output": ""
    })):
        for node, output in event.items():
            print(f"\n步骤 {step_num + 1} - 节点: {node}")
            if node == "research":
                print(f"  研究完成: {output.get('research_result', '')[:50]}...")
            elif node == "writing":
                print(f"  写作完成: {output.get('writing_result', '')[:50]}...")
            elif node == "combine":
                print(f"  合并完成: 最终报告已生成")


# =============================================================================
# 第四部分：动态团队（根据任务选择 Agent）
# =============================================================================

class DynamicTeamState(TypedDict):
    """动态团队状态"""
    task: str
    task_type: str
    selected_agents: list[str]
    results: dict
    final_answer: str


# Agent 能力定义
AGENT_CAPABILITIES = {
    "data_analyst": {
        "skills": ["数据分析", "统计", "可视化"],
        "handle": lambda t: f"[数据分析师] 对 '{t}' 进行了数据分析，发现关键指标..."
    },
    "ml_engineer": {
        "skills": ["机器学习", "模型", "训练"],
        "handle": lambda t: f"[ML工程师] 为 '{t}' 设计了机器学习方案..."
    },
    "backend_dev": {
        "skills": ["API", "后端", "数据库", "服务"],
        "handle": lambda t: f"[后端开发] 为 '{t}' 实现了后端服务..."
    },
    "frontend_dev": {
        "skills": ["UI", "前端", "界面", "交互"],
        "handle": lambda t: f"[前端开发] 为 '{t}' 创建了用户界面..."
    },
    "tech_writer": {
        "skills": ["文档", "说明", "教程"],
        "handle": lambda t: f"[技术作家] 为 '{t}' 撰写了技术文档..."
    }
}


def analyze_task_and_select_team(state: DynamicTeamState) -> dict:
    """分析任务并选择合适的团队成员"""
    task = state["task"].lower()
    
    selected = []
    for agent_name, agent_info in AGENT_CAPABILITIES.items():
        for skill in agent_info["skills"]:
            if skill in task:
                if agent_name not in selected:
                    selected.append(agent_name)
                break
    
    # 如果没有匹配，默认选择技术作家
    if not selected:
        selected = ["tech_writer"]
    
    return {"selected_agents": selected}


def execute_team(state: DynamicTeamState) -> dict:
    """执行选定团队的工作"""
    results = {}
    task = state["task"]
    
    for agent_name in state["selected_agents"]:
        if agent_name in AGENT_CAPABILITIES:
            handler = AGENT_CAPABILITIES[agent_name]["handle"]
            results[agent_name] = handler(task)
    
    return {"results": results}


def compile_team_output(state: DynamicTeamState) -> dict:
    """汇总团队输出"""
    results = state["results"]
    
    output = f"# 任务: {state['task']}\n\n"
    output += f"## 参与团队: {', '.join(state['selected_agents'])}\n\n"
    
    for agent, result in results.items():
        output += f"### {agent}\n{result}\n\n"
    
    return {"final_answer": output}


def build_dynamic_team_graph():
    """构建动态团队图"""
    graph = StateGraph(DynamicTeamState)
    
    graph.add_node("select_team", analyze_task_and_select_team)
    graph.add_node("execute", execute_team)
    graph.add_node("compile", compile_team_output)
    
    graph.add_edge(START, "select_team")
    graph.add_edge("select_team", "execute")
    graph.add_edge("execute", "compile")
    graph.add_edge("compile", END)
    
    return graph.compile()


def demo_dynamic_team():
    """演示动态团队"""
    print("\n" + "=" * 60)
    print("演示 4: 动态团队选择")
    print("=" * 60)
    
    app = build_dynamic_team_graph()
    
    tasks = [
        "创建一个数据分析和可视化的API服务",
        "编写机器学习模型训练教程"
    ]
    
    for task in tasks:
        print(f"\n任务: '{task}'")
        print("-" * 40)
        
        result = app.invoke({
            "task": task,
            "task_type": "",
            "selected_agents": [],
            "results": {},
            "final_answer": ""
        })
        
        print(f"选择的团队: {result['selected_agents']}")
        print(f"团队成员数: {len(result['selected_agents'])}")


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    demo_supervisor()
    demo_collaboration()
    demo_subgraph()
    demo_dynamic_team()
    
    print("\n" + "=" * 60)
    print("多智能体协作教程完成！")
    print("=" * 60)
    print("""
关键要点：
1. Supervisor 模式：中央协调者分配任务给专家
2. 协作模式：Agent 之间链式传递和协作
3. 子图模式：将复杂功能封装为可复用模块
4. 动态团队：根据任务需求动态选择 Agent

多 Agent 系统设计原则：
- 单一职责：每个 Agent 专注一个领域
- 清晰接口：明确定义 Agent 之间的通信格式
- 状态共享：通过共享状态传递信息
- 可扩展性：易于添加新的 Agent

应用场景：
- 复杂任务分解与并行处理
- 专业领域知识整合
- 多角度问题分析
- 工作流自动化

恭喜！你已完成 LangGraph 基础教程的全部内容！
""")
