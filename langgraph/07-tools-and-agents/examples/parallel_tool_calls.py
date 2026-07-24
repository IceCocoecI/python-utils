"""Execute multiple tool calls emitted by one deterministic model message.

Run:
    conda run -n langgraph python langgraph/07-tools-and-agents/examples/parallel_tool_calls.py
    conda run -n langgraph python langgraph/07-tools-and-agents/examples/parallel_tool_calls.py --self-test
"""

from __future__ import annotations

import argparse

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


class AnalysisState(MessagesState):
    text: str


@tool
def word_count(text: str) -> int:
    """Count whitespace-separated words."""

    return len(text.split())


@tool
def unique_words(text: str) -> str:
    """Return sorted unique case-insensitive words as comma-separated text."""

    words = sorted({word.casefold() for word in text.split()})
    return ",".join(words)


def deterministic_model(state: AnalysisState) -> dict[str, list[AIMessage]]:
    tool_messages = [
        message for message in state["messages"] if isinstance(message, ToolMessage)
    ]

    if not tool_messages:
        text = state["text"]
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "word_count",
                            "args": {"text": text},
                            "id": "call-count",
                            "type": "tool_call",
                        },
                        {
                            "name": "unique_words",
                            "args": {"text": text},
                            "id": "call-unique",
                            "type": "tool_call",
                        },
                    ],
                )
            ]
        }

    results = {message.name: str(message.content) for message in tool_messages}
    summary = f"words={results['word_count']}; unique={results['unique_words']}"
    return {"messages": [AIMessage(content=summary)]}


def build_graph():
    builder = StateGraph(AnalysisState)
    builder.add_node("model", deterministic_model)
    builder.add_node("tools", ToolNode([word_count, unique_words]))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")
    return builder.compile()


def run(text: str) -> AnalysisState:
    return build_graph().invoke(
        {
            "messages": [{"role": "user", "content": f"Analyze: {text}"}],
            "text": text,
        }
    )


def self_test() -> None:
    result = run("Graph state graph tools")
    messages = result["messages"]

    assert len(messages) == 5
    request = messages[1]
    assert isinstance(request, AIMessage)
    assert [call["name"] for call in request.tool_calls] == [
        "word_count",
        "unique_words",
    ]

    tool_messages = [
        message for message in messages if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 2
    by_name = {message.name: message.content for message in tool_messages}
    assert by_name == {
        "word_count": "4",
        "unique_words": "graph,state,tools",
    }
    assert messages[-1].content == "words=4; unique=graph,state,tools"
    print("parallel_tool_calls self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="Graph state graph tools")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    result = run(args.text)
    for message in result["messages"]:
        if isinstance(message, AIMessage) and message.tool_calls:
            print(f"model requested {len(message.tool_calls)} tools")
        elif isinstance(message, ToolMessage):
            print(f"tool {message.name}: {message.content}")
        elif message.type == "ai":
            print(f"final: {message.content}")


if __name__ == "__main__":
    main()
