"""Run a deterministic model -> tools -> model loop without network access.

Run:
    conda run -n langgraph python langgraph/07-tools-and-agents/examples/deterministic_tool_loop.py
    conda run -n langgraph python langgraph/07-tools-and-agents/examples/deterministic_tool_loop.py --self-test
"""

from __future__ import annotations

import argparse
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


class AgentState(MessagesState):
    """Messages drive the loop; numeric fields make the request deterministic."""

    left: int
    right: int
    increment: int


@tool
def multiply(left: int, right: int) -> int:
    """Multiply two integers."""

    return left * right


@tool
def add(left: int, right: int) -> int:
    """Add two integers."""

    return left + right


TOOLS = [multiply, add]


def deterministic_model(state: AgentState) -> dict[str, list[AIMessage]]:
    """Emit tool calls in a fixed sequence, then produce a final answer."""

    tool_messages = [
        message for message in state["messages"] if isinstance(message, ToolMessage)
    ]

    if not tool_messages:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "multiply",
                            "args": {
                                "left": state["left"],
                                "right": state["right"],
                            },
                            "id": "call-multiply",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    latest_tool = tool_messages[-1]
    if latest_tool.name == "multiply":
        product = int(str(latest_tool.content))
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {
                                "left": product,
                                "right": state["increment"],
                            },
                            "id": "call-add",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    total = int(str(latest_tool.content))
    return {"messages": [AIMessage(content=f"Final result: {total}")]}


def build_graph():
    """Compile the canonical model -> tools -> model agent loop."""

    builder = StateGraph(AgentState)
    builder.add_node("model", deterministic_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")
    return builder.compile()


def run(left: int, right: int, increment: int) -> AgentState:
    return build_graph().invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (f"Multiply {left} by {right}, then add {increment}."),
                }
            ],
            "left": left,
            "right": right,
            "increment": increment,
        }
    )


def self_test() -> None:
    result = run(7, 6, 8)
    messages = result["messages"]

    assert [message.type for message in messages] == [
        "human",
        "ai",
        "tool",
        "ai",
        "tool",
        "ai",
    ]
    assert messages[1].tool_calls[0]["name"] == "multiply"
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].name == "multiply"
    assert messages[2].content == "42"
    assert messages[3].tool_calls[0]["name"] == "add"
    assert messages[4].content == "50"
    assert messages[-1].content == "Final result: 50"

    updates = list(
        build_graph().stream(
            {
                "messages": [{"role": "user", "content": "Calculate."}],
                "left": 2,
                "right": 5,
                "increment": 1,
            },
            stream_mode="updates",
        )
    )
    assert [next(iter(update)) for update in updates] == [
        "model",
        "tools",
        "model",
        "tools",
        "model",
    ]
    assert END not in updates
    print("deterministic_tool_loop self-test passed")


def describe_message(message: Any) -> str:
    if isinstance(message, AIMessage) and message.tool_calls:
        calls = [f"{call['name']}({call['args']})" for call in message.tool_calls]
        return f"ai -> tool calls: {', '.join(calls)}"
    if isinstance(message, ToolMessage):
        return f"tool {message.name}: {message.content}"
    return f"{message.type}: {message.content}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=int, default=7)
    parser.add_argument("--right", type=int, default=6)
    parser.add_argument("--increment", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    result = run(args.left, args.right, args.increment)
    for index, message in enumerate(result["messages"], start=1):
        print(f"{index}. {describe_message(message)}")


if __name__ == "__main__":
    main()
