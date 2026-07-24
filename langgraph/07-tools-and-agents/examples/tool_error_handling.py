"""Demonstrate ToolNode's default and explicit error-handling boundaries.

Run:
    conda run -n langgraph python langgraph/07-tools-and-agents/examples/tool_error_handling.py
    conda run -n langgraph python langgraph/07-tools-and-agents/examples/tool_error_handling.py --self-test
"""

from __future__ import annotations

import argparse
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


@tool
def guarded_divide(numerator: int, denominator: int) -> float:
    """Divide two integers, rejecting a zero denominator."""

    if denominator == 0:
        raise ValueError("denominator must not be zero")
    return numerator / denominator


def tool_input(args: dict[str, Any], call_id: str) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "guarded_divide",
                        "args": args,
                        "id": call_id,
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }


def build_graph(*, handle_tool_errors: bool | None = None):
    builder = StateGraph(MessagesState)
    if handle_tool_errors is None:
        tool_node = ToolNode([guarded_divide])
    else:
        tool_node = ToolNode(
            [guarded_divide],
            handle_tool_errors=handle_tool_errors,
        )
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    return builder.compile()


def only_tool_message(result: dict[str, Any]) -> ToolMessage:
    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 1
    return tool_messages[0]


def demonstrate() -> tuple[ToolMessage, ValueError, ToolMessage]:
    default_graph = build_graph()

    # Invalid tool-call arguments are invocation errors. The default handler
    # returns them to the model as an error ToolMessage.
    schema_error = only_tool_message(
        default_graph.invoke(tool_input({"numerator": 10}, "call-missing-denominator"))
    )

    # An exception raised inside a successfully invoked tool propagates by default.
    try:
        default_graph.invoke(
            tool_input(
                {"numerator": 10, "denominator": 0},
                "call-default-execution-error",
            )
        )
    except ValueError as exc:
        execution_error = exc
    else:
        raise AssertionError("the default ToolNode must propagate tool body errors")

    # Explicitly opting in catches tool body errors and returns an error message.
    handled_graph = build_graph(handle_tool_errors=True)
    handled_error = only_tool_message(
        handled_graph.invoke(
            tool_input(
                {"numerator": 10, "denominator": 0},
                "call-handled-execution-error",
            )
        )
    )
    return schema_error, execution_error, handled_error


def self_test() -> None:
    schema_error, execution_error, handled_error = demonstrate()

    assert schema_error.status == "error"
    assert schema_error.tool_call_id == "call-missing-denominator"
    assert "denominator" in str(schema_error.content)

    assert str(execution_error) == "denominator must not be zero"

    assert handled_error.status == "error"
    assert handled_error.tool_call_id == "call-handled-execution-error"
    assert "denominator must not be zero" in str(handled_error.content)
    print("tool_error_handling self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    schema_error, execution_error, handled_error = demonstrate()
    print(f"default invocation error -> ToolMessage: {schema_error.content}")
    print(f"default tool body error -> raised: {execution_error}")
    print(f"handle_tool_errors=True -> ToolMessage: {handled_error.content}")


if __name__ == "__main__":
    main()
