"""Hand off from a specialist subgraph to a destination in its parent graph.

Run:
    conda run -n langgraph python langgraph/10-subgraphs-and-multi-agent/examples/handoff_parent_command.py
    conda run -n langgraph python langgraph/10-subgraphs-and-multi-agent/examples/handoff_parent_command.py --show-events
    conda run -n langgraph python langgraph/10-subgraphs-and-multi-agent/examples/handoff_parent_command.py --self-test
"""

from __future__ import annotations

import argparse
from pprint import pprint
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command


class SupportState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: str
    specialist_note: str
    handled_by: str


class BillingState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    specialist_note: str


def inspect_billing_request(state: BillingState) -> dict[str, str]:
    request = str(state["messages"][-1].content)
    return {"specialist_note": f"verified billing request: {request}"}


def handoff_to_parent(state: BillingState) -> Command:
    """Update parent channels and jump to a node that exists in the parent graph."""

    return Command(
        graph=Command.PARENT,
        goto="compose_billing_reply",
        update={
            "specialist_note": state["specialist_note"],
            "messages": [
                AIMessage(
                    content="Billing specialist finished analysis.",
                    id="billing-specialist",
                )
            ],
        },
    )


def build_billing_subgraph():
    builder = StateGraph(BillingState)
    builder.add_node("inspect_billing_request", inspect_billing_request)
    builder.add_node("handoff_to_parent", handoff_to_parent)
    builder.add_edge(START, "inspect_billing_request")
    builder.add_edge("inspect_billing_request", "handoff_to_parent")
    builder.add_edge("handoff_to_parent", END)
    return builder.compile()


def classify_request(state: SupportState) -> dict[str, str]:
    request = str(state["messages"][-1].content).casefold()
    route = (
        "billing" if any(word in request for word in ("bill", "charge")) else "general"
    )
    return {"route": route}


def route_request(state: SupportState) -> str:
    return state["route"]


def compose_billing_reply(state: SupportState) -> dict[str, object]:
    return {
        "handled_by": "parent:compose_billing_reply",
        "messages": [
            AIMessage(
                content=f"Parent received: {state['specialist_note']}",
                id="parent-final",
            )
        ],
    }


def compose_general_reply(state: SupportState) -> dict[str, object]:
    return {
        "handled_by": "parent:compose_general_reply",
        "messages": [
            AIMessage(
                content="A general support specialist will help.",
                id="parent-general",
            )
        ],
    }


def build_graph():
    builder = StateGraph(SupportState)
    builder.add_node("classify_request", classify_request)
    builder.add_node("billing_specialist", build_billing_subgraph())
    builder.add_node("compose_billing_reply", compose_billing_reply)
    builder.add_node("compose_general_reply", compose_general_reply)
    builder.add_edge(START, "classify_request")
    builder.add_conditional_edges(
        "classify_request",
        route_request,
        {
            "billing": "billing_specialist",
            "general": "compose_general_reply",
        },
    )
    builder.add_edge("compose_billing_reply", END)
    builder.add_edge("compose_general_reply", END)
    return builder.compile()


def initial_state(request: str) -> SupportState:
    return {
        "messages": [HumanMessage(content=request, id="user-request")],
        "route": "",
        "specialist_note": "",
        "handled_by": "",
    }


def run(request: str) -> SupportState:
    return build_graph().invoke(initial_state(request))


def collect_updates(request: str) -> list[tuple[tuple[str, ...], Any]]:
    return list(
        build_graph().stream(
            initial_state(request),
            stream_mode="updates",
            subgraphs=True,
        )
    )


def self_test() -> None:
    request = "Why was I charged twice?"
    result = run(request)
    assert result["route"] == "billing"
    assert result["handled_by"] == "parent:compose_billing_reply"
    assert result["specialist_note"] == f"verified billing request: {request}"
    assert [message.id for message in result["messages"]] == [
        "user-request",
        "billing-specialist",
        "parent-final",
    ]

    events = collect_updates(request)
    nested_nodes = {
        node_name for namespace, update in events if namespace for node_name in update
    }
    parent_nodes = {
        node_name
        for namespace, update in events
        if not namespace
        for node_name in update
    }
    assert "inspect_billing_request" in nested_nodes
    assert {
        "classify_request",
        "billing_specialist",
        "compose_billing_reply",
    } <= parent_nodes
    assert "compose_general_reply" not in parent_nodes

    general = run("How do I reset my password?")
    assert general["route"] == "general"
    assert general["handled_by"] == "parent:compose_general_reply"
    assert [message.id for message in general["messages"]] == [
        "user-request",
        "parent-general",
    ]
    print("handoff_parent_command self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", default="Why was I charged twice?")
    parser.add_argument("--show-events", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    result = run(args.request)
    print(f"route: {result['route']}")
    print(f"handled by: {result['handled_by']}")
    for message in result["messages"]:
        print(f"{message.type}: {message.content}")

    if args.show_events:
        print("\nsubgraph-aware updates:")
        for namespace, update in collect_updates(args.request):
            pprint({"namespace": namespace, "update": update})


if __name__ == "__main__":
    main()
