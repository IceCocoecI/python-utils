"""Compare overwrite, list accumulation, and add_messages behavior.

Run:
    conda run -n langgraph python langgraph/02-state-and-reducers/examples/reducers_and_messages.py
    conda run -n langgraph python langgraph/02-state-and-reducers/examples/reducers_and_messages.py --self-test
"""

from __future__ import annotations

import argparse
from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class WorkflowState(TypedDict):
    topic: str
    latest: str
    steps: Annotated[list[str], add]
    messages: Annotated[list[AnyMessage], add_messages]


def draft(state: WorkflowState) -> dict[str, object]:
    content = f"draft about {state['topic']}"
    return {
        "latest": content,
        "steps": ["draft"],
        "messages": [AIMessage(content=content, id="answer")],
    }


def review(state: WorkflowState) -> dict[str, object]:
    content = f"reviewed: {state['latest']}"
    return {
        "latest": content,
        "steps": ["review"],
        "messages": [AIMessage(content=content, id="answer")],
    }


def publish(state: WorkflowState) -> dict[str, object]:
    content = f"published: {state['latest']}"
    return {
        "latest": content,
        "steps": ["publish"],
        "messages": [AIMessage(content=content, id="answer")],
    }


def build_graph():
    builder = StateGraph(WorkflowState)
    builder.add_node("draft", draft)
    builder.add_node("review", review)
    builder.add_node("publish", publish)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "review")
    builder.add_edge("review", "publish")
    builder.add_edge("publish", END)
    return builder.compile()


def run(topic: str) -> WorkflowState:
    initial_message = HumanMessage(content=f"Write about {topic}", id="request")
    return build_graph().invoke(
        {
            "topic": topic,
            "latest": "",
            "steps": [],
            "messages": [initial_message],
        }
    )


def self_test() -> None:
    result = run("reducers")

    assert result["latest"] == ("published: reviewed: draft about reducers")
    assert result["steps"] == ["draft", "review", "publish"]

    assert len(result["messages"]) == 2
    request, answer = result["messages"]
    assert isinstance(request, HumanMessage)
    assert isinstance(answer, AIMessage)
    assert request.id == "request"
    assert answer.id == "answer"
    assert answer.content == "published: reviewed: draft about reducers"

    print("reducers_and_messages self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="reducers")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    result = run(args.topic)
    print(f"latest: {result['latest']}")
    print(f"steps: {result['steps']}")
    print("messages:")
    for message in result["messages"]:
        print(f"  type={message.type} id={message.id} content={message.content!r}")


if __name__ == "__main__":
    main()
