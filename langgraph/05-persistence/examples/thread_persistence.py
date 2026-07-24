"""Persist graph State by thread_id with InMemorySaver.

Run:
    conda run -n langgraph python langgraph/05-persistence/examples/thread_persistence.py
    conda run -n langgraph python langgraph/05-persistence/examples/thread_persistence.py --self-test
"""

from __future__ import annotations

import argparse
from operator import add
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class ConversationState(TypedDict, total=False):
    """State may arrive incrementally over several calls in one thread."""

    messages: Annotated[list[str], add]
    turn: int
    last_response: str


def respond(state: ConversationState) -> dict[str, object]:
    messages = state.get("messages", [])
    if not messages or not messages[-1].startswith("user: "):
        raise ValueError("the latest message must start with 'user: '")

    turn = state.get("turn", 0) + 1
    user_text = messages[-1].removeprefix("user: ")
    response = f"assistant: turn={turn}; echo={user_text}"
    return {
        "messages": [response],
        "turn": turn,
        "last_response": response,
    }


def build_graph(checkpointer: InMemorySaver | None = None):
    saver = checkpointer if checkpointer is not None else InMemorySaver()
    builder = StateGraph(ConversationState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=saver)


def thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def send(graph, thread_id: str, text: str) -> ConversationState:
    return graph.invoke(
        {"messages": [f"user: {text}"]},
        thread_config(thread_id),
    )


def self_test() -> None:
    graph = build_graph()

    first = send(graph, "thread-alice", "hello")
    second = send(graph, "thread-alice", "remember this")
    isolated = send(graph, "thread-bob", "fresh start")

    assert first["turn"] == 1
    assert first["messages"] == [
        "user: hello",
        "assistant: turn=1; echo=hello",
    ]

    assert second["turn"] == 2
    assert second["messages"] == [
        "user: hello",
        "assistant: turn=1; echo=hello",
        "user: remember this",
        "assistant: turn=2; echo=remember this",
    ]

    assert isolated["turn"] == 1
    assert isolated["messages"] == [
        "user: fresh start",
        "assistant: turn=1; echo=fresh start",
    ]

    snapshot = graph.get_state(thread_config("thread-alice"))
    assert snapshot.values == second
    assert snapshot.next == ()
    assert snapshot.config["configurable"]["thread_id"] == "thread-alice"
    print("thread_persistence self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-id", default="demo-thread")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    graph = build_graph()
    config = thread_config(args.thread_id)

    print("first invocation:")
    first = graph.invoke({"messages": ["user: my name is Ada"]}, config)
    print(first)

    print("\nsecond invocation with the same thread_id:")
    second = graph.invoke({"messages": ["user: what turn is this?"]}, config)
    print(second)

    print("\nlatest checkpoint:")
    snapshot = graph.get_state(config)
    print(f"values={snapshot.values}")
    print(f"next={snapshot.next}")
    print(f"checkpoint_id={snapshot.config['configurable']['checkpoint_id']}")


if __name__ == "__main__":
    main()
