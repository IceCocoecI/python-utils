"""Share explicit user memory across threads while keeping users isolated.

Run:
    conda run -n langgraph python langgraph/06-long-term-memory/examples/cross_thread_memory.py
    conda run -n langgraph python langgraph/06-long-term-memory/examples/cross_thread_memory.py --self-test
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


@dataclass(frozen=True)
class UserContext:
    """Trusted identity used to derive the Store namespace."""

    user_id: str


class RememberRequest(TypedDict):
    operation: Literal["remember"]
    key: str
    value: str


class RecallRequest(TypedDict):
    operation: Literal["recall"]
    key: str


PreferenceRequest = RememberRequest | RecallRequest


class PreferenceState(TypedDict, total=False):
    request: PreferenceRequest
    result: str


def preference_node(
    state: PreferenceState,
    runtime: Runtime[UserContext],
) -> dict[str, str]:
    store = runtime.store
    if store is None:
        raise RuntimeError("this graph requires a Store")

    request = state["request"]
    namespace = ("users", runtime.context.user_id, "preferences")
    key = request["key"]

    if request["operation"] == "remember":
        value = request.get("value")
        if not value:
            raise ValueError("remember requires a non-empty value")
        store.put(namespace, key, {"value": value})
        return {"result": f"saved {key}={value}"}

    item = store.get(namespace, key)
    if item is None:
        return {"result": f"no memory for {key}"}
    return {"result": f"remembered {key}={item.value['value']}"}


def build_graph(
    store: InMemoryStore | None = None,
    checkpointer: InMemorySaver | None = None,
):
    memory_store = store if store is not None else InMemoryStore()
    saver = checkpointer if checkpointer is not None else InMemorySaver()

    builder = StateGraph(PreferenceState, context_schema=UserContext)
    builder.add_node("preference", preference_node)
    builder.add_edge(START, "preference")
    builder.add_edge("preference", END)
    return builder.compile(checkpointer=saver, store=memory_store), memory_store


def thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def self_test() -> None:
    graph, store = build_graph()
    alice = UserContext(user_id="alice")
    bob = UserContext(user_id="bob")

    saved = graph.invoke(
        {
            "request": {
                "operation": "remember",
                "key": "theme",
                "value": "dark",
            }
        },
        thread_config("alice-thread-a"),
        context=alice,
    )
    recalled = graph.invoke(
        {"request": {"operation": "recall", "key": "theme"}},
        thread_config("alice-thread-b"),
        context=alice,
    )
    isolated = graph.invoke(
        {"request": {"operation": "recall", "key": "theme"}},
        thread_config("bob-thread-a"),
        context=bob,
    )

    assert saved["result"] == "saved theme=dark"
    assert recalled["result"] == "remembered theme=dark"
    assert isolated["result"] == "no memory for theme"

    # Checkpoints remain thread-scoped even though the Store is user-scoped.
    alice_a = graph.get_state(thread_config("alice-thread-a")).values
    alice_b = graph.get_state(thread_config("alice-thread-b")).values
    assert alice_a["request"]["operation"] == "remember"
    assert alice_b["request"]["operation"] == "recall"
    assert alice_a != alice_b

    # A whole request envelope replaces the previous request in the same thread.
    # The missing value must not be inherited from the earlier checkpoint.
    merge_config = thread_config("alice-merge-regression")
    graph.invoke(
        {
            "request": {
                "operation": "remember",
                "key": "theme",
                "value": "dark",
            }
        },
        merge_config,
        context=alice,
    )
    try:
        graph.invoke(
            {"request": {"operation": "remember", "key": "language"}},
            merge_config,
            context=alice,
        )
    except ValueError as exc:
        assert str(exc) == "remember requires a non-empty value"
    else:
        raise AssertionError("a missing value must not reuse checkpointed input")
    assert store.get(("users", "alice", "preferences"), "language") is None

    item = store.get(("users", "alice", "preferences"), "theme")
    assert item is not None and item.value == {"value": "dark"}
    assert store.get(("users", "bob", "preferences"), "theme") is None
    print("cross_thread_memory self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    graph, _ = build_graph()
    alice = UserContext(user_id="alice")
    bob = UserContext(user_id="bob")

    print("Alice saves from thread A:")
    print(
        graph.invoke(
            {
                "request": {
                    "operation": "remember",
                    "key": "theme",
                    "value": "dark",
                }
            },
            thread_config("alice-thread-a"),
            context=alice,
        )["result"]
    )

    print("Alice recalls from a different thread B:")
    print(
        graph.invoke(
            {"request": {"operation": "recall", "key": "theme"}},
            thread_config("alice-thread-b"),
            context=alice,
        )["result"]
    )

    print("Bob uses another namespace:")
    print(
        graph.invoke(
            {"request": {"operation": "recall", "key": "theme"}},
            thread_config("bob-thread-a"),
            context=bob,
        )["result"]
    )


if __name__ == "__main__":
    main()
