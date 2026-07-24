"""Build and run a first offline LangGraph StateGraph.

Run:
    conda run -n langgraph python langgraph/01-foundations/examples/hello_state_graph.py
    conda run -n langgraph python langgraph/01-foundations/examples/hello_state_graph.py --self-test
"""

from __future__ import annotations

import argparse
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class GreetingState(TypedDict):
    """Shared state for the complete graph."""

    name: str
    normalized_name: str
    greeting: str
    greeting_length: int


def normalize_name(state: GreetingState) -> dict[str, str]:
    """Return only the state field changed by this node."""

    normalized = " ".join(state["name"].strip().split())
    if not normalized:
        raise ValueError("name must not be empty")
    return {"normalized_name": normalized}


def build_greeting(state: GreetingState) -> dict[str, str]:
    """Read a previous update and produce the next update."""

    return {"greeting": f"Hello, {state['normalized_name']}!"}


def measure_greeting(state: GreetingState) -> dict[str, int]:
    """Compute one final field without rebuilding the whole state."""

    return {"greeting_length": len(state["greeting"])}


def build_graph():
    """Compile START -> normalize -> greet -> measure -> END."""

    builder = StateGraph(GreetingState)
    builder.add_node("normalize", normalize_name)
    builder.add_node("greet", build_greeting)
    builder.add_node("measure", measure_greeting)

    builder.add_edge(START, "normalize")
    builder.add_edge("normalize", "greet")
    builder.add_edge("greet", "measure")
    builder.add_edge("measure", END)
    return builder.compile()


def run(name: str) -> GreetingState:
    """Invoke the graph with the only field needed at entry time."""

    graph = build_graph()
    return graph.invoke({"name": name})


def self_test() -> None:
    result = run("  Ada   Lovelace  ")
    assert result == {
        "name": "  Ada   Lovelace  ",
        "normalized_name": "Ada Lovelace",
        "greeting": "Hello, Ada Lovelace!",
        "greeting_length": 20,
    }

    try:
        run("   ")
    except ValueError as exc:
        assert str(exc) == "name must not be empty"
    else:
        raise AssertionError("an empty name must fail")

    print("hello_state_graph self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="Ada Lovelace")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    result = run(args.name)
    print(f"input name: {result['name']!r}")
    print(f"normalized name: {result['normalized_name']!r}")
    print(f"greeting: {result['greeting']}")
    print(f"greeting length: {result['greeting_length']}")


if __name__ == "__main__":
    main()
