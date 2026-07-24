"""Separate public input, private working state, and public output.

Run:
    conda run -n langgraph python langgraph/02-state-and-reducers/examples/input_output_schema.py
    conda run -n langgraph python langgraph/02-state-and-reducers/examples/input_output_schema.py --self-test
"""

from __future__ import annotations

import argparse
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class InputState(TypedDict):
    raw_text: str


class OutputState(TypedDict):
    result: str
    word_count: int


class OverallState(InputState, OutputState):
    normalized: str


def normalize(state: InputState) -> dict[str, str]:
    normalized = " ".join(state["raw_text"].strip().casefold().split())
    return {"normalized": normalized}


def count_words(state: OverallState) -> dict[str, int]:
    return {"word_count": len(state["normalized"].split())}


def render_result(state: OverallState) -> dict[str, str]:
    rendered = state["normalized"].upper() or "<EMPTY>"
    return {"result": rendered}


def build_graph():
    builder = StateGraph(
        OverallState,
        input_schema=InputState,
        output_schema=OutputState,
    )
    builder.add_node("normalize", normalize)
    builder.add_node("count", count_words)
    builder.add_node("render", render_result)
    builder.add_edge(START, "normalize")
    builder.add_edge("normalize", "count")
    builder.add_edge("count", "render")
    builder.add_edge("render", END)
    return builder.compile()


def run(raw_text: str) -> OutputState:
    return build_graph().invoke({"raw_text": raw_text})


def self_test() -> None:
    result = run("  LangGraph   makes contracts explicit  ")
    assert result == {
        "result": "LANGGRAPH MAKES CONTRACTS EXPLICIT",
        "word_count": 4,
    }
    assert "raw_text" not in result
    assert "normalized" not in result

    empty = run("   ")
    assert empty == {"result": "<EMPTY>", "word_count": 0}
    print("input_output_schema self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="LangGraph makes contracts explicit")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    result = run(args.text)
    print(f"public output: {result}")
    print(f"public keys: {sorted(result)}")


if __name__ == "__main__":
    main()
