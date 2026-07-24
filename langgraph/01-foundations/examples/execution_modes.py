"""Compare invoke, update streaming, and graph inspection offline.

Run:
    conda run -n langgraph python langgraph/01-foundations/examples/execution_modes.py
    conda run -n langgraph python langgraph/01-foundations/examples/execution_modes.py --self-test
"""

from __future__ import annotations

import argparse
from pprint import pprint
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class PipelineState(TypedDict):
    text: str
    normalized: str
    tokens: list[str]
    summary: str


def normalize_text(state: PipelineState) -> dict[str, str]:
    normalized = " ".join(state["text"].strip().casefold().split())
    return {"normalized": normalized}


def split_tokens(state: PipelineState) -> dict[str, list[str]]:
    return {"tokens": state["normalized"].split()}


def summarize(state: PipelineState) -> dict[str, str]:
    tokens = state["tokens"]
    first = tokens[0] if tokens else "<none>"
    last = tokens[-1] if tokens else "<none>"
    return {
        "summary": f"count={len(tokens)}; first={first}; last={last}",
    }


def build_graph():
    builder = StateGraph(PipelineState)
    builder.add_node("normalize", normalize_text)
    builder.add_node("tokenize", split_tokens)
    builder.add_node("summarize", summarize)
    builder.add_edge(START, "normalize")
    builder.add_edge("normalize", "tokenize")
    builder.add_edge("tokenize", "summarize")
    builder.add_edge("summarize", END)
    return builder.compile()


def run_invoke(text: str) -> PipelineState:
    return build_graph().invoke({"text": text})


def collect_updates(text: str) -> list[dict[str, Any]]:
    graph = build_graph()
    return list(graph.stream({"text": text}, stream_mode="updates"))


def self_test() -> None:
    text = "  LangGraph   Keeps State Visible  "
    result = run_invoke(text)
    assert result["normalized"] == "langgraph keeps state visible"
    assert result["tokens"] == ["langgraph", "keeps", "state", "visible"]
    assert result["summary"] == "count=4; first=langgraph; last=visible"

    updates = collect_updates(text)
    assert [next(iter(update)) for update in updates] == [
        "normalize",
        "tokenize",
        "summarize",
    ]
    assert updates[0] == {
        "normalize": {"normalized": "langgraph keeps state visible"},
    }
    assert updates[-1] == {
        "summarize": {"summary": "count=4; first=langgraph; last=visible"},
    }

    mermaid = build_graph().get_graph().draw_mermaid()
    assert "normalize" in mermaid
    assert "tokenize" in mermaid
    assert "summarize" in mermaid
    print("execution_modes self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="LangGraph keeps state visible")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    graph = build_graph()
    print("invoke result:")
    pprint(graph.invoke({"text": args.text}))

    print("\nstreamed updates:")
    for update in graph.stream({"text": args.text}, stream_mode="updates"):
        pprint(update)

    print("\nmermaid definition:")
    print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
