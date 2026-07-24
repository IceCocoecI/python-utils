"""Stream node updates together with application-defined progress events.

Run:
    conda run -n langgraph python langgraph/09-streaming-and-resilience/examples/streaming_modes.py
    conda run -n langgraph python langgraph/09-streaming-and-resilience/examples/streaming_modes.py --self-test
"""

from __future__ import annotations

import argparse
from pprint import pprint
from typing import Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph


class PipelineState(TypedDict):
    text: str
    normalized: str
    words: list[str]
    summary: str


def normalize(state: PipelineState) -> dict[str, str]:
    writer = get_stream_writer()
    writer({"event": "progress", "step": "normalize", "percent": 25})
    normalized = " ".join(state["text"].casefold().split())
    return {"normalized": normalized}


def tokenize(state: PipelineState) -> dict[str, list[str]]:
    writer = get_stream_writer()
    writer({"event": "progress", "step": "tokenize", "percent": 60})
    return {"words": state["normalized"].split()}


def summarize(state: PipelineState) -> dict[str, str]:
    writer = get_stream_writer()
    writer({"event": "progress", "step": "summarize", "percent": 100})
    unique = len(set(state["words"]))
    return {
        "summary": f"words={len(state['words'])}; unique={unique}",
    }


def build_graph():
    builder = StateGraph(PipelineState)
    builder.add_node("normalize", normalize)
    builder.add_node("tokenize", tokenize)
    builder.add_node("summarize", summarize)
    builder.add_edge(START, "normalize")
    builder.add_edge("normalize", "tokenize")
    builder.add_edge("tokenize", "summarize")
    builder.add_edge("summarize", END)
    return builder.compile()


def collect_events(text: str) -> list[tuple[str, Any]]:
    graph = build_graph()
    return list(
        graph.stream(
            {
                "text": text,
                "normalized": "",
                "words": [],
                "summary": "",
            },
            stream_mode=["updates", "custom"],
        )
    )


def self_test() -> None:
    events = collect_events("  State Streams State  ")
    modes = [mode for mode, _ in events]
    assert modes == [
        "custom",
        "updates",
        "custom",
        "updates",
        "custom",
        "updates",
    ]

    custom = [chunk for mode, chunk in events if mode == "custom"]
    assert [event["percent"] for event in custom] == [25, 60, 100]
    assert [event["step"] for event in custom] == [
        "normalize",
        "tokenize",
        "summarize",
    ]

    updates = [chunk for mode, chunk in events if mode == "updates"]
    assert updates[0] == {"normalize": {"normalized": "state streams state"}}
    assert updates[1] == {"tokenize": {"words": ["state", "streams", "state"]}}
    assert updates[2] == {"summarize": {"summary": "words=3; unique=2"}}
    print("streaming_modes self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="State streams state")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    for mode, chunk in collect_events(args.text):
        print(f"[{mode}]")
        pprint(chunk)


if __name__ == "__main__":
    main()
