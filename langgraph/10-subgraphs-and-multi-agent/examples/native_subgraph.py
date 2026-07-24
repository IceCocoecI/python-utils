"""Add a compiled StateGraph directly as a parent-graph node.

Run:
    conda run -n langgraph python langgraph/10-subgraphs-and-multi-agent/examples/native_subgraph.py
    conda run -n langgraph python langgraph/10-subgraphs-and-multi-agent/examples/native_subgraph.py --compare-wrapper
    conda run -n langgraph python langgraph/10-subgraphs-and-multi-agent/examples/native_subgraph.py --self-test
"""

from __future__ import annotations

import argparse
from pprint import pprint
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class ResearchState(TypedDict):
    topic: str
    sources: list[str]
    notes: str


class ReportState(TypedDict):
    topic: str
    sources: list[str]
    notes: str
    report: str


def find_sources(state: ResearchState) -> dict[str, list[str]]:
    topic = state["topic"]
    return {
        "sources": [
            f"official guide for {topic}",
            f"design notes for {topic}",
        ]
    }


def extract_notes(state: ResearchState) -> dict[str, str]:
    joined = " | ".join(state["sources"])
    return {"notes": f"verified sources: {joined}"}


def build_research_subgraph():
    builder = StateGraph(ResearchState)
    builder.add_node("find_sources", find_sources)
    builder.add_node("extract_notes", extract_notes)
    builder.add_edge(START, "find_sources")
    builder.add_edge("find_sources", "extract_notes")
    builder.add_edge("extract_notes", END)
    return builder.compile()


def render_report(state: ReportState) -> dict[str, str]:
    return {
        "report": f"Report: {state['topic']}\n{state['notes']}",
    }


def _build_parent(research_node):
    builder = StateGraph(ReportState)
    builder.add_node("research", research_node)
    builder.add_node("render", render_report)
    builder.add_edge(START, "research")
    builder.add_edge("research", "render")
    builder.add_edge("render", END)
    return builder.compile()


def build_graph():
    """Register the compiled graph directly; shared keys are mapped automatically."""

    return _build_parent(build_research_subgraph())


def build_wrapped_graph():
    """Invoke the child in a node and map its input/output explicitly."""

    research = build_research_subgraph()

    def run_research(state: ReportState) -> dict[str, object]:
        child_input: ResearchState = {
            "topic": state["topic"],
            "sources": [],
            "notes": "",
        }
        child_result = research.invoke(child_input)
        return {
            "sources": child_result["sources"],
            "notes": child_result["notes"],
        }

    return _build_parent(run_research)


def initial_state(topic: str) -> ReportState:
    return {
        "topic": topic,
        "sources": [],
        "notes": "",
        "report": "",
    }


def collect_subgraph_updates(
    topic: str,
    *,
    wrapped: bool = False,
) -> list[tuple[tuple[str, ...], Any]]:
    graph = build_wrapped_graph() if wrapped else build_graph()
    return list(
        graph.stream(
            initial_state(topic),
            stream_mode="updates",
            subgraphs=True,
        )
    )


def self_test() -> None:
    graph = build_graph()
    result = graph.invoke(initial_state("subgraphs"))

    assert result["sources"] == [
        "official guide for subgraphs",
        "design notes for subgraphs",
    ]
    assert result["notes"].startswith("verified sources:")
    assert result["report"].startswith("Report: subgraphs\n")

    direct_events = collect_subgraph_updates("subgraphs")
    wrapped_events = collect_subgraph_updates("subgraphs", wrapped=True)
    for events in (direct_events, wrapped_events):
        namespaces = [namespace for namespace, _ in events]
        assert any(namespace for namespace in namespaces)
        assert any(not namespace for namespace in namespaces)

        nested_node_names = {
            node_name
            for namespace, update in events
            if namespace
            for node_name in update
        }
        assert {"find_sources", "extract_notes"} <= nested_node_names

        parent_updates = [update for namespace, update in events if not namespace]
        assert any("research" in update for update in parent_updates)
        assert any("render" in update for update in parent_updates)

    direct_research_update = next(
        update["research"]
        for namespace, update in direct_events
        if not namespace and "research" in update
    )
    wrapped_research_update = next(
        update["research"]
        for namespace, update in wrapped_events
        if not namespace and "research" in update
    )
    assert set(direct_research_update) == {"topic", "sources", "notes"}
    assert set(wrapped_research_update) == {"sources", "notes"}
    print("native_subgraph self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="subgraphs")
    parser.add_argument("--show-events", action="store_true")
    parser.add_argument("--compare-wrapper", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    graph = build_graph()
    result = graph.invoke(initial_state(args.topic))
    print(result["report"])

    if args.show_events:
        print("\nsubgraph-aware updates:")
        for namespace, update in collect_subgraph_updates(args.topic):
            pprint({"namespace": namespace, "update": update})

    if args.compare_wrapper:
        for label, wrapped in (("direct", False), ("wrapper", True)):
            print(f"\n{label} registration:")
            for namespace, update in collect_subgraph_updates(
                args.topic,
                wrapped=wrapped,
            ):
                pprint({"namespace": namespace, "update": update})


if __name__ == "__main__":
    main()
