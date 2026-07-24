"""Use a supervisor to dispatch a dynamic worker team with Send.

Run:
    conda run -n langgraph python langgraph/10-subgraphs-and-multi-agent/examples/send_supervisor.py
    conda run -n langgraph python langgraph/10-subgraphs-and-multi-agent/examples/send_supervisor.py --self-test
"""

from __future__ import annotations

import argparse
from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class TeamState(TypedDict):
    request: str
    selected_workers: list[str]
    worker_results: Annotated[list[dict[str, str]], add]
    report: str


class WorkerInput(TypedDict):
    request: str
    worker: str


def supervisor_plan(state: TeamState) -> dict[str, list[str]]:
    """Select workers from request content; no worker runs here."""

    request = state["request"].casefold()
    selected: list[str] = []
    keyword_roles = [
        (("data", "metric", "analysis"), "analyst"),
        (("api", "service", "backend"), "engineer"),
        (("document", "guide", "tutorial"), "writer"),
    ]

    for keywords, role in keyword_roles:
        if any(keyword in request for keyword in keywords):
            selected.append(role)

    if not selected:
        selected.append("generalist")

    return {"selected_workers": selected}


def dispatch_workers(state: TeamState) -> list[Send]:
    """Create one independent worker invocation for every selected role."""

    return [
        Send(
            "worker",
            {"request": state["request"], "worker": worker},
        )
        for worker in state["selected_workers"]
    ]


def worker(state: WorkerInput) -> dict[str, list[dict[str, str]]]:
    role = state["worker"]
    request = state["request"]
    templates = {
        "analyst": f"metrics and risks for: {request}",
        "engineer": f"service design for: {request}",
        "writer": f"reader guide for: {request}",
        "generalist": f"general plan for: {request}",
    }
    return {"worker_results": [{"worker": role, "result": templates[role]}]}


def supervisor_synthesize(state: TeamState) -> dict[str, str]:
    """Join parallel updates only after the fan-out superstep completes."""

    ordered = sorted(state["worker_results"], key=lambda item: item["worker"])
    lines = [f"Supervisor report for: {state['request']}"]
    lines.extend(f"- {item['worker']}: {item['result']}" for item in ordered)
    return {"report": "\n".join(lines)}


def build_graph():
    builder = StateGraph(TeamState)
    builder.add_node("supervisor_plan", supervisor_plan)
    builder.add_node("worker", worker)
    builder.add_node("supervisor_synthesize", supervisor_synthesize)
    builder.add_edge(START, "supervisor_plan")
    builder.add_conditional_edges(
        "supervisor_plan",
        dispatch_workers,
        ["worker"],
    )
    builder.add_edge("worker", "supervisor_synthesize")
    builder.add_edge("supervisor_synthesize", END)
    return builder.compile()


def run(request: str) -> TeamState:
    return build_graph().invoke(
        {
            "request": request,
            "selected_workers": [],
            "worker_results": [],
            "report": "",
        }
    )


def self_test() -> None:
    request = "Build a data API and write a document"
    result = run(request)

    assert result["selected_workers"] == ["analyst", "engineer", "writer"]
    by_worker = {item["worker"]: item["result"] for item in result["worker_results"]}
    assert set(by_worker) == {"analyst", "engineer", "writer"}
    assert len(result["worker_results"]) == 3
    assert len(result["report"].splitlines()) == 4
    assert result["report"].splitlines()[1].startswith("- analyst:")

    fallback = run("Plan a small internal change")
    assert fallback["selected_workers"] == ["generalist"]
    assert fallback["worker_results"][0]["worker"] == "generalist"

    updates = list(
        build_graph().stream(
            {
                "request": request,
                "selected_workers": [],
                "worker_results": [],
                "report": "",
            },
            stream_mode="updates",
        )
    )
    worker_updates = [update for update in updates if "worker" in update]
    assert len(worker_updates) == 3
    print("send_supervisor self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request",
        default="Build a data API and write a document",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    result = run(args.request)
    print(f"selected workers: {result['selected_workers']}")
    print(result["report"])


if __name__ == "__main__":
    main()
