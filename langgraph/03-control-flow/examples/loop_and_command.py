"""Compare a conditional-edge loop with a Command-driven loop.

Run:
    conda run -n langgraph python langgraph/03-control-flow/examples/loop_and_command.py
    conda run -n langgraph python langgraph/03-control-flow/examples/loop_and_command.py --self-test
"""

from __future__ import annotations

import argparse
from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class CounterState(TypedDict):
    target: int
    value: int
    trace: Annotated[list[int], add]
    status: str


def initialize(state: CounterState) -> dict[str, object]:
    target = state["target"]
    if target < 1:
        raise ValueError("target must be at least 1")
    return {"value": 0, "trace": [0], "status": "running"}


def increment(state: CounterState) -> dict[str, object]:
    next_value = state["value"] + 1
    return {"value": next_value, "trace": [next_value]}


def route_after_increment(state: CounterState) -> Literal["again", "done"]:
    if state["value"] < state["target"]:
        return "again"
    return "done"


def finish(state: CounterState) -> dict[str, str]:
    return {"status": f"done at {state['value']}"}


def build_conditional_loop():
    builder = StateGraph(CounterState)
    builder.add_node("initialize", initialize)
    builder.add_node("increment", increment)
    builder.add_node("finish", finish)
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "increment")
    builder.add_conditional_edges(
        "increment",
        route_after_increment,
        {"again": "increment", "done": "finish"},
    )
    builder.add_edge("finish", END)
    return builder.compile()


def command_increment(
    state: CounterState,
) -> Command[Literal["command_increment", "finish"]]:
    next_value = state["value"] + 1
    destination = "command_increment" if next_value < state["target"] else "finish"
    return Command(
        update={"value": next_value, "trace": [next_value]},
        goto=destination,
    )


def build_command_loop():
    builder = StateGraph(CounterState)
    builder.add_node("initialize", initialize)
    builder.add_node("command_increment", command_increment)
    builder.add_node("finish", finish)
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "command_increment")
    builder.add_edge("finish", END)
    return builder.compile()


def run(kind: Literal["conditional", "command"], target: int) -> CounterState:
    if kind == "conditional":
        graph = build_conditional_loop()
    elif kind == "command":
        graph = build_command_loop()
    else:
        raise ValueError(f"unknown loop kind: {kind}")

    config = {"recursion_limit": max(25, target + 5)}
    return graph.invoke({"target": target, "trace": []}, config=config)


def self_test() -> None:
    for target in (1, 4):
        conditional = run("conditional", target)
        command = run("command", target)
        expected_trace = list(range(target + 1))

        for result in (conditional, command):
            assert result["target"] == target
            assert result["value"] == target
            assert result["trace"] == expected_trace
            assert result["status"] == f"done at {target}"

    try:
        run("command", 0)
    except ValueError as exc:
        assert str(exc) == "target must be at least 1"
    else:
        raise AssertionError("target=0 must fail")

    print("loop_and_command self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", choices=["conditional", "command", "both"], default="both"
    )
    parser.add_argument("--target", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    kinds = ("conditional", "command") if args.kind == "both" else (args.kind,)
    for kind in kinds:
        result = run(kind, args.target)
        print(
            f"kind={kind} value={result['value']} "
            f"trace={result['trace']} status={result['status']}"
        )


if __name__ == "__main__":
    main()
