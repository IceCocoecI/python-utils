"""Inspect checkpoint history, branch from the past, and edit State.

Run:
    conda run -n langgraph python langgraph/05-persistence/examples/history_and_time_travel.py
    conda run -n langgraph python langgraph/05-persistence/examples/history_and_time_travel.py --self-test
"""

from __future__ import annotations

import argparse
from operator import add
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import StateSnapshot


class CounterState(TypedDict, total=False):
    value: int
    delta: int
    audit: Annotated[list[str], add]


def apply_delta(state: CounterState) -> dict[str, object]:
    before = state.get("value", 0)
    delta = state["delta"]
    after = before + delta
    return {
        "value": after,
        "audit": [f"{before}{delta:+d}={after}"],
    }


def build_graph(checkpointer: InMemorySaver | None = None):
    saver = checkpointer if checkpointer is not None else InMemorySaver()
    builder = StateGraph(CounterState)
    builder.add_node("apply_delta", apply_delta)
    builder.add_edge(START, "apply_delta")
    builder.add_edge("apply_delta", END)
    return builder.compile(checkpointer=saver)


def thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def find_completed_value(
    history: list[StateSnapshot],
    value: int,
) -> StateSnapshot:
    for snapshot in history:
        if snapshot.values.get("value") == value and snapshot.next == ():
            return snapshot
    raise AssertionError(f"no completed checkpoint with value={value}")


def self_test() -> None:
    graph = build_graph()
    config = thread_config("counter-test")

    first = graph.invoke({"value": 10, "delta": 2, "audit": []}, config)
    second = graph.invoke({"delta": 5}, config)
    assert first["value"] == 12
    assert second["value"] == 17

    # get_state_history returns newest checkpoints first.
    history = list(graph.get_state_history(config))
    assert len(history) >= 4
    checkpoint_at_12 = find_completed_value(history, 12)

    # Supplying a historical checkpoint config creates a new branch from it.
    branch = graph.invoke({"delta": 100}, checkpoint_at_12.config)
    assert branch["value"] == 112
    assert branch["audit"] == ["10+2=12", "12+100=112"]

    # The historical checkpoint still exists; time travel did not delete history.
    old_snapshot = graph.get_state(checkpoint_at_12.config)
    assert old_snapshot.values["value"] == 12

    # update_state also creates a checkpoint. Reducers still apply to the patch.
    corrected_config = graph.update_state(
        config,
        {"value": 40, "audit": ["manual correction -> 40"]},
    )
    corrected_snapshot = graph.get_state(corrected_config)
    assert corrected_snapshot.values["value"] == 40
    assert corrected_snapshot.values["audit"][-1] == "manual correction -> 40"

    corrected = graph.invoke({"delta": 2}, corrected_config)
    assert corrected["value"] == 42
    assert corrected["audit"][-1] == "40+2=42"
    print("history_and_time_travel self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    graph = build_graph()
    config = thread_config("counter-demo")

    print("normal execution:")
    print(graph.invoke({"value": 10, "delta": 2, "audit": []}, config))
    print(graph.invoke({"delta": 5}, config))

    history = list(graph.get_state_history(config))
    print("\ncheckpoint history (newest first):")
    for snapshot in history:
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        source = snapshot.metadata.get("source")
        step = snapshot.metadata.get("step")
        print(
            f"value={snapshot.values.get('value')!r}; next={snapshot.next}; "
            f"source={source}; step={step}; checkpoint_id={checkpoint_id}"
        )

    checkpoint_at_12 = find_completed_value(history, 12)
    branch = graph.invoke({"delta": 100}, checkpoint_at_12.config)
    print("\nbranch from value=12 with delta=100:")
    print(branch)

    corrected_config = graph.update_state(
        config,
        {"value": 40, "audit": ["manual correction -> 40"]},
    )
    print("\nafter update_state:")
    print(graph.get_state(corrected_config).values)
    print("after one new invocation:")
    print(graph.invoke({"delta": 2}, corrected_config))


if __name__ == "__main__":
    main()
