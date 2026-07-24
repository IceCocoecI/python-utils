"""Edit checkpointed state at a static breakpoint, then continue with None.

Run:
    conda run -n langgraph python langgraph/08-human-in-the-loop/examples/update_state_review.py
    conda run -n langgraph python langgraph/08-human-in-the-loop/examples/update_state_review.py --self-test
"""

from __future__ import annotations

import argparse
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class ReviewState(TypedDict):
    topic: str
    draft: str
    reviewed_by: str
    published: str


def write_draft(state: ReviewState) -> dict[str, str]:
    return {"draft": f"Draft about {state['topic']}: first version."}


def publish(state: ReviewState) -> dict[str, str]:
    reviewer = state["reviewed_by"] or "unreviewed"
    return {"published": f"{state['draft']} [published after review by {reviewer}]"}


def build_graph():
    builder = StateGraph(ReviewState)
    builder.add_node("draft", write_draft)
    builder.add_node("publish", publish)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "publish")
    builder.add_edge("publish", END)
    return builder.compile(
        checkpointer=InMemorySaver(),
        interrupt_before=["publish"],
    )


def run_review(topic: str, edited_draft: str, reviewer: str, thread_id: str):
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    graph.invoke(
        {
            "topic": topic,
            "draft": "",
            "reviewed_by": "",
            "published": "",
        },
        config,
    )
    before_edit = graph.get_state(config)

    graph.update_state(
        config,
        {"draft": edited_draft, "reviewed_by": reviewer},
    )
    after_edit = graph.get_state(config)

    final = graph.invoke(None, config)
    return before_edit, after_edit, final


def self_test() -> None:
    before, after, final = run_review(
        topic="state editing",
        edited_draft="Reviewed draft: state edits are checkpointed.",
        reviewer="morgan",
        thread_id="state-edit-test",
    )

    assert before.next == ("publish",)
    assert before.values["draft"] == ("Draft about state editing: first version.")
    assert after.next == ("publish",)
    assert after.values["draft"] == ("Reviewed draft: state edits are checkpointed.")
    assert after.values["reviewed_by"] == "morgan"
    assert final["published"] == (
        "Reviewed draft: state edits are checkpointed. "
        "[published after review by morgan]"
    )
    assert final["draft"] == after.values["draft"]
    print("update_state_review self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="checkpoint review")
    parser.add_argument(
        "--draft",
        default="Reviewed draft: checkpoints make edits explicit.",
    )
    parser.add_argument("--reviewer", default="morgan")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    before, after, final = run_review(
        args.topic,
        args.draft,
        args.reviewer,
        "state-edit-demo",
    )
    print(f"paused before: {before.next}")
    print(f"generated draft: {before.values['draft']}")
    print(f"edited draft: {after.values['draft']}")
    print(f"final: {final['published']}")


if __name__ == "__main__":
    main()
