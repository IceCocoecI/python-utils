"""Express a small workflow with Functional API entrypoint and tasks.

Run:
    conda run -n langgraph python langgraph/09-streaming-and-resilience/examples/functional_workflow.py
    conda run -n langgraph python langgraph/09-streaming-and-resilience/examples/functional_workflow.py --self-test
"""

from __future__ import annotations

import argparse
from typing import TypedDict

from langgraph.func import entrypoint, task


class WorkflowInput(TypedDict):
    text: str
    minimum_length: int


class WorkflowOutput(TypedDict):
    normalized: str
    word_count: int
    long_words: list[str]


@task
def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


@task
def count_words(text: str) -> int:
    return len(text.split())


@task
def select_long_words(payload: tuple[str, int]) -> list[str]:
    text, minimum_length = payload
    return sorted({word for word in text.split() if len(word) >= minimum_length})


@entrypoint()
def analyze_text(inputs: WorkflowInput) -> WorkflowOutput:
    """Use ordinary Python variables while tasks remain observable units."""

    normalized = normalize(inputs["text"]).result()

    count_future = count_words(normalized)
    long_words_future = select_long_words((normalized, inputs["minimum_length"]))

    return {
        "normalized": normalized,
        "word_count": count_future.result(),
        "long_words": long_words_future.result(),
    }


def run(text: str, minimum_length: int) -> WorkflowOutput:
    return analyze_text.invoke({"text": text, "minimum_length": minimum_length})


def self_test() -> None:
    result = run("  Graph workflows stay explicit graph  ", 6)
    assert result == {
        "normalized": "graph workflows stay explicit graph",
        "word_count": 5,
        "long_words": ["explicit", "workflows"],
    }

    task_updates = list(
        analyze_text.stream(
            {"text": "small functional workflow", "minimum_length": 8},
            stream_mode="updates",
        )
    )
    assert task_updates
    assert any("analyze_text" in update for update in task_updates)
    print("functional_workflow self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="Graph workflows stay explicit graph")
    parser.add_argument("--minimum-length", type=int, default=6)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    result = run(args.text, args.minimum_length)
    print(result)


if __name__ == "__main__":
    main()
