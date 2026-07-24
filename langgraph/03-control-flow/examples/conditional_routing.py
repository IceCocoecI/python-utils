"""Route deterministic local requests through conditional edges.

Run:
    conda run -n langgraph python langgraph/03-control-flow/examples/conditional_routing.py
    conda run -n langgraph python langgraph/03-control-flow/examples/conditional_routing.py --self-test
"""

from __future__ import annotations

import argparse
from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class RouteState(TypedDict):
    request: str
    category: str
    response: str
    path: Annotated[list[str], add]


def classify(state: RouteState) -> dict[str, object]:
    request = state["request"].strip()
    if request.casefold().startswith("sum:"):
        category = "math"
    elif request.casefold().startswith("upper:"):
        category = "text"
    else:
        category = "fallback"
    return {"category": category, "path": ["classify"]}


def choose_branch(state: RouteState) -> Literal["math", "text", "fallback"]:
    category = state["category"]
    if category == "math":
        return "math"
    if category == "text":
        return "text"
    return "fallback"


def sum_numbers(state: RouteState) -> dict[str, object]:
    payload = state["request"].split(":", maxsplit=1)[1]
    values = [int(part) for part in payload.split()]
    return {"response": f"sum={sum(values)}", "path": ["sum"]}


def uppercase_text(state: RouteState) -> dict[str, object]:
    payload = state["request"].split(":", maxsplit=1)[1].strip()
    return {"response": payload.upper(), "path": ["uppercase"]}


def unsupported(state: RouteState) -> dict[str, object]:
    return {
        "response": f"unsupported request: {state['request'].strip()}",
        "path": ["fallback"],
    }


def finalize(state: RouteState) -> dict[str, list[str]]:
    return {"path": ["finalize"]}


def build_graph():
    builder = StateGraph(RouteState)
    builder.add_node("classify", classify)
    builder.add_node("sum", sum_numbers)
    builder.add_node("uppercase", uppercase_text)
    builder.add_node("fallback", unsupported)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        choose_branch,
        {
            "math": "sum",
            "text": "uppercase",
            "fallback": "fallback",
        },
    )
    builder.add_edge("sum", "finalize")
    builder.add_edge("uppercase", "finalize")
    builder.add_edge("fallback", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


def run(request: str) -> RouteState:
    return build_graph().invoke({"request": request, "path": []})


def self_test() -> None:
    math_result = run("sum: 3 5 8")
    assert math_result["category"] == "math"
    assert math_result["response"] == "sum=16"
    assert math_result["path"] == ["classify", "sum", "finalize"]

    text_result = run("upper: state is data")
    assert text_result["category"] == "text"
    assert text_result["response"] == "STATE IS DATA"
    assert text_result["path"] == ["classify", "uppercase", "finalize"]

    fallback_result = run("ping")
    assert fallback_result["category"] == "fallback"
    assert fallback_result["response"] == "unsupported request: ping"
    assert fallback_result["path"] == ["classify", "fallback", "finalize"]

    print("conditional_routing self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", nargs="?", default="sum: 2 4 6")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    result = run(args.request)
    print(f"category: {result['category']}")
    print(f"path: {' -> '.join(result['path'])}")
    print(f"response: {result['response']}")


if __name__ == "__main__":
    main()
