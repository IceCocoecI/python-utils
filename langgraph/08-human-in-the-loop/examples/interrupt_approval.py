"""Pause for approval and resume with Command(resume=...).

Run:
    conda run -n langgraph python langgraph/08-human-in-the-loop/examples/interrupt_approval.py
    conda run -n langgraph python langgraph/08-human-in-the-loop/examples/interrupt_approval.py --reject
    conda run -n langgraph python langgraph/08-human-in-the-loop/examples/interrupt_approval.py --self-test
"""

from __future__ import annotations

import argparse
from operator import add
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict):
    request: str
    amount: int
    plan: str
    approved: bool | None
    reviewer: str
    result: str
    audit: Annotated[list[str], add]


def prepare_request(state: ApprovalState) -> dict[str, object]:
    plan = f"transfer {state['amount']} credits for {state['request']}"
    return {"plan": plan, "audit": ["request prepared"]}


def validate_decision(decision: Any) -> tuple[bool, str]:
    """Validate an untrusted resume payload without truthiness coercion."""

    if not isinstance(decision, dict):
        raise TypeError("resume value must be a dictionary")
    if "approved" not in decision:
        raise ValueError("resume value must include approved")
    if type(decision["approved"]) is not bool:
        raise TypeError("approved must be exactly true or false")
    if "reviewer" not in decision:
        raise ValueError("resume value must include reviewer")

    reviewer = decision["reviewer"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise TypeError("reviewer must be a non-empty string")
    return decision["approved"], reviewer.strip()


def request_approval(state: ApprovalState) -> dict[str, object]:
    """Interrupt execution; this function starts again when resumed."""

    decision = interrupt(
        {
            "question": "Approve this transfer?",
            "request": state["request"],
            "amount": state["amount"],
            "plan": state["plan"],
        }
    )
    approved, reviewer = validate_decision(decision)
    verdict = "approved" if approved else "rejected"
    return {
        "approved": approved,
        "reviewer": reviewer,
        "audit": [f"{verdict} by {reviewer}"],
    }


def route_decision(state: ApprovalState) -> Literal["execute", "reject"]:
    return "execute" if state["approved"] else "reject"


def execute_transfer(state: ApprovalState) -> dict[str, object]:
    return {
        "result": f"executed: {state['plan']}",
        "audit": ["transfer executed"],
    }


def reject_transfer(state: ApprovalState) -> dict[str, object]:
    return {
        "result": f"cancelled by {state['reviewer']}",
        "audit": ["transfer cancelled"],
    }


def build_graph():
    builder = StateGraph(ApprovalState)
    builder.add_node("prepare", prepare_request)
    builder.add_node("approval", request_approval)
    builder.add_node("execute", execute_transfer)
    builder.add_node("reject", reject_transfer)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "approval")
    builder.add_conditional_edges(
        "approval",
        route_decision,
        {"execute": "execute", "reject": "reject"},
    )
    builder.add_edge("execute", END)
    builder.add_edge("reject", END)
    return builder.compile(checkpointer=InMemorySaver())


def initial_state() -> ApprovalState:
    return {
        "request": "team training",
        "amount": 2500,
        "plan": "",
        "approved": None,
        "reviewer": "",
        "result": "",
        "audit": [],
    }


def run_once(approved: bool, thread_id: str) -> tuple[dict[str, Any], ApprovalState]:
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    paused = graph.invoke(initial_state(), config)
    final = graph.invoke(
        Command(resume={"approved": approved, "reviewer": "casey"}),
        config,
    )
    return paused, final


def self_test() -> None:
    graph = build_graph()
    approved_config = {"configurable": {"thread_id": "approval-test-yes"}}
    paused = graph.invoke(initial_state(), approved_config)

    assert "__interrupt__" in paused
    interrupts = paused["__interrupt__"]
    assert len(interrupts) == 1
    assert interrupts[0].value["amount"] == 2500
    snapshot = graph.get_state(approved_config)
    assert snapshot.next == ("approval",)
    assert snapshot.values["audit"] == ["request prepared"]

    approved = graph.invoke(
        Command(resume={"approved": True, "reviewer": "casey"}),
        approved_config,
    )
    assert approved["approved"] is True
    assert approved["result"].startswith("executed:")
    assert approved["audit"] == [
        "request prepared",
        "approved by casey",
        "transfer executed",
    ]
    assert graph.get_state(approved_config).next == ()

    rejected_config = {"configurable": {"thread_id": "approval-test-no"}}
    graph.invoke(initial_state(), rejected_config)
    rejected = graph.invoke(
        Command(resume={"approved": False, "reviewer": "riley"}),
        rejected_config,
    )
    assert rejected["approved"] is False
    assert rejected["result"] == "cancelled by riley"
    assert rejected["audit"][-1] == "transfer cancelled"

    invalid_decisions: list[tuple[object, type[Exception], str]] = [
        (
            {"approved": "false", "reviewer": "casey"},
            TypeError,
            "approved must be exactly true or false",
        ),
        (
            {"approved": 1, "reviewer": "casey"},
            TypeError,
            "approved must be exactly true or false",
        ),
        (
            {"reviewer": "casey"},
            ValueError,
            "resume value must include approved",
        ),
        (
            {"approved": True},
            ValueError,
            "resume value must include reviewer",
        ),
        (
            {"approved": True, "reviewer": "  "},
            TypeError,
            "reviewer must be a non-empty string",
        ),
    ]
    for index, (payload, exception_type, message) in enumerate(invalid_decisions):
        invalid_graph = build_graph()
        invalid_config = {
            "configurable": {"thread_id": f"approval-test-invalid-{index}"}
        }
        invalid_graph.invoke(initial_state(), invalid_config)
        try:
            invalid_graph.invoke(Command(resume=payload), invalid_config)
        except exception_type as exc:
            assert str(exc) == message
        else:
            raise AssertionError(f"invalid resume payload was accepted: {payload!r}")
    print("interrupt_approval self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    decision = parser.add_mutually_exclusive_group()
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    approved = not args.reject
    paused, final = run_once(approved, "approval-demo")
    interrupt_value = paused["__interrupt__"][0].value
    print(f"paused for review: {interrupt_value}")
    print(f"resumed decision: approved={final['approved']}")
    print(f"result: {final['result']}")
    print(f"audit: {final['audit']}")


if __name__ == "__main__":
    main()
