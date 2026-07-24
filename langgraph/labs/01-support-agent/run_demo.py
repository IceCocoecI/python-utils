from __future__ import annotations

import argparse
from typing import Any

from langgraph.types import Command

from support_agent import (
    SupportContext,
    create_support_app,
    new_ticket,
    seed_profile,
)


def run_scenario(
    query: str,
    *,
    ticket_id: str,
    user_id: str,
    approve: bool,
    stream: bool = False,
) -> tuple[dict[str, Any], Any]:
    graph, store, _ = create_support_app()
    seed_profile(store, user_id, reply_style="concise", tier="gold")
    customer_context = SupportContext(user_id=user_id)
    reviewer_context = SupportContext(
        user_id=user_id,
        actor_id="demo-reviewer",
        role="reviewer",
    )
    config = {"configurable": {"thread_id": ticket_id}}
    initial = new_ticket(ticket_id, query)

    if stream:
        for event in graph.stream(
            initial,
            config,
            context=customer_context,
            stream_mode="updates",
        ):
            print(f"event: {event}")
        snapshot = graph.get_state(config, context=customer_context)
        if snapshot.interrupts:
            decision = {"approved": approve}
            for event in graph.stream(
                Command(resume=decision),
                config,
                context=reviewer_context,
                stream_mode="updates",
            ):
                print(f"event: {event}")
        result = dict(graph.get_state(config, context=customer_context).values)
    else:
        result = graph.invoke(initial, config, context=customer_context)
        if result.get("__interrupt__"):
            result = graph.invoke(
                Command(resume={"approved": approve}),
                config,
                context=reviewer_context,
            )

    return result, store


def print_summary(result: dict[str, Any]) -> None:
    print(f"intent: {result['intent']}")
    print(f"resolution: {result['resolution']}")
    print(f"trace: {' -> '.join(result['trace'])}")
    if result.get("evidence"):
        print(f"evidence_count: {len(result['evidence'])}")


def self_test() -> None:
    faq, _ = run_scenario(
        "How do I reset my password?",
        ticket_id="self-faq",
        user_id="self-user",
        approve=True,
    )
    assert faq["intent"] == "faq"
    assert "style=concise" in faq["resolution"]

    billing, _ = run_scenario(
        "I see a duplicate invoice charge",
        ticket_id="self-billing",
        user_id="self-user",
        approve=True,
    )
    findings = [
        item for item in billing["evidence"] if item["ticket_id"] == "self-billing"
    ]
    assert len(findings) == 3
    assert any("duplicate" in item["finding"] for item in findings)

    refund, store = run_scenario(
        "Please refund 1200",
        ticket_id="self-refund",
        user_id="self-user",
        approve=True,
    )
    assert refund["risk_level"] == "high"
    assert refund["approval"]["approved"] is True
    operation = store.get(
        ("tenants", "default", "operations", "self-user", "refunds"),
        "self-refund",
    )
    assert operation is not None
    print("self-test: ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="I see a duplicate invoice charge")
    parser.add_argument("--ticket-id", default="demo-ticket")
    parser.add_argument("--user-id", default="demo-user")
    parser.add_argument(
        "--approve",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    result, _ = run_scenario(
        args.query,
        ticket_id=args.ticket_id,
        user_id=args.user_id,
        approve=args.approve,
        stream=args.stream,
    )
    print_summary(result)


if __name__ == "__main__":
    main()
