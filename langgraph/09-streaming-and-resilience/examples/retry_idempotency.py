"""Retry a transient post-commit failure without duplicating a side effect.

Run:
    conda run -n langgraph python langgraph/09-streaming-and-resilience/examples/retry_idempotency.py
    conda run -n langgraph python langgraph/09-streaming-and-resilience/examples/retry_idempotency.py --self-test
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy


class TransientAfterCommit(RuntimeError):
    """The remote side effect succeeded, but its response was lost."""


class IdempotencyConflict(RuntimeError):
    """An idempotency key was reused for a different business request."""


class ChargeRecord(TypedDict):
    amount: int
    receipt: str


@dataclass
class IdempotentLedger:
    """A deterministic stand-in for a service with idempotency keys."""

    attempts: int = 0
    records: dict[str, ChargeRecord] = field(default_factory=dict)
    failed_once: set[str] = field(default_factory=set)

    def charge(self, operation_id: str, amount: int) -> tuple[str, bool]:
        self.attempts += 1

        if operation_id in self.records:
            record = self.records[operation_id]
            if record["amount"] != amount:
                raise IdempotencyConflict(
                    f"operation {operation_id!r} already exists with "
                    f"amount={record['amount']}, not amount={amount}"
                )
            return record["receipt"], True

        receipt = f"receipt:{operation_id}:{amount}"
        self.records[operation_id] = {"amount": amount, "receipt": receipt}

        if operation_id not in self.failed_once:
            self.failed_once.add(operation_id)
            raise TransientAfterCommit("response lost after remote commit")

        return receipt, False


class PaymentState(TypedDict):
    operation_id: str
    amount: int
    receipt: str
    deduplicated: bool


def build_graph(ledger: IdempotentLedger):
    def charge_once(state: PaymentState) -> dict[str, object]:
        receipt, deduplicated = ledger.charge(
            state["operation_id"],
            state["amount"],
        )
        return {
            "receipt": receipt,
            "deduplicated": deduplicated,
        }

    retry = RetryPolicy(
        initial_interval=0.01,
        backoff_factor=1.0,
        max_interval=0.01,
        max_attempts=3,
        jitter=False,
        retry_on=TransientAfterCommit,
    )

    builder = StateGraph(PaymentState)
    builder.add_node("charge", charge_once, retry_policy=retry)
    builder.add_edge(START, "charge")
    builder.add_edge("charge", END)
    return builder.compile()


def run(operation_id: str, amount: int):
    ledger = IdempotentLedger()
    result = build_graph(ledger).invoke(
        {
            "operation_id": operation_id,
            "amount": amount,
            "receipt": "",
            "deduplicated": False,
        }
    )
    return ledger, result


def self_test() -> None:
    ledger, result = run("order-42", 900)

    assert ledger.attempts == 2
    assert ledger.records == {
        "order-42": {"amount": 900, "receipt": "receipt:order-42:900"}
    }
    assert result["receipt"] == "receipt:order-42:900"
    assert result["deduplicated"] is True

    same_receipt, reused = ledger.charge("order-42", 900)
    assert same_receipt == result["receipt"]
    assert reused is True
    assert len(ledger.records) == 1

    conflicting_graph = build_graph(ledger)
    try:
        conflicting_graph.invoke(
            {
                "operation_id": "order-42",
                "amount": 1_200,
                "receipt": "",
                "deduplicated": False,
            }
        )
    except IdempotencyConflict as exc:
        assert "amount=900" in str(exc)
        assert "amount=1200" in str(exc)
    else:
        raise AssertionError("a reused key with a different amount must fail")
    assert ledger.attempts == 4
    assert ledger.records["order-42"]["amount"] == 900
    print("retry_idempotency self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation-id", default="order-42")
    parser.add_argument("--amount", type=int, default=900)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    ledger, result = run(args.operation_id, args.amount)
    print(f"attempts: {ledger.attempts}")
    print(f"committed records: {ledger.records}")
    print(f"receipt: {result['receipt']}")
    print(f"retry reused prior commit: {result['deduplicated']}")


if __name__ == "__main__":
    main()
