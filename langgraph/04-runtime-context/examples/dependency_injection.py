"""Inject a replaceable local dependency through Runtime context.

Run:
    conda run -n langgraph python langgraph/04-runtime-context/examples/dependency_injection.py
    conda run -n langgraph python langgraph/04-runtime-context/examples/dependency_injection.py --self-test
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime


PriceLookup = Callable[[str], int]


@dataclass(frozen=True)
class CheckoutContext:
    """Dependencies and settings that stay stable during one invocation."""

    tenant_id: str
    currency: str
    tax_basis_points: int
    lookup_price_cents: PriceLookup


class CheckoutState(TypedDict):
    sku: str
    quantity: int
    unit_price_cents: int
    subtotal_cents: int
    tax_cents: int
    total_cents: int
    summary: str


def load_price(
    state: CheckoutState,
    runtime: Runtime[CheckoutContext],
) -> dict[str, int]:
    if state["quantity"] <= 0:
        raise ValueError("quantity must be positive")

    unit_price = runtime.context.lookup_price_cents(state["sku"])
    return {
        "unit_price_cents": unit_price,
        "subtotal_cents": unit_price * state["quantity"],
    }


def calculate_tax(
    state: CheckoutState,
    runtime: Runtime[CheckoutContext],
) -> dict[str, int]:
    # Basis points keep this deterministic and avoid floating-point money math.
    tax = (state["subtotal_cents"] * runtime.context.tax_basis_points + 5_000) // 10_000
    return {
        "tax_cents": tax,
        "total_cents": state["subtotal_cents"] + tax,
    }


def render_summary(
    state: CheckoutState,
    runtime: Runtime[CheckoutContext],
) -> dict[str, str]:
    amount = f"{state['total_cents'] // 100}.{state['total_cents'] % 100:02d}"
    context = runtime.context
    return {
        "summary": (
            f"tenant={context.tenant_id}; sku={state['sku']}; "
            f"quantity={state['quantity']}; total={context.currency} {amount}"
        )
    }


def build_graph():
    builder = StateGraph(CheckoutState, context_schema=CheckoutContext)
    builder.add_node("load_price", load_price)
    builder.add_node("calculate_tax", calculate_tax)
    builder.add_node("render", render_summary)
    builder.add_edge(START, "load_price")
    builder.add_edge("load_price", "calculate_tax")
    builder.add_edge("calculate_tax", "render")
    builder.add_edge("render", END)
    return builder.compile()


def make_price_lookup(catalog: dict[str, int]) -> PriceLookup:
    """Return a deterministic stand-in for a database or HTTP client."""

    def lookup(sku: str) -> int:
        try:
            return catalog[sku]
        except KeyError as exc:
            raise ValueError(f"unknown sku: {sku}") from exc

    return lookup


def run(sku: str, quantity: int, context: CheckoutContext) -> CheckoutState:
    return build_graph().invoke(
        {"sku": sku, "quantity": quantity},
        context=context,
    )


def self_test() -> None:
    retail = CheckoutContext(
        tenant_id="retail",
        currency="CNY",
        tax_basis_points=600,
        lookup_price_cents=make_price_lookup({"book": 2_500}),
    )
    wholesale = CheckoutContext(
        tenant_id="wholesale",
        currency="CNY",
        tax_basis_points=0,
        lookup_price_cents=make_price_lookup({"book": 2_000}),
    )

    retail_result = run("book", 2, retail)
    wholesale_result = run("book", 2, wholesale)

    assert retail_result["subtotal_cents"] == 5_000
    assert retail_result["tax_cents"] == 300
    assert retail_result["total_cents"] == 5_300
    assert retail_result["summary"].endswith("total=CNY 53.00")

    assert wholesale_result["subtotal_cents"] == 4_000
    assert wholesale_result["tax_cents"] == 0
    assert wholesale_result["total_cents"] == 4_000
    assert "tenant_id" not in retail_result
    assert "lookup_price_cents" not in retail_result

    try:
        run("missing", 1, retail)
    except ValueError as exc:
        assert str(exc) == "unknown sku: missing"
    else:
        raise AssertionError("an unknown sku must fail")

    print("dependency_injection self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sku", default="book")
    parser.add_argument("--quantity", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    context = CheckoutContext(
        tenant_id="demo-shop",
        currency="CNY",
        tax_basis_points=600,
        lookup_price_cents=make_price_lookup({"book": 2_500, "pen": 350}),
    )
    result = run(args.sku, args.quantity, context)
    print(result["summary"])
    print(
        "calculation: "
        f"subtotal={result['subtotal_cents']} cents, "
        f"tax={result['tax_cents']} cents"
    )


if __name__ == "__main__":
    main()
