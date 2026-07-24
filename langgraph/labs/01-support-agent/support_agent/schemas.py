from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Literal
from uuid import uuid4

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


Intent = Literal["faq", "billing", "refund"]
RiskLevel = Literal["low", "high", "clarify"]
SupportRole = Literal["customer", "reviewer", "admin"]


class Evidence(TypedDict):
    ticket_id: str
    check: str
    finding: str


def merge_evidence(current: list[Evidence], updates: list[Evidence]) -> list[Evidence]:
    """Merge parallel findings and replace a repeated check deterministically."""
    merged = {(item["ticket_id"], item["check"]): item for item in current}
    for item in updates:
        merged[(item["ticket_id"], item["check"])] = item
    return [merged[key] for key in sorted(merged)]


class SupportInput(TypedDict):
    ticket_id: str
    turn_id: str
    query: str
    messages: list[AnyMessage]


class SupportState(TypedDict, total=False):
    owner_user_id: str
    owner_tenant_id: str
    ticket_id: str
    turn_id: str
    query: str
    messages: Annotated[list[AnyMessage], add_messages]
    intent: Intent | None
    profile: dict[str, str]
    faq_article: str
    faq_trace: list[str]
    checks: list[str]
    current_check: str
    evidence: Annotated[list[Evidence], merge_evidence]
    amount: str | None
    risk_level: RiskLevel | None
    refund_issue: str | None
    approval: dict[str, object] | None
    resolution: str
    trace: Annotated[list[str], operator.add]


class FaqState(TypedDict, total=False):
    query: str
    profile: dict[str, str]
    faq_article: str
    resolution: str
    faq_trace: Annotated[list[str], operator.add]


@dataclass(frozen=True)
class SupportContext:
    """Trusted request identity and policy, supplied by the host application.

    ``user_id`` is the customer/case owner. ``actor_id`` is the authenticated
    caller performing the current action and is required for staff approval.
    Neither value is accepted through graph input state.
    """

    user_id: str
    tenant_id: str = "default"
    actor_id: str | None = None
    locale: str = "en-US"
    role: SupportRole = "customer"
    auto_refund_limit: float = 500.0


def new_ticket(
    ticket_id: str,
    query: str,
    *,
    turn_id: str | None = None,
) -> SupportInput:
    if not ticket_id.strip():
        raise ValueError("ticket_id must not be empty")
    if not query.strip():
        raise ValueError("query must not be empty")
    resolved_turn_id = turn_id or uuid4().hex
    if not resolved_turn_id.strip():
        raise ValueError("turn_id must not be empty")
    return {
        "ticket_id": ticket_id,
        "turn_id": resolved_turn_id,
        "query": query,
        "messages": [
            HumanMessage(
                content=query,
                id=f"human-{resolved_turn_id}",
            )
        ],
    }
