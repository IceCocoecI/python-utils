from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from langgraph.types import Send, interrupt

from .schemas import Evidence, SupportContext, SupportState


FAQ_ARTICLES = {
    "password": "Reset the password from Settings > Security > Reset password.",
    "shipping": "Standard shipping takes three to five business days.",
    "returns": "Returns are accepted within 30 days if the item is unused.",
    "default": "A support specialist will reply within one business day.",
}

_ROLES = {"customer", "reviewer", "admin"}
_APPROVER_ROLES = {"reviewer", "admin"}
_MONEY_TOKEN = re.compile(r"(?<![\w.])[-+]?(?:\d[\d,]*)(?:\.\d+)?(?![\w.])")


def _store(runtime: Runtime[SupportContext]):
    if runtime.store is None:
        raise RuntimeError("This workflow requires a Store")
    return runtime.store


def validate_context_access(
    context: SupportContext,
    *,
    owner_user_id: str | None = None,
    owner_tenant_id: str | None = None,
) -> str:
    """Validate a trusted caller against an optional persisted thread owner."""
    user_id = context.user_id.strip()
    tenant_id = context.tenant_id.strip()
    if not user_id or not tenant_id:
        raise ValueError("trusted context requires non-empty user_id and tenant_id")
    if context.role not in _ROLES:
        raise PermissionError(f"unsupported support role: {context.role!r}")

    actor_id = user_id if context.actor_id is None else context.actor_id.strip()
    if not actor_id:
        raise ValueError("trusted context requires a non-empty actor identity")
    if context.role == "customer" and actor_id != user_id:
        raise PermissionError("a customer cannot act as another identity")
    if context.role in _APPROVER_ROLES and context.actor_id is None:
        raise ValueError("staff context must provide an explicit actor_id")

    if owner_user_id is not None and owner_user_id != user_id:
        raise PermissionError("thread owner does not match trusted context")
    if owner_tenant_id is not None and owner_tenant_id != tenant_id:
        raise PermissionError("thread tenant does not match trusted context")
    return actor_id


def _validate_thread_access(
    state: SupportState,
    runtime: Runtime[SupportContext],
) -> str:
    """Validate trusted context against the owner persisted in checkpoint state."""
    return validate_context_access(
        runtime.context,
        owner_user_id=state.get("owner_user_id"),
        owner_tenant_id=state.get("owner_tenant_id"),
    )


def begin_ticket(
    state: SupportState,
    runtime: Runtime[SupportContext],
) -> dict:
    """Bind the thread owner and clear state that belongs to the previous ticket."""
    _validate_thread_access(state, runtime)
    return {
        "owner_user_id": runtime.context.user_id.strip(),
        "owner_tenant_id": runtime.context.tenant_id.strip(),
        "intent": None,
        "profile": {},
        "faq_article": "",
        "faq_trace": [],
        "checks": [],
        "current_check": "",
        "amount": None,
        "risk_level": None,
        "refund_issue": None,
        "approval": None,
        "resolution": "",
        "trace": [f"ticket:started:{state['turn_id']}"],
    }


def _user_namespace(context: SupportContext, suffix: str) -> tuple[str, ...]:
    return (
        "tenants",
        context.tenant_id,
        "users",
        context.user_id,
        suffix,
    )


def load_profile(
    state: SupportState,
    runtime: Runtime[SupportContext],
) -> dict:
    _validate_thread_access(state, runtime)
    namespace = _user_namespace(runtime.context, "profile")
    item = _store(runtime).get(namespace, "settings")
    profile = (
        dict(item.value)
        if item is not None
        else {"reply_style": "detailed", "tier": "standard"}
    )
    return {"profile": profile, "trace": ["profile:loaded"]}


def classify_intent(state: SupportState) -> dict:
    query = state["query"].lower()
    policy_question = any(
        phrase in query
        for phrase in (
            "return policy",
            "refund policy",
            "how do returns work",
            "what is your return",
            "退货政策",
            "退款政策",
        )
    )
    if policy_question:
        intent = "faq"
    elif any(word in query for word in ("refund", "return", "退款", "退货")):
        intent = "refund"
    elif any(word in query for word in ("bill", "invoice", "charge", "账单")):
        intent = "billing"
    else:
        intent = "faq"
    return {"intent": intent, "trace": [f"intent:{intent}"]}


def route_intent(state: SupportState) -> Literal["faq", "billing", "refund"]:
    intent = state["intent"]
    if intent is None:
        raise RuntimeError("intent must be classified before routing")
    return intent


def search_faq(state: SupportState) -> dict:
    query = state["query"].lower()
    article_key = next(
        (key for key in ("password", "shipping") if key in query),
        "default",
    )
    if any(word in query for word in ("return", "refund", "退货", "退款")):
        article_key = "returns"
    return {
        "faq_article": FAQ_ARTICLES[article_key],
        "faq_trace": [f"faq:search:{article_key}"],
    }


def draft_faq_answer(state: SupportState) -> dict:
    style = state.get("profile", {}).get("reply_style", "detailed")
    answer = f"{state['faq_article']} (style={style})"
    return {"resolution": answer, "faq_trace": ["faq:drafted"]}


def record_faq_trace(state: SupportState) -> dict:
    return {"trace": list(state.get("faq_trace", []))}


def plan_billing_checks(state: SupportState) -> dict:
    return {
        "checks": ["payment", "invoice", "duplicates"],
        "trace": ["billing:planned"],
    }


def dispatch_billing_checks(state: SupportState) -> list[Send]:
    return [
        Send(
            "run_billing_check",
            {
                "ticket_id": state["ticket_id"],
                "query": state["query"],
                "current_check": check,
            },
        )
        for check in state["checks"]
    ]


def run_billing_check(state: SupportState) -> dict:
    check = state["current_check"]
    query = state["query"].lower()
    findings = {
        "payment": "payment was captured successfully",
        "invoice": "invoice total matches the order",
        "duplicates": (
            "possible duplicate charge detected"
            if "duplicate" in query
            else "no duplicate charge detected"
        ),
    }
    evidence: Evidence = {
        "ticket_id": state["ticket_id"],
        "check": check,
        "finding": findings[check],
    }
    return {
        "evidence": [evidence],
        "trace": [f"billing:checked:{check}"],
    }


def synthesize_billing(state: SupportState) -> dict:
    current = [
        item
        for item in state.get("evidence", [])
        if item["ticket_id"] == state["ticket_id"]
    ]
    details = "; ".join(
        f"{item['check']}={item['finding']}"
        for item in sorted(current, key=lambda item: item["check"])
    )
    return {
        "resolution": f"Billing investigation: {details}",
        "trace": ["billing:synthesized"],
    }


def _parse_refund_amount(query: str) -> tuple[str | None, str | None]:
    if re.search(r"\b(?:nan|inf(?:inity)?)\b", query, re.I):
        return None, "amount must be a finite decimal value"

    tokens = _MONEY_TOKEN.findall(query)
    if not tokens:
        return None, "refund amount is missing"
    if len(tokens) != 1:
        return None, "multiple numeric values make the refund amount ambiguous"

    token = tokens[0]
    if "," in token:
        return None, "comma-formatted amounts require locale-aware clarification"
    try:
        amount = Decimal(token)
    except InvalidOperation:
        return None, "refund amount is not a valid decimal value"
    if not amount.is_finite() or amount <= 0:
        return None, "refund amount must be finite and greater than zero"
    try:
        has_currency_precision = amount.quantize(Decimal("0.01")) == amount
    except InvalidOperation:
        return None, "refund amount is outside the supported decimal range"
    if not has_currency_precision:
        return None, "refund amount cannot have more than two decimal places"
    return format(amount.normalize(), "f"), None


def _validated_auto_refund_limit(value: object) -> Decimal | None:
    try:
        limit = Decimal(str(value))
    except InvalidOperation:
        return None
    if not limit.is_finite() or limit < 0:
        return None
    return limit


def assess_refund(
    state: SupportState,
    runtime: Runtime[SupportContext],
) -> dict:
    amount, issue = _parse_refund_amount(state["query"])
    if issue is not None:
        return {
            "amount": None,
            "risk_level": "clarify",
            "refund_issue": issue,
            "trace": ["refund:clarification-required"],
        }

    limit = _validated_auto_refund_limit(runtime.context.auto_refund_limit)
    if limit is None:
        return {
            "amount": amount,
            "risk_level": "high",
            "refund_issue": "automatic refund policy is invalid; manual review required",
            "trace": ["refund:limit-invalid", "refund:risk:high"],
        }

    risk_level = "high" if Decimal(amount) > limit else "low"
    return {
        "amount": amount,
        "risk_level": risk_level,
        "refund_issue": None,
        "trace": [f"refund:risk:{risk_level}"],
    }


def route_refund(state: SupportState) -> Literal["auto", "approval", "clarify"]:
    risk_level = state["risk_level"]
    if risk_level == "clarify":
        return "clarify"
    if risk_level == "high":
        return "approval"
    if risk_level == "low":
        return "auto"
    raise RuntimeError("refund risk must be assessed before routing")


def clarify_refund(state: SupportState) -> dict:
    issue = state.get("refund_issue") or "refund details require clarification"
    return {
        "resolution": f"Refund was not executed: {issue}.",
        "trace": ["refund:clarification-requested"],
    }


def request_refund_approval(
    state: SupportState,
    runtime: Runtime[SupportContext],
) -> dict:
    _validate_thread_access(state, runtime)
    decision = interrupt(
        {
            "kind": "refund_approval",
            "ticket_id": state["ticket_id"],
            "amount": state["amount"],
            "reason": state.get("refund_issue"),
            "question": "Approve this high-value refund?",
        }
    )
    reviewer = _validate_thread_access(state, runtime)
    if runtime.context.role not in _APPROVER_ROLES:
        raise PermissionError("only reviewer or admin roles may decide a refund")
    if not isinstance(decision, dict) or type(decision.get("approved")) is not bool:
        raise ValueError("resume payload must contain an exact boolean 'approved'")
    approved = decision["approved"]
    return {
        "approval": {
            "approved": approved,
            "reviewer": reviewer,
            "role": runtime.context.role,
        },
        "trace": [f"refund:reviewed:{reviewer}"],
    }


def route_after_approval(state: SupportState) -> Literal["execute", "decline"]:
    approval = state.get("approval")
    if approval is None or type(approval.get("approved")) is not bool:
        raise RuntimeError("a strict approval decision is required")
    return "execute" if approval["approved"] is True else "decline"


def decline_refund(state: SupportState) -> dict:
    if state.get("amount") is None:
        raise RuntimeError("refund amount is required before decline")
    return {
        "resolution": (
            f"Refund for {Decimal(state['amount']):.2f} was declined by review."
        ),
        "trace": ["refund:declined"],
    }


def _refund_fingerprint(state: SupportState) -> str:
    payload = {
        "amount": state["amount"],
        "currency": "USD",
        "owner_tenant_id": state["owner_tenant_id"],
        "owner_user_id": state["owner_user_id"],
        "ticket_id": state["ticket_id"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def execute_refund(
    state: SupportState,
    runtime: Runtime[SupportContext],
) -> dict:
    _validate_thread_access(state, runtime)
    if state.get("amount") is None:
        raise RuntimeError("refund amount is required before execution")

    namespace = (
        "tenants",
        runtime.context.tenant_id,
        "operations",
        runtime.context.user_id,
        "refunds",
    )
    store = _store(runtime)
    fingerprint = _refund_fingerprint(state)
    existing = store.get(namespace, state["ticket_id"])
    if existing is not None:
        existing_fingerprint = existing.value.get("request_fingerprint")
        existing_amount = str(existing.value["amount"])
        if existing_fingerprint != fingerprint:
            return {
                "resolution": (
                    f"Refund was not executed: ticket {state['ticket_id']} already "
                    f"belongs to a {Decimal(existing_amount):.2f} refund; the "
                    f"requested {Decimal(state['amount']):.2f} payload conflicts."
                ),
                "trace": ["refund:idempotency-conflict"],
            }
        operation_id = str(existing.value["operation_id"])
        completed_amount = existing_amount
        status = str(existing.value["status"])
        trace = "refund:deduplicated"
    else:
        operation_id = f"rf-{state['ticket_id']}"
        completed_amount = state["amount"]
        status = "completed"
        store.put(
            namespace,
            state["ticket_id"],
            {
                "operation_id": operation_id,
                "amount": completed_amount,
                "currency": "USD",
                "status": status,
                "request_fingerprint": fingerprint,
            },
        )
        trace = "refund:executed"
    return {
        "resolution": (
            f"Refund {operation_id} for {Decimal(completed_amount):.2f} is {status}."
        ),
        "trace": [trace],
    }


def compose_response(state: SupportState) -> dict:
    message = AIMessage(
        content=state["resolution"],
        id=f"assistant-{state['turn_id']}",
    )
    return {"messages": [message], "trace": ["response:composed"]}


def persist_case_memory(
    state: SupportState,
    runtime: Runtime[SupportContext],
) -> dict:
    _validate_thread_access(state, runtime)
    namespace = _user_namespace(runtime.context, "support_history")
    _store(runtime).put(
        namespace,
        state["ticket_id"],
        {
            "intent": state["intent"],
            "resolution": state["resolution"],
            "locale": runtime.context.locale,
        },
    )
    return {"trace": ["memory:persisted"]}
