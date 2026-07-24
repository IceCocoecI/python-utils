from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from .nodes import (
    assess_refund,
    begin_ticket,
    classify_intent,
    clarify_refund,
    compose_response,
    decline_refund,
    dispatch_billing_checks,
    draft_faq_answer,
    execute_refund,
    load_profile,
    persist_case_memory,
    plan_billing_checks,
    record_faq_trace,
    request_refund_approval,
    route_after_approval,
    route_intent,
    route_refund,
    run_billing_check,
    search_faq,
    synthesize_billing,
    validate_context_access,
)
from .schemas import FaqState, SupportContext, SupportInput, SupportState


class SupportApp:
    """Authorization gateway around the compiled graph.

    Validation happens before LangGraph can write new input or a resume value to
    the checkpointer. Thread-scoped reads are authorized as well, and mutating
    compiled-graph methods such as ``update_state`` are intentionally not exposed.
    """

    __slots__ = ("__compiled",)

    def __init__(self, graph: Any) -> None:
        self.__compiled = graph

    def _authorize(
        self,
        graph_input: Any,
        config: dict | None,
        context: SupportContext | None,
    ) -> None:
        if context is None:
            raise ValueError("SupportContext is required")
        snapshot = self.__compiled.get_state(config)
        values = snapshot.values
        validate_context_access(
            context,
            owner_user_id=values.get("owner_user_id"),
            owner_tenant_id=values.get("owner_tenant_id"),
        )

        if isinstance(graph_input, Command) and graph_input.resume is not None:
            if context.role not in {"reviewer", "admin"}:
                raise PermissionError(
                    "only reviewer or admin roles may resume approval"
                )
            decision = graph_input.resume
            if (
                not isinstance(decision, dict)
                or type(decision.get("approved")) is not bool
            ):
                raise ValueError(
                    "resume payload must contain an exact boolean 'approved'"
                )

    def invoke(
        self,
        graph_input: Any,
        config: dict | None = None,
        *,
        context: SupportContext | None = None,
        **kwargs: Any,
    ) -> Any:
        self._authorize(graph_input, config, context)
        return self.__compiled.invoke(
            graph_input,
            config,
            context=context,
            **kwargs,
        )

    def stream(
        self,
        graph_input: Any,
        config: dict | None = None,
        *,
        context: SupportContext | None = None,
        **kwargs: Any,
    ) -> Any:
        self._authorize(graph_input, config, context)
        return self.__compiled.stream(
            graph_input,
            config,
            context=context,
            **kwargs,
        )

    def get_state(
        self,
        config: dict,
        *,
        context: SupportContext,
        **kwargs: Any,
    ) -> Any:
        self._authorize(None, config, context)
        return self.__compiled.get_state(config, **kwargs)

    def get_state_history(
        self,
        config: dict,
        *,
        context: SupportContext,
        **kwargs: Any,
    ) -> Any:
        self._authorize(None, config, context)
        return self.__compiled.get_state_history(config, **kwargs)

    def get_graph(self, *args: Any, **kwargs: Any) -> Any:
        """Return static topology; this method does not read thread state."""
        return self.__compiled.get_graph(*args, **kwargs)


def build_faq_subgraph():
    builder = StateGraph(FaqState, context_schema=SupportContext)
    builder.add_node("search_faq", search_faq)
    builder.add_node("draft_faq_answer", draft_faq_answer)
    builder.add_edge(START, "search_faq")
    builder.add_edge("search_faq", "draft_faq_answer")
    builder.add_edge("draft_faq_answer", END)
    return builder.compile()


def build_support_graph(
    *,
    checkpointer: InMemorySaver,
    store: InMemoryStore,
) -> SupportApp:
    builder = StateGraph(
        SupportState,
        context_schema=SupportContext,
        input_schema=SupportInput,
    )

    builder.add_node("begin_ticket", begin_ticket)
    builder.add_node("load_profile", load_profile)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("faq_flow", build_faq_subgraph())
    builder.add_node("record_faq_trace", record_faq_trace)
    builder.add_node("plan_billing_checks", plan_billing_checks)
    builder.add_node("run_billing_check", run_billing_check)
    builder.add_node("synthesize_billing", synthesize_billing)
    builder.add_node("assess_refund", assess_refund)
    builder.add_node("clarify_refund", clarify_refund)
    builder.add_node("request_refund_approval", request_refund_approval)
    builder.add_node("execute_refund", execute_refund)
    builder.add_node("decline_refund", decline_refund)
    builder.add_node("compose_response", compose_response)
    builder.add_node("persist_case_memory", persist_case_memory)

    builder.add_edge(START, "begin_ticket")
    builder.add_edge("begin_ticket", "load_profile")
    builder.add_edge("load_profile", "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "faq": "faq_flow",
            "billing": "plan_billing_checks",
            "refund": "assess_refund",
        },
    )

    builder.add_edge("faq_flow", "record_faq_trace")
    builder.add_edge("record_faq_trace", "compose_response")

    builder.add_conditional_edges(
        "plan_billing_checks",
        dispatch_billing_checks,
        ["run_billing_check"],
    )
    builder.add_edge("run_billing_check", "synthesize_billing")
    builder.add_edge("synthesize_billing", "compose_response")

    builder.add_conditional_edges(
        "assess_refund",
        route_refund,
        {
            "auto": "execute_refund",
            "approval": "request_refund_approval",
            "clarify": "clarify_refund",
        },
    )
    builder.add_conditional_edges(
        "request_refund_approval",
        route_after_approval,
        {"execute": "execute_refund", "decline": "decline_refund"},
    )
    builder.add_edge("execute_refund", "compose_response")
    builder.add_edge("decline_refund", "compose_response")
    builder.add_edge("clarify_refund", "compose_response")

    builder.add_edge("compose_response", "persist_case_memory")
    builder.add_edge("persist_case_memory", END)

    compiled = builder.compile(checkpointer=checkpointer, store=store)
    return SupportApp(compiled)


def create_support_app() -> tuple[SupportApp, InMemoryStore, InMemorySaver]:
    store = InMemoryStore()
    checkpointer = InMemorySaver()
    graph = build_support_graph(checkpointer=checkpointer, store=store)
    return graph, store, checkpointer


def seed_profile(
    store: InMemoryStore,
    user_id: str,
    *,
    reply_style: str = "concise",
    tier: str = "standard",
    tenant_id: str = "default",
) -> None:
    store.put(
        ("tenants", tenant_id, "users", user_id, "profile"),
        "settings",
        {"reply_style": reply_style, "tier": tier},
    )
