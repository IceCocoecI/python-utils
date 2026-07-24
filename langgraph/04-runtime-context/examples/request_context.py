"""Inject trusted request metadata with context_schema and Runtime.

Run:
    conda run -n langgraph python langgraph/04-runtime-context/examples/request_context.py
    conda run -n langgraph python langgraph/04-runtime-context/examples/request_context.py --self-test
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime


Action = Literal["read", "delete"]
Locale = Literal["zh-CN", "en-US"]


@dataclass(frozen=True)
class RequestContext:
    """Trusted, run-scoped metadata supplied by the application."""

    user_id: str
    locale: Locale
    permissions: frozenset[Action]


class AccessState(TypedDict):
    """Mutable business data that flows through the graph."""

    action: Action
    resource: str
    allowed: bool
    response: str


def authorize(
    state: AccessState,
    runtime: Runtime[RequestContext],
) -> dict[str, bool]:
    """Read identity and permissions without copying them into State."""

    allowed = state["action"] in runtime.context.permissions
    return {"allowed": allowed}


def render_response(
    state: AccessState,
    runtime: Runtime[RequestContext],
) -> dict[str, str]:
    """Use the same immutable context in a later node."""

    context = runtime.context
    if context.locale == "zh-CN":
        decision = "允许" if state["allowed"] else "拒绝"
        response = (
            f"用户 {context.user_id} 的 {state['action']} 请求已{decision}: "
            f"{state['resource']}"
        )
    else:
        decision = "allowed" if state["allowed"] else "denied"
        response = (
            f"Request by {context.user_id} to {state['action']} "
            f"{state['resource']} was {decision}"
        )
    return {"response": response}


def build_graph():
    builder = StateGraph(AccessState, context_schema=RequestContext)
    builder.add_node("authorize", authorize)
    builder.add_node("render", render_response)
    builder.add_edge(START, "authorize")
    builder.add_edge("authorize", "render")
    builder.add_edge("render", END)
    return builder.compile()


def run(action: Action, resource: str, context: RequestContext) -> AccessState:
    return build_graph().invoke(
        {"action": action, "resource": resource},
        context=context,
    )


def self_test() -> None:
    viewer = RequestContext(
        user_id="alice",
        locale="zh-CN",
        permissions=frozenset({"read"}),
    )
    admin = RequestContext(
        user_id="root",
        locale="en-US",
        permissions=frozenset({"read", "delete"}),
    )

    denied = run("delete", "report-42", viewer)
    allowed = run("delete", "report-42", admin)

    assert denied["allowed"] is False
    assert denied["response"] == "用户 alice 的 delete 请求已拒绝: report-42"
    assert allowed["allowed"] is True
    assert allowed["response"] == ("Request by root to delete report-42 was allowed")

    # Context participates in computation but is not automatically added to State.
    assert "user_id" not in denied
    assert "permissions" not in denied
    assert set(denied) == {"action", "resource", "allowed", "response"}
    print("request_context self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("read", "delete"), default="delete")
    parser.add_argument("--resource", default="report-42")
    parser.add_argument("--user", default="alice")
    parser.add_argument("--locale", choices=("zh-CN", "en-US"), default="zh-CN")
    parser.add_argument("--admin", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    permissions: frozenset[Action]
    if args.admin:
        permissions = frozenset({"read", "delete"})
    else:
        permissions = frozenset({"read"})

    context = RequestContext(
        user_id=args.user,
        locale=args.locale,
        permissions=permissions,
    )
    result = run(args.action, args.resource, context)
    print(f"allowed: {result['allowed']}")
    print(f"response: {result['response']}")
    print(f"state keys: {sorted(result)}")


if __name__ == "__main__":
    main()
