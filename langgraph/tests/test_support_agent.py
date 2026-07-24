from __future__ import annotations

import sys
import unittest
from pathlib import Path

from langgraph.types import Command


LAB_ROOT = Path(__file__).resolve().parents[1] / "labs" / "01-support-agent"
sys.path.insert(0, str(LAB_ROOT))

from support_agent import (  # noqa: E402
    SupportContext,
    create_support_app,
    new_ticket,
    seed_profile,
)
from support_agent.schemas import merge_evidence  # noqa: E402


def config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def reviewer_context(
    owner_user_id: str,
    reviewer_id: str = "reviewer-1",
    *,
    tenant_id: str = "default",
) -> SupportContext:
    return SupportContext(
        user_id=owner_user_id,
        tenant_id=tenant_id,
        actor_id=reviewer_id,
        role="reviewer",
    )


def refund_namespace(
    user_id: str,
    tenant_id: str = "default",
) -> tuple[str, ...]:
    return ("tenants", tenant_id, "operations", user_id, "refunds")


def history_namespace(
    user_id: str,
    tenant_id: str = "default",
) -> tuple[str, ...]:
    return ("tenants", tenant_id, "users", user_id, "support_history")


class ReducerTests(unittest.TestCase):
    def test_evidence_reducer_replaces_same_logical_check(self) -> None:
        current = [{"ticket_id": "t-1", "check": "invoice", "finding": "old"}]
        updates = [
            {"ticket_id": "t-1", "check": "invoice", "finding": "new"},
            {"ticket_id": "t-1", "check": "payment", "finding": "ok"},
        ]
        result = merge_evidence(current, updates)
        self.assertEqual(2, len(result))
        by_check = {item["check"]: item["finding"] for item in result}
        self.assertEqual("new", by_check["invoice"])
        self.assertEqual("ok", by_check["payment"])


class SupportGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph, self.store, _ = create_support_app()

    def test_faq_uses_user_profile_and_native_subgraph(self) -> None:
        seed_profile(self.store, "u-faq", reply_style="concise", tier="gold")
        result = self.graph.invoke(
            new_ticket("faq-1", "How do I reset my password?"),
            config("faq-thread"),
            context=SupportContext(user_id="u-faq"),
        )
        self.assertEqual("faq", result["intent"])
        self.assertIn("style=concise", result["resolution"])
        self.assertEqual(1, result["trace"].count("profile:loaded"))
        self.assertIn("faq:search:password", result["trace"])

    def test_billing_fan_out_merges_three_findings(self) -> None:
        result = self.graph.invoke(
            new_ticket("bill-1", "There is a duplicate invoice charge"),
            config("bill-thread"),
            context=SupportContext(user_id="u-bill"),
        )
        evidence = [
            item for item in result["evidence"] if item["ticket_id"] == "bill-1"
        ]
        self.assertEqual(
            {"duplicates", "invoice", "payment"},
            {item["check"] for item in evidence},
        )
        self.assertEqual(3, len(evidence))
        self.assertIn("possible duplicate", result["resolution"])

    def test_high_refund_pauses_and_resumes_without_restarting(self) -> None:
        thread_config = config("refund-thread")
        context = SupportContext(user_id="u-refund")
        paused = self.graph.invoke(
            new_ticket("refund-1", "Please refund 1200"),
            thread_config,
            context=context,
        )
        self.assertIn("__interrupt__", paused)
        self.assertNotIn("response:composed", paused["trace"])

        finished = self.graph.invoke(
            Command(resume={"approved": True}),
            thread_config,
            context=reviewer_context("u-refund", "alice"),
        )
        self.assertTrue(finished["approval"]["approved"])
        self.assertIn("refund:executed", finished["trace"])
        self.assertEqual(1, finished["trace"].count("profile:loaded"))
        self.assertEqual(1, finished["trace"].count("intent:refund"))
        operation = self.store.get(
            refund_namespace("u-refund"),
            "refund-1",
        )
        self.assertEqual("rf-refund-1", operation.value["operation_id"])

    def test_high_refund_can_be_declined(self) -> None:
        thread_config = config("decline-thread")
        context = SupportContext(user_id="u-decline")
        paused = self.graph.invoke(
            new_ticket("refund-2", "Please refund 900"),
            thread_config,
            context=context,
        )
        self.assertIn("__interrupt__", paused)
        finished = self.graph.invoke(
            Command(resume={"approved": False}),
            thread_config,
            context=reviewer_context("u-decline", "bob"),
        )
        self.assertIn("declined", finished["resolution"])
        self.assertIsNone(
            self.store.get(
                refund_namespace("u-decline"),
                "refund-2",
            )
        )

    def test_same_thread_accumulates_messages_and_history(self) -> None:
        thread_config = config("conversation-1")
        context = SupportContext(user_id="u-conversation")
        first = self.graph.invoke(
            new_ticket("conversation-ticket-1", "What is the shipping time?"),
            thread_config,
            context=context,
        )
        second = self.graph.invoke(
            new_ticket("conversation-ticket-2", "How do I reset my password?"),
            thread_config,
            context=context,
        )
        self.assertEqual(2, len(first["messages"]))
        self.assertEqual(4, len(second["messages"]))
        self.assertEqual(2, second["trace"].count("faq:drafted"))
        self.assertEqual(1, second["faq_trace"].count("faq:drafted"))
        namespace = history_namespace("u-conversation")
        self.assertIsNotNone(self.store.get(namespace, "conversation-ticket-1"))
        self.assertIsNotNone(self.store.get(namespace, "conversation-ticket-2"))
        history = list(self.graph.get_state_history(thread_config, context=context))
        self.assertGreater(len(history), 2)

    def test_store_is_shared_across_threads_but_isolated_by_user(self) -> None:
        seed_profile(self.store, "user-a", reply_style="concise")
        result_a1 = self.graph.invoke(
            new_ticket("a-1", "What is the shipping time?"),
            config("thread-a1"),
            context=SupportContext(user_id="user-a"),
        )
        result_a2 = self.graph.invoke(
            new_ticket("a-2", "How do I reset my password?"),
            config("thread-a2"),
            context=SupportContext(user_id="user-a"),
        )
        result_b = self.graph.invoke(
            new_ticket("b-1", "How do I reset my password?"),
            config("thread-b1"),
            context=SupportContext(user_id="user-b"),
        )
        self.assertIn("style=concise", result_a1["resolution"])
        self.assertIn("style=concise", result_a2["resolution"])
        self.assertIn("style=detailed", result_b["resolution"])
        self.assertIsNone(
            self.store.get(
                history_namespace("user-b"),
                "a-1",
            )
        )

    def test_repeated_refund_reuses_idempotency_record(self) -> None:
        thread_config = config("repeat-refund-thread")
        context = SupportContext(user_id="u-repeat")

        self.graph.invoke(
            new_ticket("repeat-refund", "Please refund 100"),
            thread_config,
            context=context,
        )
        result = self.graph.invoke(
            new_ticket("repeat-refund", "Please refund 100"),
            thread_config,
            context=context,
        )
        self.assertIn("refund:deduplicated", result["trace"])
        self.assertIn("100.00", result["resolution"])
        items = self.store.search(refund_namespace("u-repeat"))
        self.assertEqual(1, len(items))

    def test_thread_owner_and_tenant_are_bound_to_checkpoint(self) -> None:
        thread_config = config("owned-thread")
        owner = SupportContext(user_id="owner", tenant_id="tenant-a")
        first = self.graph.invoke(
            new_ticket("owned-1", "How do I reset my password?"),
            thread_config,
            context=owner,
        )
        self.assertEqual("owner", first["owner_user_id"])
        self.assertEqual("tenant-a", first["owner_tenant_id"])

        with self.assertRaises(PermissionError):
            self.graph.invoke(
                new_ticket("stolen-1", "What is the shipping time?"),
                thread_config,
                context=SupportContext(user_id="attacker", tenant_id="tenant-a"),
            )
        with self.assertRaises(PermissionError):
            self.graph.invoke(
                new_ticket("stolen-2", "What is the shipping time?"),
                thread_config,
                context=SupportContext(user_id="owner", tenant_id="tenant-b"),
            )
        with self.assertRaises(PermissionError):
            self.graph.get_state(
                thread_config,
                context=SupportContext(user_id="attacker", tenant_id="tenant-a"),
            )
        with self.assertRaises(PermissionError):
            list(
                self.graph.get_state_history(
                    thread_config,
                    context=SupportContext(
                        user_id="owner",
                        tenant_id="tenant-b",
                    ),
                )
            )
        with self.assertRaises(AttributeError):
            self.graph.update_state(thread_config, {"owner_user_id": "attacker"})

        snapshot = self.graph.get_state(thread_config, context=owner)
        self.assertEqual(2, len(snapshot.values["messages"]))

    def test_resume_rejects_wrong_owner_then_allows_trusted_reviewer(self) -> None:
        thread_config = config("protected-resume")
        paused = self.graph.invoke(
            new_ticket("protected-refund", "Please refund 900"),
            thread_config,
            context=SupportContext(user_id="case-owner", tenant_id="tenant-a"),
        )
        self.assertIn("__interrupt__", paused)

        with self.assertRaises(PermissionError):
            self.graph.invoke(
                Command(resume={"approved": True}),
                thread_config,
                context=reviewer_context(
                    "other-owner",
                    "malicious-reviewer",
                    tenant_id="tenant-a",
                ),
            )

        finished = self.graph.invoke(
            Command(resume={"approved": False}),
            thread_config,
            context=reviewer_context(
                "case-owner",
                "trusted-reviewer",
                tenant_id="tenant-a",
            ),
        )
        self.assertEqual("trusted-reviewer", finished["approval"]["reviewer"])
        self.assertIn("declined", finished["resolution"])

    def test_customer_cannot_approve_and_decision_requires_exact_bool(self) -> None:
        thread_config = config("strict-approval")
        customer = SupportContext(user_id="strict-owner")
        self.graph.invoke(
            new_ticket("strict-refund", "Please refund 900"),
            thread_config,
            context=customer,
        )

        with self.assertRaises(PermissionError):
            self.graph.invoke(
                Command(resume={"approved": True}),
                thread_config,
                context=customer,
            )
        with self.assertRaises(ValueError):
            self.graph.invoke(
                Command(resume={"approved": "false"}),
                thread_config,
                context=reviewer_context("strict-owner", "real-reviewer"),
            )

        finished = self.graph.invoke(
            Command(resume={"approved": True, "reviewer": "forged-name"}),
            thread_config,
            context=reviewer_context("strict-owner", "real-reviewer"),
        )
        self.assertTrue(finished["approval"]["approved"])
        self.assertEqual("real-reviewer", finished["approval"]["reviewer"])

    def test_refund_policy_and_unsafe_amounts_never_execute(self) -> None:
        cases = {
            "missing": "Please refund this order",
            "comma": "Please refund 1,200",
            "negative": "Please refund -10",
            "multiple": "Please refund 10 instead of 20",
            "nonfinite": "Please refund NaN",
            "precision": "Please refund 10.001",
        }
        for suffix, query in cases.items():
            with self.subTest(case=suffix):
                result = self.graph.invoke(
                    new_ticket(f"unsafe-{suffix}", query),
                    config(f"unsafe-thread-{suffix}"),
                    context=SupportContext(user_id="unsafe-user"),
                )
                self.assertEqual("clarify", result["risk_level"])
                self.assertIn("not executed", result["resolution"])
                self.assertIsNone(
                    self.store.get(
                        refund_namespace("unsafe-user"),
                        f"unsafe-{suffix}",
                    )
                )

        policy = self.graph.invoke(
            new_ticket("policy", "What is your return policy?"),
            config("policy-thread"),
            context=SupportContext(user_id="unsafe-user"),
        )
        self.assertEqual("faq", policy["intent"])
        self.assertIn("30 days", policy["resolution"])

    def test_invalid_auto_refund_limit_requires_manual_review(self) -> None:
        for suffix, limit in (
            ("nan", float("nan")),
            ("inf", float("inf")),
            ("negative", -1.0),
        ):
            with self.subTest(limit=suffix):
                paused = self.graph.invoke(
                    new_ticket(f"limit-{suffix}", "Please refund 10"),
                    config(f"limit-thread-{suffix}"),
                    context=SupportContext(
                        user_id="limit-user",
                        auto_refund_limit=limit,
                    ),
                )
                self.assertEqual("high", paused["risk_level"])
                self.assertIn("__interrupt__", paused)
                self.assertIn("refund:limit-invalid", paused["trace"])

    def test_idempotency_key_rejects_different_refund_payload(self) -> None:
        thread_config = config("conflicting-refund-thread")
        context = SupportContext(user_id="u-conflict")
        self.graph.invoke(
            new_ticket("same-ticket", "Please refund 100"),
            thread_config,
            context=context,
        )
        result = self.graph.invoke(
            new_ticket("same-ticket", "Please refund 200"),
            thread_config,
            context=context,
        )
        self.assertIn("refund:idempotency-conflict", result["trace"])
        self.assertIn("100.00", result["resolution"])
        operation = self.store.get(refund_namespace("u-conflict"), "same-ticket")
        self.assertEqual("100", operation.value["amount"])

    def test_new_ticket_resets_scratch_state_and_preserves_turn_messages(self) -> None:
        thread_config = config("scratch-reset")
        self.graph.invoke(
            new_ticket("same-ticket", "Please refund 900"),
            thread_config,
            context=SupportContext(user_id="scratch-user"),
        )
        approved = self.graph.invoke(
            Command(resume={"approved": True}),
            thread_config,
            context=reviewer_context("scratch-user", "scratch-reviewer"),
        )
        self.assertIsNotNone(approved["approval"])

        second = self.graph.invoke(
            new_ticket("same-ticket", "How do I reset my password?"),
            thread_config,
            context=SupportContext(user_id="scratch-user"),
        )
        self.assertEqual("faq", second["intent"])
        self.assertIsNone(second["approval"])
        self.assertIsNone(second["amount"])
        self.assertIsNone(second["risk_level"])
        self.assertEqual(4, len(second["messages"]))
        self.assertEqual(4, len({message.id for message in second["messages"]}))


if __name__ == "__main__":
    unittest.main()
