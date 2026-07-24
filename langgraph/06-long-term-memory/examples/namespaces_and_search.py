"""Organize long-term notes with namespaces, filters, and deletion.

Run:
    conda run -n langgraph python langgraph/06-long-term-memory/examples/namespaces_and_search.py
    conda run -n langgraph python langgraph/06-long-term-memory/examples/namespaces_and_search.py --self-test
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


Action = Literal["save", "list", "delete"]


@dataclass(frozen=True)
class MemoryContext:
    tenant_id: str
    user_id: str


class NoteState(TypedDict, total=False):
    action: Action
    note_id: str
    text: str
    tag: str
    result: str
    records: list[dict[str, str]]


def notes_namespace(context: MemoryContext) -> tuple[str, ...]:
    return (
        "tenants",
        context.tenant_id,
        "users",
        context.user_id,
        "notes",
    )


def manage_notes(
    state: NoteState,
    runtime: Runtime[MemoryContext],
) -> dict[str, object]:
    store = runtime.store
    if store is None:
        raise RuntimeError("this graph requires a Store")

    namespace = notes_namespace(runtime.context)
    action = state["action"]

    if action == "save":
        note_id = state.get("note_id", "")
        text = state.get("text", "")
        tag = state.get("tag", "general")
        if not note_id or not text:
            raise ValueError("save requires note_id and text")
        store.put(namespace, note_id, {"text": text, "tag": tag})
        return {
            "result": f"saved note {note_id}",
            "records": [],
        }

    if action == "delete":
        note_id = state.get("note_id", "")
        if not note_id:
            raise ValueError("delete requires note_id")
        store.delete(namespace, note_id)
        return {
            "result": f"deleted note {note_id}",
            "records": [],
        }

    tag = state.get("tag", "")
    item_filter = {"tag": tag} if tag else None
    items = store.search(namespace, filter=item_filter, limit=100)
    records = sorted(
        (
            {
                "note_id": item.key,
                "text": str(item.value["text"]),
                "tag": str(item.value["tag"]),
            }
            for item in items
        ),
        key=lambda record: record["note_id"],
    )
    return {
        "result": f"found {len(records)} note(s)",
        "records": records,
    }


def build_graph(store: InMemoryStore | None = None):
    memory_store = store if store is not None else InMemoryStore()
    builder = StateGraph(NoteState, context_schema=MemoryContext)
    builder.add_node("manage_notes", manage_notes)
    builder.add_edge(START, "manage_notes")
    builder.add_edge("manage_notes", END)
    return builder.compile(store=memory_store), memory_store


def self_test() -> None:
    graph, store = build_graph()
    alice = MemoryContext(tenant_id="acme", user_id="alice")
    bob = MemoryContext(tenant_id="acme", user_id="bob")

    for payload in (
        {
            "action": "save",
            "note_id": "n1",
            "text": "TypedDict describes State",
            "tag": "python",
        },
        {
            "action": "save",
            "note_id": "n2",
            "text": "Bake bread at 220C",
            "tag": "cooking",
        },
        {
            "action": "save",
            "note_id": "n3",
            "text": "Runtime carries Context",
            "tag": "python",
        },
    ):
        graph.invoke(payload, context=alice)

    python_notes = graph.invoke(
        {"action": "list", "tag": "python"},
        context=alice,
    )
    assert [note["note_id"] for note in python_notes["records"]] == ["n1", "n3"]

    # The same tenant does not remove user-level namespace isolation.
    bob_empty = graph.invoke({"action": "list"}, context=bob)
    assert bob_empty["records"] == []

    graph.invoke(
        {
            "action": "save",
            "note_id": "n1",
            "text": "Bob owns a different n1",
            "tag": "private",
        },
        context=bob,
    )
    alice_n1 = store.get(notes_namespace(alice), "n1")
    bob_n1 = store.get(notes_namespace(bob), "n1")
    assert alice_n1 is not None
    assert bob_n1 is not None
    assert alice_n1.value["text"] == "TypedDict describes State"
    assert bob_n1.value["text"] == "Bob owns a different n1"

    graph.invoke({"action": "delete", "note_id": "n1"}, context=alice)
    remaining = graph.invoke(
        {"action": "list", "tag": "python"},
        context=alice,
    )
    assert [note["note_id"] for note in remaining["records"]] == ["n3"]

    namespaces = set(store.list_namespaces(prefix=("tenants", "acme")))
    assert notes_namespace(alice) in namespaces
    assert notes_namespace(bob) in namespaces
    print("namespaces_and_search self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    graph, store = build_graph()
    context = MemoryContext(tenant_id="demo", user_id="alice")

    graph.invoke(
        {
            "action": "save",
            "note_id": "langgraph",
            "text": "Store is explicit long-term memory",
            "tag": "python",
        },
        context=context,
    )
    graph.invoke(
        {
            "action": "save",
            "note_id": "shopping",
            "text": "Buy coffee",
            "tag": "personal",
        },
        context=context,
    )

    result = graph.invoke(
        {"action": "list", "tag": "python"},
        context=context,
    )
    print(result["result"])
    for record in result["records"]:
        print(f"  {record['note_id']}: {record['text']} (tag={record['tag']})")

    print("namespaces:")
    for namespace in store.list_namespaces(prefix=("tenants", "demo")):
        print(f"  {namespace}")


if __name__ == "__main__":
    main()
