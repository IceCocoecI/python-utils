"""Gradio and Streamlit demo templates backed by the toy chat function.

Run self-test:
    conda run -n aigc python demo_apps.py --mode self-test

Run Gradio after installing optional deps:
    conda run -n aigc python demo_apps.py --mode gradio

Run Streamlit after installing optional deps:
    conda run -n aigc streamlit run demo_apps.py -- --mode streamlit
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator

from common import chunk_words, toy_chat_completion


SYSTEM_PROMPT = "You are a concise inference deployment assistant."


def normalize_history(history: list[dict] | list[tuple] | None) -> list[dict[str, str]]:
    if not history:
        return []
    normalized: list[dict[str, str]] = []
    for item in history:
        if isinstance(item, dict):
            role = str(item.get("role", "assistant"))
            content = str(item.get("content", ""))
            normalized.append({"role": role, "content": content})
        elif isinstance(item, tuple) and len(item) == 2:
            user, assistant = item
            normalized.append({"role": "user", "content": str(user)})
            normalized.append({"role": "assistant", "content": str(assistant)})
    return normalized


def chat_once(message: str, history: list[dict] | list[tuple] | None = None) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(normalize_history(history))
    messages.append({"role": "user", "content": message})
    return toy_chat_completion(messages, max_tokens=64)


def chat_stream(message: str, history: list[dict] | list[tuple] | None = None) -> Iterator[str]:
    full = ""
    for chunk in chunk_words(chat_once(message, history), chunk_size=2):
        full += chunk
        yield full


def launch_gradio() -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit("Install optional deps first: pip install gradio") from exc

    demo = gr.ChatInterface(
        fn=chat_stream,
        title="Inference Deployment Toy Demo",
        description="A deterministic local chat demo for testing streaming UI mechanics.",
        examples=[
            "Explain KV Cache",
            "Why does continuous batching improve throughput?",
            "How does SSE streaming help LLM UX?",
        ],
        type="messages",
    )
    demo.launch(server_name="127.0.0.1", server_port=7860)


def launch_streamlit() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Install optional deps first: pip install streamlit") from exc

    st.set_page_config(page_title="Inference Deployment Toy Demo", layout="centered")
    st.title("Inference Deployment Toy Demo")
    st.caption("Deterministic local backend for validating chat UI and streaming behavior.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about inference deployment"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            response = st.write_stream(chat_stream(prompt, st.session_state.messages[:-1]))
        st.session_state.messages.append({"role": "assistant", "content": response})


def self_test() -> None:
    answer = chat_once("Explain KV Cache", [])
    chunks = list(chat_stream("Explain SSE streaming", []))
    assert "KV Cache" in answer
    assert chunks and chunks[-1]
    print("demo_apps self-test passed")
    print(answer)
    print(f"stream_chunks={len(chunks)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["self-test", "gradio", "streamlit"], default="self-test")
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "self-test":
        self_test()
    elif args.mode == "gradio":
        launch_gradio()
    elif args.mode == "streamlit":
        launch_streamlit()


if __name__ == "__main__":
    main()
