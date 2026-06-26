"""HuggingFace Transformers 快速入门示例。
演示：tokenizer / chat template / 生成 / 流式输出 / pipeline

默认离线运行：python transformers_quickstart.py
联网真实模型：python transformers_quickstart.py --model Qwen/Qwen2.5-0.5B-Instruct --real-model
建议 GPU 环境。CPU 也能跑，用 500M 以下小模型。
需要联网下载模型权重（Qwen2.5-0.5B 约 1GB）。
"""
from __future__ import annotations

import argparse

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    TextIteratorStreamer,
    pipeline,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


def build_tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "Hello": 4,
        ",": 5,
        "diffusion": 6,
        "model": 7,
        "!": 8,
        "hi": 9,
        "how": 10,
        "are": 11,
        "you": 12,
        "today": 13,
        "?": 14,
        "LoRA": 15,
        "is": 16,
        "a": 17,
        "low": 18,
        "rank": 19,
        "adapter": 20,
        "for": 21,
        "efficient": 22,
        "fine": 23,
        "tuning": 24,
        ".": 25,
        "AI": 26,
        "writes": 27,
        "small": 28,
        "poems": 29,
        "with": 30,
        "tokens": 31,
        "and": 32,
        "attention": 33,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )


def build_tiny_model(vocab_size: int) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=64,
        n_embd=64,
        n_layer=2,
        n_head=4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def load_real_model(model_name: str, local_files_only: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=local_files_only,
    )
    model.eval()
    return tokenizer, model


def format_prompt(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return f"{tokenizer.bos_token or ''} {user_text}".strip()


def demo_tokenizer(tok):
    print("== 1. Tokenizer 基础 ==")

    text = "Hello, diffusion model!"
    ids = tok.encode(text)
    print(f"text: {text}")
    print(f"tokens: {tok.convert_ids_to_tokens(ids)}")
    print(f"ids:    {ids}")
    print(f"decode: {tok.decode(ids)}")

    batch = tok(
        ["hi", "how are you today?"],
        padding=True,
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    print(f"batch input_ids shape: {batch['input_ids'].shape}")
    print(f"attention_mask shape:  {batch['attention_mask'].shape}")


def demo_chat_template(tok):
    print("\n== 2. Chat Template ==")
    messages = [
        {"role": "system", "content": "你是一个乐于助人的助手"},
        {"role": "user", "content": "用一句话介绍扩散模型"},
    ]
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        prompt = "<bos> system 你是一个乐于助人的助手 user 用一句话介绍扩散模型"
    print(prompt)


def demo_generate(tok, model):
    print("\n== 3. 生成 ==")
    prompt = format_prompt(tok, "LoRA is a low rank adapter for")
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("answer:", answer)


def demo_streaming(tok, model):
    print("\n== 4. 流式生成 ==")
    from threading import Thread

    prompt = format_prompt(tok, "AI writes small poems with")
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=16,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tok.eos_token_id,
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    for chunk in streamer:
        print(chunk, end="", flush=True)
    print()
    thread.join()


def demo_pipeline(real_model: bool):
    print("\n== 5. Pipeline：一行代码 ==")
    if real_model:
        sentiment = pipeline("sentiment-analysis")
        print(sentiment(["I love diffusion models!", "LLMs are mostly boring."]))
    else:
        tok = build_tiny_tokenizer()
        gen_cfg = GenerationConfig(
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        generator = pipeline(
            "text-generation",
            model=build_tiny_model(vocab_size=len(tok)),
            tokenizer=tok,
        )
        print(generator("Hello", generation_config=gen_cfg))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--real-model", action="store_true", help="Load a Hub model; requires network/cache.")
    parser.add_argument("--local-files-only", action="store_true", help="Use local HuggingFace cache only.")
    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(42)
    args = parse_args()
    if args.real_model:
        tokenizer, causal_lm = load_real_model(args.model, local_files_only=args.local_files_only)
    else:
        print("note: offline mode uses a randomly initialized tiny GPT-2; outputs only validate API behavior.")
        tokenizer = build_tiny_tokenizer()
        causal_lm = build_tiny_model(vocab_size=len(tokenizer))

    demo_tokenizer(tokenizer)
    demo_chat_template(tokenizer)
    demo_generate(tokenizer, causal_lm)
    demo_streaming(tokenizer, causal_lm)
    demo_pipeline(args.real_model)
