"""HuggingFace Transformers 快速入门示例。
演示：tokenizer / chat template / 生成 / 流式输出 / pipeline

运行：python transformers_quickstart.py
建议 GPU 环境。CPU 也能跑，用 500M 以下小模型。
需要联网下载模型权重（Qwen2.5-0.5B 约 1GB）。
"""
from __future__ import annotations

import torch


def demo_tokenizer():
    print("== 1. Tokenizer 基础 ==")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

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


def demo_chat_template():
    print("\n== 2. Chat Template ==")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    messages = [
        {"role": "system", "content": "你是一个乐于助人的助手"},
        {"role": "user", "content": "用一句话介绍扩散模型"},
    ]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print(prompt)


def demo_generate():
    print("\n== 3. 生成 ==")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    messages = [
        {"role": "user", "content": "什么是 LoRA？用三句话说明"},
    ]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("answer:", answer)


def demo_streaming():
    print("\n== 4. 流式生成 ==")
    from threading import Thread
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    messages = [{"role": "user", "content": "写一个关于 AI 的 4 行小诗"}]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=128,
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


def demo_pipeline():
    print("\n== 5. Pipeline：一行代码 ==")
    from transformers import pipeline

    sentiment = pipeline("sentiment-analysis")
    print(sentiment(["I love diffusion models!", "LLMs are mostly boring."]))


if __name__ == "__main__":
    demo_tokenizer()
    demo_chat_template()
    demo_generate()
    demo_streaming()
    demo_pipeline()
