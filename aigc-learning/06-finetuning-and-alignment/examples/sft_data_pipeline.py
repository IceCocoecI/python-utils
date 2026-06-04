"""Demonstrate SFT data conversion, chat templating, packing, and loss masks.

Run:
    conda run -n aigc python sft_data_pipeline.py --max-length 80
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


IGNORE_INDEX = -100


ALPACA_SAMPLES = [
    {
        "instruction": "把句子翻译成英文",
        "input": "今天天气很好。",
        "output": "The weather is nice today.",
    },
    {
        "instruction": "解释 LoRA 的核心思想",
        "input": "",
        "output": "LoRA freezes the base weight and trains a low-rank update matrix.",
    },
]

SHAREGPT_SAMPLE = {
    "conversations": [
        {"from": "human", "value": "如何减少 SFT 过拟合？"},
        {"from": "gpt", "value": "可以减少 epoch、降低学习率、混入通用数据，并监控 eval loss。"},
        {"from": "human", "value": "LoRA rank 应该怎么选？"},
        {"from": "gpt", "value": "简单风格适配可用 4-8，通用 SFT 常用 16-32。"},
    ]
}


@dataclass
class EncodedSample:
    text: str
    tokens: list[str]
    labels: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--pack", action="store_true")
    return parser.parse_args()


def alpaca_to_messages(sample: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    user_content = sample["instruction"]
    if sample.get("input"):
        user_content += "\n\n" + sample["input"]
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": sample["output"]},
        ]
    }


def sharegpt_to_messages(sample: dict[str, list[dict[str, str]]]) -> dict[str, list[dict[str, str]]]:
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    return {
        "messages": [
            {"role": role_map[turn["from"]], "content": turn["value"]}
            for turn in sample["conversations"]
        ]
    }


def apply_simple_chat_template(messages: list[dict[str, str]]) -> str:
    chunks = []
    for message in messages:
        chunks.append(f"<|{message['role']}|>\n{message['content']}<|end|>\n")
    return "".join(chunks)


def simple_tokenize(text: str) -> list[str]:
    """A tiny tokenizer that keeps chat markers as tokens and splits CJK text by char."""
    tokens: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith("<|", i):
            end = text.find("|>", i)
            if end != -1:
                tokens.append(text[i : end + 2])
                i = end + 2
                continue
        char = text[i]
        if char.isspace():
            i += 1
            continue
        if char.isascii() and (char.isalnum() or char in "_-"):
            j = i + 1
            while j < len(text) and text[j].isascii() and (text[j].isalnum() or text[j] in "_-"):
                j += 1
            tokens.append(text[i:j])
            i = j
            continue
        tokens.append(char)
        i += 1
    return tokens


def build_labels(tokens: list[str]) -> list[int]:
    labels = [IGNORE_INDEX] * len(tokens)
    in_assistant = False
    for i, token in enumerate(tokens):
        if token == "<|assistant|>":
            in_assistant = True
            continue
        if token == "<|end|>":
            in_assistant = False
            continue
        if in_assistant:
            labels[i] = i
    return labels


def encode_sample(messages: list[dict[str, str]], max_length: int) -> EncodedSample:
    text = apply_simple_chat_template(messages)
    tokens = simple_tokenize(text)[:max_length]
    labels = build_labels(tokens)
    return EncodedSample(text=text, tokens=tokens, labels=labels)


def pack_samples(samples: list[EncodedSample], max_length: int) -> EncodedSample:
    packed_tokens: list[str] = []
    packed_labels: list[int] = []
    for sample in samples:
        room = max_length - len(packed_tokens)
        if room <= 0:
            break
        packed_tokens.extend(sample.tokens[:room])
        packed_labels.extend(sample.labels[:room])
    return EncodedSample(text="<packed>", tokens=packed_tokens, labels=packed_labels)


def count_supervised_tokens(sample: EncodedSample) -> int:
    return sum(label != IGNORE_INDEX for label in sample.labels)


def main() -> None:
    args = parse_args()
    message_samples = [alpaca_to_messages(sample) for sample in ALPACA_SAMPLES]
    message_samples.append(sharegpt_to_messages(SHAREGPT_SAMPLE))

    encoded = [encode_sample(sample["messages"], args.max_length) for sample in message_samples]
    if args.pack:
        encoded = [pack_samples(encoded, args.max_length)]

    print("SFT data pipeline demo")
    print(f"num_sequences={len(encoded)}, max_length={args.max_length}, packing={args.pack}")
    for idx, sample in enumerate(encoded):
        supervised = count_supervised_tokens(sample)
        masked = len(sample.tokens) - supervised
        preview = " ".join(sample.tokens[:24])
        print(f"sample={idx} tokens={len(sample.tokens)} supervised_tokens={supervised} masked_tokens={masked}")
        print(f"preview={preview}")


if __name__ == "__main__":
    main()
