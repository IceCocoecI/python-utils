import os
# 指定下载目录
download_dir = "/media/cz/c813eb9d-3b03-3844-bdf7-9498585a13bf/workspace2/open_source_models"  # 本地路径
os.makedirs(download_dir, exist_ok=True)

# Use a pipeline as a high-level helper
from transformers import pipeline

# 在pipeline中指定下载目录
pipe = pipeline("text-generation", model="openai/gpt-oss-20b", cache_dir=download_dir)
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)

# Load model directly with specified directory
from transformers import AutoTokenizer, AutoModelForCausalLM

# 指定模型下载目录
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", cache_dir=download_dir)
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", cache_dir=download_dir)

messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
