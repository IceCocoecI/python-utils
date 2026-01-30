
# gpt‑oss‑20b
1
## Install vLLM from pip:
conda activate vllm



注意： pip install vllm会提示错误Unknown quantization method: mxfp4

https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html#gpt-oss-vllm-usage-guide

uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match
pip install -U openai


2
## Load and run the model:
vllm serve "openai/gpt-oss-20b"
下载到指定路径（如果目录里已经有模型，会直接使用）： vllm serve "openai/gpt-oss-20b" --download-dir "/media/cz/c813eb9d-3b03-3844-bdf7-9498585a13bf/workspace2/open_source_models"



3
## Call the server using curl:
curl -X POST "http://localhost:8000/v1/chat/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "openai/gpt-oss-20b",
		"messages": [
			{
				"role": "user",
				"content": "What is the capital of France?"
			}
		]
	}'


报错[Bug]: Unknown quantization method: mxfp4 #22276
