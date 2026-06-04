# Module 06 Examples

These examples are designed for the current `aigc` environment. They use tiny
synthetic tensors and can run on CPU without downloading model weights or
datasets.

Run from this directory:

```bash
conda run -n aigc python lora_tiny_train.py --epochs 6 --rank 2
conda run -n aigc python quantization_sim.py --bits 4 --group-size 32
conda run -n aigc python sft_data_pipeline.py --max-length 80
conda run -n aigc python sft_data_pipeline.py --max-length 80 --pack
conda run -n aigc python dpo_loss_demo.py --beta 0.1
```

What each script covers:

| Script | Concept | What to check |
|---|---|---|
| `lora_tiny_train.py` | LoRA parameter freezing, low-rank update, merge | LoRA trains far fewer parameters and still fits a low-rank shift |
| `quantization_sim.py` | symmetric quantization and group quantization | per-group quantization usually has lower error than one global scale |
| `sft_data_pipeline.py` | Alpaca/ShareGPT conversion, chat template, assistant-only labels, packing | user tokens are masked with `-100`, assistant tokens are supervised |
| `dpo_loss_demo.py` | reward-model pair loss, DPO, IPO, GRPO advantages | chosen/rejected margins drive the DPO objective |

The production examples in the Markdown files use `transformers`, `peft`,
`trl`, `bitsandbytes`, and large checkpoints. Those workflows require model
downloads and suitable GPUs. These local scripts keep the same mechanics small
enough to verify quickly.
