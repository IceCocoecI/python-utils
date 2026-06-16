# 实践优先路线

> 目标：用最短路径从“会看教程”推进到“能独立做出 AIGC 工程项目”。
> 这条路线优先跑通代码、形成作品，再反补理论。

---

## 总原则

- 每一阶段必须有可运行产出。
- 先跑离线 toy 示例，再接真实模型、真实数据、真实服务。
- 不追求一开始就学完所有分布式、CUDA、前沿架构；用到再深入。
- 每个项目都要留下运行命令、配置、结果截图或日志摘要。

---

## P0：环境与速查

**目标**

确认 `aigc` 环境可用，熟悉常用代码片段。

**阅读**

- [../CHEATSHEET.md](../CHEATSHEET.md)
- [../README.md](../README.md)

**建议命令**

```bash
conda run -n aigc python --version
conda run -n aigc python -c "import torch; print(torch.__version__)"
```

**验收**

- [ ] 能说明当前 Python、PyTorch、CUDA 是否可用。
- [ ] 知道忘记 API 时先查 `CHEATSHEET.md`。

---

## P1：Python、数据、Tensor 基础

**阅读**

- [../01-python-foundations/01-modern-python-basics.md](../01-python-foundations/01-modern-python-basics.md)
- [../01-python-foundations/02-advanced-features.md](../01-python-foundations/02-advanced-features.md)
- [../01-python-foundations/04-type-hints.md](../01-python-foundations/04-type-hints.md)
- [../03-data-and-scientific-computing/01-numpy-essentials.md](../03-data-and-scientific-computing/01-numpy-essentials.md)
- [../03-data-and-scientific-computing/02-einops-tutorial.md](../03-data-and-scientific-computing/02-einops-tutorial.md)
- [../02-deep-learning-libraries/01-pytorch-fundamentals.md](../02-deep-learning-libraries/01-pytorch-fundamentals.md)

**运行**

```bash
conda run -n aigc python aigc-learning/01-python-foundations/examples/decorators_demo.py
conda run -n aigc python aigc-learning/01-python-foundations/examples/generators_demo.py
conda run -n aigc python aigc-learning/01-python-foundations/examples/type_hints_demo.py
conda run -n aigc python aigc-learning/03-data-and-scientific-computing/examples/numpy_basics.py
conda run -n aigc python aigc-learning/03-data-and-scientific-computing/examples/einops_demo.py
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/pytorch_basics.py
```

**配套 Lab**

- [../labs/01-python-data-tensors/README.md](../labs/01-python-data-tensors/README.md)

**验收**

- [ ] 能用 `dataclass` 写配置对象。
- [ ] 能解释 generator 为什么适合流式数据。
- [ ] 能用 NumPy/einops/PyTorch 完成常见 shape 变换。
- [ ] 能解释 tensor 的 shape、dtype、device。

---

## P2：训练一个小模型

**阅读**

- [../02-deep-learning-libraries/02-pytorch-training-loop.md](../02-deep-learning-libraries/02-pytorch-training-loop.md)
- [../04-training-engineering/01-experiment-tracking.md](../04-training-engineering/01-experiment-tracking.md)
- [../04-training-engineering/02-config-management.md](../04-training-engineering/02-config-management.md)
- [../04-training-engineering/04-reproducibility.md](../04-training-engineering/04-reproducibility.md)

**运行**

```bash
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/mlp_mnist.py --synthetic --epochs 1 --max-train-batches 3 --max-val-batches 2 --workers 0
conda run -n aigc python aigc-learning/04-training-engineering/examples/wandb_train.py --epochs 1
conda run -n aigc python aigc-learning/04-training-engineering/examples/hydra_train.py --epochs 1
conda run -n aigc python aigc-learning/04-training-engineering/examples/reproducible_train.py --epochs 1
```

**配套 Lab**

- [../labs/02-train-mini-model/README.md](../labs/02-train-mini-model/README.md)

**验收**

- [ ] 能写出 train/eval/save 三段训练循环。
- [ ] 能保存模型权重和训练配置。
- [ ] 能解释 optimizer、scheduler、AMP、gradient clipping 的作用。
- [ ] 能复现实验结果，并说明可复现边界。

---

## P3：Transformer 与 HuggingFace 上手

**阅读**

- [../02-deep-learning-libraries/06-transformer-principles-overview.md](../02-deep-learning-libraries/06-transformer-principles-overview.md)
- [../02-deep-learning-libraries/05-transformer-from-scratch.md](../02-deep-learning-libraries/05-transformer-from-scratch.md)
- [../02-deep-learning-libraries/03-huggingface-transformers.md](../02-deep-learning-libraries/03-huggingface-transformers.md)
- [../02-deep-learning-libraries/04-huggingface-diffusers.md](../02-deep-learning-libraries/04-huggingface-diffusers.md)

**运行**

```bash
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/transformer_from_scratch.py
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/transformers_quickstart.py
conda run -n aigc python aigc-learning/02-deep-learning-libraries/examples/diffusers_quickstart.py --toy-steps 1 --toy-batch-size 2
```

**配套 Lab**

- [../labs/03-transformer-from-scratch/README.md](../labs/03-transformer-from-scratch/README.md)

**验收**

- [ ] 能画出 decoder-only Transformer 的数据流。
- [ ] 能解释 attention mask、RoPE、KV cache、采样策略。
- [ ] 能用 `AutoTokenizer` 和 `AutoModelForCausalLM` 做一次文本生成。

---

## P4：微调与对齐小闭环

**阅读**

- [../06-finetuning-and-alignment/01-peft-and-lora.md](../06-finetuning-and-alignment/01-peft-and-lora.md)
- [../06-finetuning-and-alignment/02-quantization.md](../06-finetuning-and-alignment/02-quantization.md)
- [../06-finetuning-and-alignment/03-sft-data-and-training.md](../06-finetuning-and-alignment/03-sft-data-and-training.md)
- [../06-finetuning-and-alignment/04-alignment-rlhf-dpo.md](../06-finetuning-and-alignment/04-alignment-rlhf-dpo.md)

**运行**

```bash
conda run -n aigc python aigc-learning/06-finetuning-and-alignment/examples/sft_data_pipeline.py
conda run -n aigc python aigc-learning/06-finetuning-and-alignment/examples/lora_tiny_train.py --epochs 1
conda run -n aigc python aigc-learning/06-finetuning-and-alignment/examples/quantization_sim.py
conda run -n aigc python aigc-learning/06-finetuning-and-alignment/examples/dpo_loss_demo.py
```

**配套 Lab**

- [../labs/04-lora-sft-mini-llm/README.md](../labs/04-lora-sft-mini-llm/README.md)

**验收**

- [ ] 能构造 instruction/chat 格式 SFT 数据。
- [ ] 能说明 LoRA 的 `r`、`alpha`、`target_modules` 怎么选。
- [ ] 能解释 SFT、RLHF、DPO 的差异。

---

## P5：推理与服务化

**阅读**

- [../07-inference-and-deployment/00-inference-and-deployment-theory.md](../07-inference-and-deployment/00-inference-and-deployment-theory.md)
- [../07-inference-and-deployment/01-llm-inference-engines.md](../07-inference-and-deployment/01-llm-inference-engines.md)
- [../07-inference-and-deployment/03-serving-frameworks.md](../07-inference-and-deployment/03-serving-frameworks.md)
- [../07-inference-and-deployment/04-demo-and-frontend.md](../07-inference-and-deployment/04-demo-and-frontend.md)

**运行**

```bash
conda run -n aigc python aigc-learning/07-inference-and-deployment/examples/kv_cache_and_batching.py
conda run -n aigc python aigc-learning/07-inference-and-deployment/examples/smoke_test.py
conda run -n aigc python aigc-learning/07-inference-and-deployment/examples/openai_compatible_toy_server.py
```

**配套 Lab**

- [../labs/05-openai-compatible-server/README.md](../labs/05-openai-compatible-server/README.md)

**验收**

- [ ] 能说明 TTFT、TPOT、吞吐、并发之间的关系。
- [ ] 能实现一个最小 OpenAI-compatible 接口。
- [ ] 能说清楚 vLLM 的 PagedAttention 解决了什么问题。

---

## P6：RAG 与 Agent 应用

**阅读**

- [../08-llm-applications/01-rag-fundamentals.md](../08-llm-applications/01-rag-fundamentals.md)
- [../08-llm-applications/02-vector-databases.md](../08-llm-applications/02-vector-databases.md)
- [../08-llm-applications/03-orchestration-frameworks.md](../08-llm-applications/03-orchestration-frameworks.md)
- [../08-llm-applications/04-agent-engineering.md](../08-llm-applications/04-agent-engineering.md)

**配套 Lab**

- [../labs/06-rag-mini-system/README.md](../labs/06-rag-mini-system/README.md)

**验收**

- [ ] 能完成文档切分、embedding、索引、检索、生成的最小链路。
- [ ] 能解释 chunk size、overlap、top-k、reranker 对效果的影响。
- [ ] 能识别 prompt injection 和工具调用风险。

---

## P7：规模化与性能优化

**阅读**

- [../05-distributed-training/01-distributed-basics-and-ddp.md](../05-distributed-training/01-distributed-basics-and-ddp.md)
- [../05-distributed-training/03-accelerate.md](../05-distributed-training/03-accelerate.md)
- [../05-distributed-training/05-parallelism-strategies.md](../05-distributed-training/05-parallelism-strategies.md)
- [../10-cuda-and-triton/03-performance-profiling.md](../10-cuda-and-triton/03-performance-profiling.md)

**运行**

```bash
conda run -n aigc python aigc-learning/05-distributed-training/examples/fsdp_memory_math.py
conda run -n aigc python aigc-learning/05-distributed-training/examples/parallelism_planner.py
```

**配套 Lab**

- [../labs/07-profiling-and-optimization/README.md](../labs/07-profiling-and-optimization/README.md)

**验收**

- [ ] 能估算训练和推理显存。
- [ ] 能判断一个瓶颈更像 CPU、I/O、GPU compute 还是 GPU memory。
- [ ] 能给出 batch size、并发、量化、KV cache 的调优思路。

