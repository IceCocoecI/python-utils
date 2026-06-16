# 完整线性路线

> 这条路线适合希望从基础到高级完整走一遍的人。
> 如果你的目标更明确，优先选择 [practice-track](./practice-track.md)、[theory-track](./theory-track.md) 或岗位路线。

---

## 使用方式

- 每个阶段先读模块 README，再读具体章节。
- 每读完一个阶段，完成对应 [labs](../labs/README.md) 和 [assessments](../assessments/README.md)。
- 如果时间有限，可以把 `05`、`09`、`10` 中与你岗位无关的部分后置。

---

## 第 1 阶段：打牢基础

预计 2 到 3 周。

1. [../01-python-foundations/01-modern-python-basics.md](../01-python-foundations/01-modern-python-basics.md)
2. [../03-data-and-scientific-computing/01-numpy-essentials.md](../03-data-and-scientific-computing/01-numpy-essentials.md)
3. [../01-python-foundations/02-advanced-features.md](../01-python-foundations/02-advanced-features.md)
4. [../01-python-foundations/04-type-hints.md](../01-python-foundations/04-type-hints.md)
5. [../01-python-foundations/05-engineering-best-practices.md](../01-python-foundations/05-engineering-best-practices.md)

配套 Lab：

- [../labs/01-python-data-tensors/README.md](../labs/01-python-data-tensors/README.md)

---

## 第 2 阶段：深度学习实战

预计 3 到 4 周。

1. [../02-deep-learning-libraries/00-deep-learning-theory.md](../02-deep-learning-libraries/00-deep-learning-theory.md)
2. [../02-deep-learning-libraries/01-pytorch-fundamentals.md](../02-deep-learning-libraries/01-pytorch-fundamentals.md)
3. [../02-deep-learning-libraries/02-pytorch-training-loop.md](../02-deep-learning-libraries/02-pytorch-training-loop.md)
4. [../03-data-and-scientific-computing/02-einops-tutorial.md](../03-data-and-scientific-computing/02-einops-tutorial.md)
5. [../03-data-and-scientific-computing/03-image-processing.md](../03-data-and-scientific-computing/03-image-processing.md)
6. [../03-data-and-scientific-computing/04-data-formats-and-pipelines.md](../03-data-and-scientific-computing/04-data-formats-and-pipelines.md)
7. [../02-deep-learning-libraries/05-transformer-from-scratch.md](../02-deep-learning-libraries/05-transformer-from-scratch.md)

配套 Lab：

- [../labs/02-train-mini-model/README.md](../labs/02-train-mini-model/README.md)
- [../labs/03-transformer-from-scratch/README.md](../labs/03-transformer-from-scratch/README.md)

---

## 第 3 阶段：AIGC 框架应用

预计 2 到 3 周。

1. [../02-deep-learning-libraries/03-huggingface-transformers.md](../02-deep-learning-libraries/03-huggingface-transformers.md)
2. [../02-deep-learning-libraries/04-huggingface-diffusers.md](../02-deep-learning-libraries/04-huggingface-diffusers.md)
3. [../01-python-foundations/03-async-programming.md](../01-python-foundations/03-async-programming.md)

---

## 第 4 阶段：训练工程化与分布式

预计 3 到 4 周。

1. [../04-training-engineering/00-training-engineering-theory.md](../04-training-engineering/00-training-engineering-theory.md)
2. [../04-training-engineering/01-experiment-tracking.md](../04-training-engineering/01-experiment-tracking.md)
3. [../04-training-engineering/02-config-management.md](../04-training-engineering/02-config-management.md)
4. [../04-training-engineering/03-hyperparameter-search.md](../04-training-engineering/03-hyperparameter-search.md)
5. [../04-training-engineering/04-reproducibility.md](../04-training-engineering/04-reproducibility.md)
6. [../05-distributed-training/00-distributed-training-theory.md](../05-distributed-training/00-distributed-training-theory.md)
7. [../05-distributed-training/01-distributed-basics-and-ddp.md](../05-distributed-training/01-distributed-basics-and-ddp.md)
8. [../05-distributed-training/02-fsdp.md](../05-distributed-training/02-fsdp.md)
9. [../05-distributed-training/03-accelerate.md](../05-distributed-training/03-accelerate.md)
10. [../05-distributed-training/04-deepspeed.md](../05-distributed-training/04-deepspeed.md)
11. [../05-distributed-training/05-parallelism-strategies.md](../05-distributed-training/05-parallelism-strategies.md)

---

## 第 5 阶段：微调、对齐与推理部署

预计 4 周。

1. [../06-finetuning-and-alignment/00-finetuning-and-alignment-theory.md](../06-finetuning-and-alignment/00-finetuning-and-alignment-theory.md)
2. [../06-finetuning-and-alignment/01-peft-and-lora.md](../06-finetuning-and-alignment/01-peft-and-lora.md)
3. [../06-finetuning-and-alignment/02-quantization.md](../06-finetuning-and-alignment/02-quantization.md)
4. [../06-finetuning-and-alignment/03-sft-data-and-training.md](../06-finetuning-and-alignment/03-sft-data-and-training.md)
5. [../06-finetuning-and-alignment/04-alignment-rlhf-dpo.md](../06-finetuning-and-alignment/04-alignment-rlhf-dpo.md)
6. [../07-inference-and-deployment/00-inference-and-deployment-theory.md](../07-inference-and-deployment/00-inference-and-deployment-theory.md)
7. [../07-inference-and-deployment/01-llm-inference-engines.md](../07-inference-and-deployment/01-llm-inference-engines.md)
8. [../07-inference-and-deployment/02-diffusion-acceleration.md](../07-inference-and-deployment/02-diffusion-acceleration.md)
9. [../07-inference-and-deployment/03-serving-frameworks.md](../07-inference-and-deployment/03-serving-frameworks.md)
10. [../07-inference-and-deployment/04-demo-and-frontend.md](../07-inference-and-deployment/04-demo-and-frontend.md)

配套 Lab：

- [../labs/04-lora-sft-mini-llm/README.md](../labs/04-lora-sft-mini-llm/README.md)
- [../labs/05-openai-compatible-server/README.md](../labs/05-openai-compatible-server/README.md)

---

## 第 6 阶段：LLM 应用与前沿

预计 4 周。

1. [../08-llm-applications/01-rag-fundamentals.md](../08-llm-applications/01-rag-fundamentals.md)
2. [../08-llm-applications/02-vector-databases.md](../08-llm-applications/02-vector-databases.md)
3. [../08-llm-applications/03-orchestration-frameworks.md](../08-llm-applications/03-orchestration-frameworks.md)
4. [../08-llm-applications/04-agent-engineering.md](../08-llm-applications/04-agent-engineering.md)
5. [../09-frontier-models/01-llm-architectures.md](../09-frontier-models/01-llm-architectures.md)
6. [../09-frontier-models/02-image-generation.md](../09-frontier-models/02-image-generation.md)
7. [../09-frontier-models/03-multimodal-models.md](../09-frontier-models/03-multimodal-models.md)
8. [../09-frontier-models/04-video-generation.md](../09-frontier-models/04-video-generation.md)
9. [../09-frontier-models/05-speech-and-audio.md](../09-frontier-models/05-speech-and-audio.md)

配套 Lab：

- [../labs/06-rag-mini-system/README.md](../labs/06-rag-mini-system/README.md)

---

## 第 7 阶段：工程深水区

持续学习。

1. [../10-cuda-and-triton/01-gpu-architecture-and-cuda-basics.md](../10-cuda-and-triton/01-gpu-architecture-and-cuda-basics.md)
2. [../10-cuda-and-triton/02-triton-programming.md](../10-cuda-and-triton/02-triton-programming.md)
3. [../10-cuda-and-triton/03-performance-profiling.md](../10-cuda-and-triton/03-performance-profiling.md)
4. [../10-cuda-and-triton/04-custom-operators-and-extensions.md](../10-cuda-and-triton/04-custom-operators-and-extensions.md)

配套 Lab：

- [../labs/07-profiling-and-optimization/README.md](../labs/07-profiling-and-optimization/README.md)

