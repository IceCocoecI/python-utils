# Lab 07：Profiling 与性能优化

> 目标：形成工程化性能分析方法，知道瓶颈来自哪里，而不是凭感觉调参数。

---

## 前置知识

- [../../07-inference-and-deployment/00-inference-and-deployment-theory.md](../../07-inference-and-deployment/00-inference-and-deployment-theory.md)
- [../../10-cuda-and-triton/01-gpu-architecture-and-cuda-basics.md](../../10-cuda-and-triton/01-gpu-architecture-and-cuda-basics.md)
- [../../10-cuda-and-triton/03-performance-profiling.md](../../10-cuda-and-triton/03-performance-profiling.md)
- [../../05-distributed-training/05-parallelism-strategies.md](../../05-distributed-training/05-parallelism-strategies.md)

---

## 运行脚本

无 GPU 也可以先运行显存和并行策略推导：

```bash
conda run -n aigc python aigc-learning/05-distributed-training/examples/fsdp_memory_math.py
conda run -n aigc python aigc-learning/05-distributed-training/examples/parallelism_planner.py
conda run -n aigc python aigc-learning/07-inference-and-deployment/examples/kv_cache_and_batching.py
```

再跑一个 `torch.profiler` 最小训练 profiling：

```bash
conda run -n aigc python aigc-learning/10-cuda-and-triton/examples/torch_profiler_demo.py --steps 5
```

默认使用 CPU，并隐藏部分 PyTorch profiler 在 CPU-only 环境下的底层设备探测日志。有 GPU 时可以加 `--device cuda --trace`，再用 TensorBoard 查看 trace；需要内存统计时再加 `--profile-memory`。

---

## 任务

1. 选择一个目标脚本：训练循环、Transformer 生成、toy serving 均可。
2. 记录 baseline：
   - batch size
   - sequence length
   - latency
   - tokens/s 或 samples/s
   - CPU/GPU 利用率
   - 显存占用
3. 用 profiler 采集一次 trace。
4. 判断瓶颈类型：
   - Python/CPU 调度
   - 数据读取/I/O
   - GPU compute
   - GPU memory bandwidth
   - 通信
5. 尝试一个优化：
   - 增大 batch 或并发
   - 使用 AMP/bfloat16
   - 使用 `torch.compile`
   - 减少同步点
   - 量化
   - 缓存中间结果
6. 对比优化前后指标，写结论。

---

## 验收标准

- [ ] 有 baseline 数据，而不是只写主观感受。
- [ ] 能指出至少一个具体瓶颈证据。
- [ ] 能说明优化为什么可能有效。
- [ ] 能区分吞吐优化和延迟优化。
- [ ] 能写出没有继续优化的原因：硬件、代码复杂度、收益不足或质量损失。

---

## 延伸挑战

- 对同一脚本分别测 batch size 1、4、16。
- 对比 fp32、fp16、bf16。
- 对比 eager、`torch.compile`。
- 对比 CPU-only 和 GPU。
- 写一份容量规划表：模型大小、上下文长度、并发、显存。
