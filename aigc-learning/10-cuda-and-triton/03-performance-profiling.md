# 03 · 性能分析与优化

> "Premature optimization is the root of all evil." —— Donald Knuth
> 但 **不 profiling 的优化** 比 premature optimization 更蠢。
> 先测量，再优化。

---

## 1. 为什么需要 Profiling？

你以为的瓶颈和实际的瓶颈，往往不是同一个东西。

```
你以为的：                      实际的：
"模型太大了，前向传播太慢"        数据加载占了 60% 时间
"GPU 不够快"                    GPU 利用率只有 30%，CPU 在拖后腿
"Attention 是瓶颈"              LayerNorm 被调用 100 次，总耗时更多
"需要更多 GPU"                  单 GPU 的算力都没吃满
```

**Profiling 的核心目标**：找到时间花在哪里，内存用在哪里。

```
┌─────────────────────────────────────────────────────────┐
│                  Profiling 工作流                         │
│                                                          │
│   1. 先让代码正确跑起来                                    │
│   2. Profiling：发现瓶颈在哪                               │
│   3. 分析：是 CPU-bound? GPU-bound? Memory-bound?         │
│   4. 优化：针对瓶颈做改动                                   │
│   5. 再 Profiling：验证优化效果                             │
│   6. 重复 2-5 直到满意                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 2. PyTorch Profiler

PyTorch 内置的 profiler 是最容易上手的工具，不需要安装额外软件。

### 2.1 基本用法

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity


model = MyModel().cuda()
inputs = torch.randn(32, 3, 224, 224, device="cuda")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,       # 记录 tensor 形状
    profile_memory=True,      # 记录内存分配
    with_stack=True,          # 记录 Python 调用栈
) as prof:
    with record_function("model_inference"):
        output = model(inputs)

# 打印表格：按 CUDA 时间排序
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

输出示例：

```
-------  ------  ------  ------  ------  ------  ------
Name     CPU %   CPU     CUDA %  CUDA    #Calls  Input
         total   total   total   total           Shapes
-------  ------  ------  ------  ------  ------  ------
aten::mm  5.2%   12ms    45.3%   89ms    24      [32,768],[768,768]
aten::  
softmax   2.1%   5ms     18.7%   37ms    12      [32,12,128,128]
aten::  
layer_    
norm      3.4%   8ms     12.1%   24ms    24      [32,128,768]
...
-------  ------  ------  ------  ------  ------  ------
```

### 2.2 记录特定代码段

```python
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("data_loading"):
        batch = next(iter(dataloader))

    with record_function("forward"):
        output = model(batch["input_ids"].cuda())

    with record_function("loss"):
        loss = criterion(output, batch["labels"].cuda())

    with record_function("backward"):
        loss.backward()

    with record_function("optimizer_step"):
        optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 2.3 生成 Chrome Trace

Chrome Trace 提供交互式的时间线视图，是分析 CPU/GPU overlap 的最佳方式。

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 5:
            break
        train_step(model, batch)

prof.export_chrome_trace("trace.json")
# 在 Chrome 浏览器打开 chrome://tracing，加载 trace.json
# 或使用 https://ui.perfetto.dev/（推荐，功能更强）
```

```
Chrome Trace 时间线示意：

CPU: ──[data_load]──[to_cuda]──[forward]──────────[backward]────[optim]──
GPU:                            ──[matmul]─[softmax]─ ──[matmul_bwd]────
                                            ↑ GPU idle（CPU 在搬数据）
```

### 2.4 训练循环 Profiling（schedule 模式）

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,       # 跳过前 1 步（warmup）
        warmup=1,     # 1 步 warmup（不记录）
        active=3,     # 记录 3 步
        repeat=2,     # 重复 2 轮
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/profiler"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 20:
            break
        train_step(model, batch)
        prof.step()  # 通知 profiler 一步结束

# 启动 TensorBoard 查看结果
# tensorboard --logdir ./log/profiler
```

TensorBoard PyTorch Profiler Plugin 提供：
- **Overview**：总体性能概览
- **Operator View**：按算子统计
- **Trace View**：时间线
- **Memory View**：内存变化
- **Module View**：按 nn.Module 聚合

---

## 3. GPU 时间测量：`torch.cuda.Event`

`time.time()` 测 GPU 操作是**不准的**——CUDA 操作是异步的。

```python
# ❌ 错误的计时方式
import time
start = time.time()
output = model(x)  # 这只是发射了 GPU 命令，没等执行完
elapsed = time.time() - start  # 测的是 launch 时间，不是执行时间

# ✅ 正确的 GPU 计时
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
output = model(x)
end_event.record()

torch.cuda.synchronize()  # 等 GPU 完成
elapsed_ms = start_event.elapsed_time(end_event)
print(f"GPU time: {elapsed_ms:.2f} ms")
```

**做 benchmark 的标准流程**：

```python
def benchmark_fn(fn, *args, warmup=10, repeats=100):
    # Warmup
    for _ in range(warmup):
        fn(*args)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeats):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = sorted(times)
    median = times[len(times) // 2]
    print(f"Median: {median:.2f} ms, "
          f"Min: {times[0]:.2f} ms, "
          f"Max: {times[-1]:.2f} ms")
    return median
```

---

## 4. 内存分析

### 4.1 基本内存统计

```python
torch.cuda.reset_peak_memory_stats()

model = MyModel().cuda()
output = model(inputs)
loss = criterion(output, labels)
loss.backward()

print(f"当前分配: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"峰值分配: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"当前预留: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"峰值预留: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
```

**allocated vs reserved**：
- `allocated`：PyTorch 实际使用的显存
- `reserved`：PyTorch 从 CUDA 申请的显存（包含内存池中未使用的部分）
- `nvidia-smi` 显示的是 `reserved`，所以通常比实际使用量大

### 4.2 详细内存统计

```python
stats = torch.cuda.memory_stats()
for key, value in stats.items():
    if "peak" in key or "allocated" in key:
        print(f"{key}: {value}")
```

### 4.3 内存快照（Memory Snapshot）

```python
torch.cuda.memory._record_memory_history(max_entries=100000)

# 运行你的代码
train_step(model, batch)

torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)

# 可视化：https://pytorch.org/memory_viz
# 上传 memory_snapshot.pickle 即可看到内存分配的时间线和调用栈
```

### 4.4 追踪显存峰值来源

```python
import traceback

class MemoryTracker:
    def __init__(self):
        self.peak = 0
        self.peak_stack = ""

    def check(self, label=""):
        current = torch.cuda.memory_allocated()
        if current > self.peak:
            self.peak = current
            self.peak_stack = f"{label}\n{''.join(traceback.format_stack()[-3:])}"

    def report(self):
        print(f"Peak memory: {self.peak / 1e9:.2f} GB")
        print(f"Location:\n{self.peak_stack}")

tracker = MemoryTracker()
output = model(x); tracker.check("after forward")
loss = loss_fn(output, y); tracker.check("after loss")
loss.backward(); tracker.check("after backward")
tracker.report()
```

---

## 5. NVIDIA Nsight Systems

Nsight Systems 是 NVIDIA 官方的**系统级** profiler，提供 CPU + GPU + 内存 + 通信的全局时间线。

### 5.1 安装与基本使用

```bash
# 安装（Ubuntu）
sudo apt install nsight-systems

# 或从 NVIDIA 官网下载：
# https://developer.nvidia.com/nsight-systems

# 命令行采集
nsys profile -o my_trace python train.py

# 常用参数
nsys profile \
    -o my_trace \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --stats=true \
    python train.py
```

### 5.2 在 PyTorch 中添加 NVTX 标记

NVTX（NVIDIA Tools Extension）让你在 Nsight Systems 时间线上看到 Python 代码对应的 GPU 区域。

```python
import torch.cuda.nvtx as nvtx

def train_step(model, batch):
    nvtx.range_push("data_transfer")
    x = batch["input_ids"].cuda()
    y = batch["labels"].cuda()
    nvtx.range_pop()

    nvtx.range_push("forward")
    output = model(x)
    nvtx.range_pop()

    nvtx.range_push("loss")
    loss = criterion(output, y)
    nvtx.range_pop()

    nvtx.range_push("backward")
    loss.backward()
    nvtx.range_pop()

    nvtx.range_push("optimizer")
    optimizer.step()
    optimizer.zero_grad()
    nvtx.range_pop()

    return loss.item()
```

也可以用装饰器：

```python
@torch.cuda.nvtx.range("forward_pass")
def forward(model, x):
    return model(x)
```

### 5.3 Nsight Systems 时间线分析要点

```
Nsight Systems 时间线（从上到下）：

┌── CPU Thread 0 (Python main) ──────────────────────────┐
│ [data_load][to_cuda][forward]   [backward]    [optim]  │
├── CUDA API ────────────────────────────────────────────┤
│ [cuLaunchKernel][cuMemcpy]  [cuLaunchKernel]...        │
├── GPU Stream 0 ────────────────────────────────────────┤
│          [mm][softmax][ln] [mm_bwd][softmax_bwd]       │
├── GPU Memory ──────────────────────────────────────────┤
│     ↗ alloc    ↗ alloc          ↘ free    ↘ free       │
└────────────────────────────────────────────────────────┘

关注点：
1. CPU 和 GPU 之间的 gap = GPU idle time
2. 连续的小 kernel = kernel launch overhead 大
3. 大块的 cuMemcpy = CPU↔GPU 数据传输瓶颈
```

---

## 6. NVIDIA Nsight Compute

Nsight Compute 是**单 kernel 级别**的深度分析工具，告诉你一个 kernel 为什么慢。

### 6.1 基本使用

```bash
# 采集所有 kernel 的完整指标（慢，但信息最全）
ncu -o my_kernel_report python my_script.py

# 只采集特定 kernel
ncu --kernel-name "my_kernel_name" -o report python my_script.py

# 只采集第 5-10 个 kernel（跳过初始化）
ncu --launch-skip 5 --launch-count 5 -o report python my_script.py

# 采集 roofline 分析需要的指标
ncu --set roofline -o report python my_script.py
```

### 6.2 关键指标解读

```
Nsight Compute 报告的关键指标：

┌──────────────────────────────────────────────────┐
│ Compute (SM) Throughput:  45%                    │
│ Memory Throughput:        87%   ← Memory-bound!  │
│ Achieved Occupancy:       62%                    │
│ Registers per Thread:     48                     │
│ Shared Memory:            16 KB / 48 KB          │
│ Warp Execution Efficiency: 95%                   │
└──────────────────────────────────────────────────┘

解读：
- Memory Throughput >> Compute Throughput → Memory-bound
  优化方向：减少内存访问（fusion）、改善访问模式（coalescing）

- Compute Throughput >> Memory Throughput → Compute-bound
  优化方向：用 Tensor Core（FP16/BF16）、减少计算量

- Achieved Occupancy 低 → 没有足够的 warp 隐藏延迟
  优化方向：减少每线程寄存器用量、减少共享内存用量、调整 block size
```

---

## 7. 识别瓶颈类型

### 7.1 CPU-bound vs GPU-bound

```python
# 判断方法：对比 CPU 和 GPU 时间
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    train_step(model, batch)

stats = prof.key_averages()
cpu_total = sum(s.cpu_time_total for s in stats)
gpu_total = sum(s.cuda_time_total for s in stats)

print(f"CPU total: {cpu_total / 1e6:.2f} s")
print(f"GPU total: {gpu_total / 1e6:.2f} s")

# CPU total >> GPU total → CPU-bound（数据加载、预处理）
# GPU total >> CPU total → GPU-bound（模型计算）
# 两者接近 → 需要看 overlap 情况
```

### 7.2 数据加载瓶颈

```python
# 用 profiler 检测数据加载是否是瓶颈
import time

data_times = []
compute_times = []

for step, batch in enumerate(dataloader):
    if step >= 20:
        break

    t0 = time.perf_counter()
    x = batch["input_ids"].cuda(non_blocking=True)
    y = batch["labels"].cuda(non_blocking=True)
    data_times.append(time.perf_counter() - t0)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = train_step(model, x, y)
    torch.cuda.synchronize()
    compute_times.append(time.perf_counter() - t0)

avg_data = sum(data_times) / len(data_times)
avg_compute = sum(compute_times) / len(compute_times)
print(f"Data loading: {avg_data*1000:.1f} ms")
print(f"Compute:      {avg_compute*1000:.1f} ms")

# 如果 data loading 时间 ≈ compute 时间 → 数据加载是瓶颈
# 解决：增加 num_workers、pin_memory=True、预处理缓存
```

---

## 8. LLM 推理 Profiling

LLM 推理有独特的 profiling 需求：**prefill** 和 **decode** 阶段性能特征完全不同。

```python
# LLM 推理的两个阶段
#
# Prefill（预填充）：处理整个 prompt
#   - 一次处理 N 个 token（N = prompt length）
#   - Compute-bound（大矩阵乘法）
#   - 时间 ∝ prompt_length
#
# Decode（解码）：逐 token 生成
#   - 每次只处理 1 个 token
#   - Memory-bound（从 HBM 读取整个模型权重，只做 1 次计算）
#   - 时间 ∝ output_length
#   - 优化关键：KV cache 管理

def profile_llm_inference(model, tokenizer, prompt, max_new_tokens=100):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    # Profiling prefill
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        outputs = model(input_ids)
    end.record()
    torch.cuda.synchronize()
    prefill_ms = start.elapsed_time(end)

    # Profiling decode (simplified)
    start.record()
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    decode_ms = total_ms - prefill_ms

    n_prompt = input_ids.shape[1]
    n_generated = generated.shape[1] - n_prompt

    print(f"Prefill:  {prefill_ms:.1f} ms ({n_prompt} tokens, "
          f"{n_prompt / prefill_ms * 1000:.0f} tok/s)")
    print(f"Decode:   {decode_ms:.1f} ms ({n_generated} tokens, "
          f"{n_generated / decode_ms * 1000:.0f} tok/s)")
    print(f"Total:    {total_ms:.1f} ms")
```

---

## 9. 常见优化手段（Profiling 之后该做什么）

### 9.1 Operator Fusion（算子融合）

```python
# 问题：每个 PyTorch 操作单独 launch 一个 kernel，中间结果写回 HBM
# x → [LayerNorm kernel] → HBM → [Dropout kernel] → HBM → [Add kernel] → HBM
#       3 次 HBM 读 + 3 次 HBM 写

# 融合后：
# x → [Fused LN+Dropout+Add kernel] → HBM
#       1 次 HBM 读 + 1 次 HBM 写

# 方案 1：torch.compile（自动融合）
model = torch.compile(model)

# 方案 2：手写 Triton kernel（手动融合）
# 见上一章的 fused softmax 示例

# 方案 3：使用已有的 fused 实现
# flash-attn, apex.fused_layer_norm, xformers
```

### 9.2 内存格式优化

```python
# channels_last 内存格式对 CNN 有显著加速
# NCHW (default) → NHWC (channels_last)
model = model.to(memory_format=torch.channels_last)
inputs = inputs.to(memory_format=torch.channels_last)

# 对 Transformer 模型，确保使用 contiguous 的 tensor
# 非 contiguous tensor 会导致额外的内存拷贝
x = x.contiguous()
```

### 9.3 `torch.compile`

```python
# torch.compile 自动做很多优化：
# - 算子融合
# - 内存规划
# - CUDA Graph
# - 自动调用 Triton 生成 fused kernel

model = torch.compile(model, mode="reduce-overhead")
# mode 选择：
#   "default"         - 平衡编译时间和性能
#   "reduce-overhead" - 最大化运行时性能（用 CUDA Graph）
#   "max-autotune"    - 最大化性能（autotune + Triton）

# ⚠️ 第一次调用会编译（几秒到几分钟），后续调用才快
# ⚠️ 动态 shape 会导致频繁重编译——设置 dynamic=True 或固定 shape
```

### 9.4 Mixed Precision

```python
# AMP (Automatic Mixed Precision) 利用 Tensor Core
# FP32 → FP16/BF16: 算力提升 2-8x，显存减半

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(x)
    loss = criterion(output, y)

scaler = torch.amp.GradScaler()  # FP16 需要 loss scaling，BF16 不需要
```

### 9.5 减少 CPU-GPU 同步

```python
# ❌ 每步都同步（GPU 等 CPU）
for step in range(1000):
    loss = train_step(model, batch)
    print(f"Step {step}: loss = {loss.item()}")  # .item() 会同步！

# ✅ 间隔打印
for step in range(1000):
    loss = train_step(model, batch)
    if step % 100 == 0:
        print(f"Step {step}: loss = {loss.item()}")
```

---

## 10. 完整 Profiling 工作流示例

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler


def full_profiling_workflow(model, dataloader, optimizer, device="cuda"):
    """一个完整的 profiling 工作流示例"""

    # === Step 1: 基础计时 ===
    model = model.to(device)
    batch = next(iter(dataloader))
    x = batch["input_ids"].to(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()

    start.record()
    with torch.no_grad():
        _ = model(x)
    end.record()
    torch.cuda.synchronize()
    print(f"Forward pass: {start.elapsed_time(end):.2f} ms")

    # === Step 2: 详细 profiling ===
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        optimizer.zero_grad()
        output = model(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()

    # 按 GPU 时间排序，看最耗时的算子
    print("\n=== Top 10 CUDA Operators ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # 按 CPU 时间排序，看 CPU 瓶颈
    print("\n=== Top 10 CPU Operators ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # 导出 Chrome trace
    prof.export_chrome_trace("profiling_trace.json")
    print("\nChrome trace saved to profiling_trace.json")
    print("Open in: https://ui.perfetto.dev/")

    # === Step 3: 内存分析 ===
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    optimizer.zero_grad()
    output = model(x)
    mem_after_fwd = torch.cuda.memory_allocated() / 1e9

    loss = output.mean()
    loss.backward()
    mem_after_bwd = torch.cuda.memory_allocated() / 1e9

    optimizer.step()
    mem_after_optim = torch.cuda.memory_allocated() / 1e9
    mem_peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n=== Memory Usage ===")
    print(f"After forward:   {mem_after_fwd:.2f} GB")
    print(f"After backward:  {mem_after_bwd:.2f} GB")
    print(f"After optimizer: {mem_after_optim:.2f} GB")
    print(f"Peak:            {mem_peak:.2f} GB")

    # === Step 4: 多步 profiling (for TensorBoard) ===
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=2, active=3),
        on_trace_ready=tensorboard_trace_handler("./tb_profiler"),
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for step, batch in enumerate(dataloader):
            if step >= 10:
                break
            x = batch["input_ids"].to(device)
            optimizer.zero_grad()
            loss = model(x).mean()
            loss.backward()
            optimizer.step()
            prof.step()

    print("\nTensorBoard profiler data saved to ./tb_profiler")
    print("Run: tensorboard --logdir ./tb_profiler")
```

---

## 11. 常见坑

### 11.1 CUDA 异步导致计时不准

```python
# ❌ time.time() 测 GPU 操作
import time
t0 = time.time()
y = model(x)
print(f"{time.time() - t0:.3f}s")  # 只测了 kernel launch 时间

# ✅ 用 CUDA events 或 torch.cuda.synchronize()
torch.cuda.synchronize()
t0 = time.time()
y = model(x)
torch.cuda.synchronize()
print(f"{time.time() - t0:.3f}s")

# 最准确的方式还是 CUDA events（见 Section 3）
```

### 11.2 Warmup 不够

```python
# GPU 第一次执行某个 kernel 时会有额外开销（JIT 编译、缓存预热）
# 前几次运行时间会偏长

# ❌ 直接测
times = [benchmark_one_step() for _ in range(100)]

# ✅ 先 warmup
for _ in range(10):  # warmup
    benchmark_one_step()
torch.cuda.synchronize()
times = [benchmark_one_step() for _ in range(100)]  # 正式测
```

### 11.3 Profiler 本身的开销

```python
# profiler 会影响性能（尤其是 with_stack=True）
# 不要把 profiler 打开的时间当作真实性能

# 正确做法：
# 1. 先不开 profiler，用 CUDA events 测基线性能
# 2. 再开 profiler 分析瓶颈分布（相对比例是准的）
# 3. 优化后再关 profiler 验证绝对性能
```

### 11.4 .item() 和 print 导致同步

```python
# .item() 会触发 CPU-GPU 同步
# 在训练循环中频繁调用会严重拖慢速度

# ❌ 每步都 .item()
for step in range(10000):
    loss = train_step()
    losses.append(loss.item())  # 每步同步

# ✅ 在 GPU tensor 上累积，偶尔 .item()
running_loss = torch.tensor(0.0, device="cuda")
for step in range(10000):
    loss = train_step()
    running_loss += loss.detach()
    if step % 100 == 0:
        avg_loss = (running_loss / 100).item()
        running_loss.zero_()
        print(f"Step {step}: {avg_loss:.4f}")
```

### 11.5 nvidia-smi 显示的显存不等于实际使用

```python
# nvidia-smi 显示的是 reserved memory（PyTorch 缓存池的大小）
# 实际使用量要看 torch.cuda.memory_allocated()

# 如果 nvidia-smi 显示 16GB 但 allocated 只有 4GB
# → PyTorch 预留了 12GB 的缓存池（碎片化或历史分配）

# 释放缓存池：
torch.cuda.empty_cache()
# ⚠️ 这不会减少 allocated，只会减少 reserved
```

### 11.6 Nsight 权限问题

```bash
# Nsight Systems/Compute 可能需要特殊权限
# 如果遇到 "ERR_NVGPUCTRPERM" 错误：

# 方案 1：用 sudo
sudo nsys profile python train.py

# 方案 2：设置 perf_event 权限
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'

# 方案 3：用 Docker 时加 --cap-add=SYS_ADMIN
docker run --gpus all --cap-add=SYS_ADMIN ...
```

---

## 小结

| 工具 | 层级 | 适用场景 |
|---|---|---|
| `torch.profiler` | 算子级 | 快速定位最耗时的 PyTorch 操作 |
| `torch.cuda.Event` | 代码段 | 精确测量特定代码段的 GPU 时间 |
| `torch.cuda.memory_*` | 内存 | 跟踪显存分配和峰值 |
| Chrome Trace / Perfetto | 时间线 | 分析 CPU-GPU overlap 和 idle |
| Nsight Systems | 系统级 | 全局时间线，找 GPU idle 原因 |
| Nsight Compute | Kernel 级 | 单个 kernel 的深度分析（roofline） |

**一句话总结**：**不要猜——先 profile，再优化，再 profile。** 90% 的"优化"尝试如果没有先 profiling，都是在浪费时间。

下一节学习自定义算子——当你找到了瓶颈，如何用 C++/CUDA 扩展 PyTorch 来解决它。
