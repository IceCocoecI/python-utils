# 04 · 自定义算子与 PyTorch 扩展

> 当 PyTorch 原生算子不够用——要么缺了你需要的操作，要么性能不够——你需要自己写。
> 从 pybind11 到 CUDA extension，再到 `torch.library` 和 FlexAttention，本节覆盖全链路。

---

## 1. 什么时候需要自定义算子？

| 场景 | 例子 | 推荐方案 |
|---|---|---|
| 多个 elementwise 操作想融合 | LayerNorm + Dropout + Add | Triton kernel / torch.compile |
| 需要特殊的 attention 变体 | Sliding window / block-sparse | FlexAttention / Triton |
| PyTorch 没有的算子 | PagedAttention、AWQ dequant | CUDA extension |
| C++ 代码需要调 Python | 推理引擎中的自定义采样 | pybind11 |
| 性能关键路径 | 推理热点 kernel | CUDA extension + torch.library |

**决策流程**：

```
需要自定义操作？
  ├─ 能用 torch.compile 解决吗？ → 是 → 用 torch.compile（最简单）
  ├─ 能用 Triton 写吗？ → 是 → 写 Triton kernel（Python，快）
  ├─ 需要极致性能或 Triton 搞不定？ → 写 CUDA C++ extension
  └─ 需要和 torch.compile 配合？ → 用 torch.library 注册
```

---

## 2. pybind11：C++ 绑定到 Python

pybind11 是把 C++ 代码暴露给 Python 的标准工具。PyTorch 的 C++ extension 就建立在它之上。

```cpp
// hello.cpp
#include <pybind11/pybind11.h>

int add(int a, int b) {
    return a + b;
}

PYBIND11_MODULE(hello, m) {
    m.doc() = "A simple example";
    m.def("add", &add, "Add two numbers",
          pybind11::arg("a"), pybind11::arg("b"));
}
```

```python
# 编译方式 1：用 setup.py
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(
    ext_modules=[Pybind11Extension("hello", ["hello.cpp"])],
)
# python setup.py install

# 编译方式 2：直接 JIT（PyTorch 提供）
from torch.utils.cpp_extension import load
hello = load(name="hello", sources=["hello.cpp"])
print(hello.add(1, 2))  # 3
```

---

## 3. `torch.utils.cpp_extension`：PyTorch 的 C++/CUDA 扩展

### 3.1 C++ Extension（纯 CPU）

```cpp
// my_ops.cpp
#include <torch/extension.h>

torch::Tensor relu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    return torch::clamp_min(input, 0);
}

torch::Tensor relu_backward(torch::Tensor grad_output, torch::Tensor input) {
    auto mask = input > 0;
    return grad_output * mask;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_forward", &relu_forward, "ReLU forward (CPU)");
    m.def("relu_backward", &relu_backward, "ReLU backward (CPU)");
}
```

```python
from torch.utils.cpp_extension import load

my_ops = load(name="my_ops", sources=["my_ops.cpp"], verbose=True)

x = torch.randn(10, requires_grad=True)
y = my_ops.relu_forward(x)
```

### 3.2 CUDA Extension

```cpp
// fused_bias_relu.cu
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bias_relu_kernel(
    const float* input,
    const float* bias,
    float* output,
    int N, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C) {
        int c = idx % C;
        float val = input[idx] + bias[c];
        output[idx] = val > 0 ? val : 0;
    }
}

torch::Tensor fused_bias_relu_cuda(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(input);
    int N = input.size(0);
    int C = input.size(1);
    int total = N * C;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_bias_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C
    );

    return output;
}
```

```cpp
// bindings.cpp
#include <torch/extension.h>

torch::Tensor fused_bias_relu_cuda(torch::Tensor input, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_bias_relu", &fused_bias_relu_cuda, "Fused bias + ReLU (CUDA)");
}
```

```python
from torch.utils.cpp_extension import load

fused_ops = load(
    name="fused_ops",
    sources=["bindings.cpp", "fused_bias_relu.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)

x = torch.randn(32, 768, device="cuda")
bias = torch.randn(768, device="cuda")
y = fused_ops.fused_bias_relu(x, bias)

# 验证
y_ref = torch.relu(x + bias)
assert torch.allclose(y, y_ref, atol=1e-5)
```

### 3.3 打包成 pip 可安装的包

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="my_cuda_ops",
    ext_modules=[
        CUDAExtension(
            name="my_cuda_ops",
            sources=[
                "csrc/bindings.cpp",
                "csrc/fused_bias_relu.cu",
                "csrc/fused_layernorm.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math",
                         "-gencode=arch=compute_80,code=sm_80",  # A100
                         "-gencode=arch=compute_90,code=sm_90"], # H100
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```bash
pip install .
# 或开发模式
pip install -e .
```

项目结构：

```
my_cuda_ops/
├── setup.py
├── csrc/
│   ├── bindings.cpp
│   ├── fused_bias_relu.cu
│   └── fused_layernorm.cu
├── my_cuda_ops/
│   ├── __init__.py         # Python wrapper
│   └── functional.py       # 高层 API
└── tests/
    └── test_ops.py
```

---

## 4. `torch.autograd.Function`：自定义反向传播

要让自定义算子支持训练，需要同时实现 forward 和 backward。

```python
import torch
from torch.autograd import Function


class FusedBiasReLU(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        output = fused_ops.fused_bias_relu(input, bias)
        ctx.save_for_backward(input, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, bias = ctx.saved_tensors
        activated = input + bias.unsqueeze(0)
        mask = (activated > 0).float()

        grad_input = grad_output * mask
        grad_bias = (grad_output * mask).sum(dim=0)
        return grad_input, grad_bias


fused_bias_relu = FusedBiasReLU.apply

# 使用
x = torch.randn(32, 768, device="cuda", requires_grad=True)
bias = torch.randn(768, device="cuda", requires_grad=True)
y = fused_bias_relu(x, bias)
y.sum().backward()  # grad 正确传播
```

### 数值梯度检查

```python
from torch.autograd import gradcheck

x = torch.randn(4, 8, device="cuda", dtype=torch.float64, requires_grad=True)
bias = torch.randn(8, device="cuda", dtype=torch.float64, requires_grad=True)

# gradcheck 会对比解析梯度和数值梯度
test = gradcheck(FusedBiasReLU.apply, (x, bias), eps=1e-6, atol=1e-4)
print(f"Gradient check passed: {test}")
```

---

## 5. 实战：Fused RMSNorm

RMSNorm 是 LLaMA/Qwen 等现代 LLM 的标准归一化层。朴素 PyTorch 实现需要多次 HBM 读写。

### 5.1 PyTorch 参考实现

```python
def rms_norm_pytorch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight
```

### 5.2 CUDA Fused 实现

```cpp
// fused_rmsnorm.cu
#include <torch/extension.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void rms_norm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    int hidden_size,
    float eps
) {
    int row = blockIdx.x;
    const T* x = input + row * hidden_size;
    T* out = output + row * hidden_size;

    extern __shared__ float smem[];

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(x[i]);
        thread_sum += val * val;
    }

    // Block reduce sum
    smem[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(smem[0] / hidden_size + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(x[i]);
        out[i] = static_cast<T>(val * rms * static_cast<float>(weight[i]));
    }
}

torch::Tensor rms_norm_cuda(torch::Tensor input, torch::Tensor weight, float eps) {
    auto output = torch::empty_like(input);
    int batch = input.numel() / input.size(-1);
    int hidden = input.size(-1);
    int threads = std::min(hidden, 1024);
    int smem_size = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm", [&] {
        rms_norm_kernel<scalar_t><<<batch, threads, smem_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            hidden, eps
        );
    });

    return output;
}
```

**`AT_DISPATCH_FLOATING_TYPES_AND_HALF`** 是 PyTorch 的类型分派宏：根据输入 tensor 的 dtype 自动实例化对应的模板。

---

## 6. `torch.library`：注册自定义算子（torch.compile 兼容）

从 PyTorch 2.4+ 开始，推荐用 `torch.library` 注册自定义算子，这样可以和 `torch.compile` 无缝配合。

### 6.1 为什么需要 `torch.library`？

```python
# 问题：用 pybind11 注册的算子对 torch.compile 不可见
# torch.compile 看到一个"黑盒"函数，无法优化

# torch.library 告诉 PyTorch：
# 1. 这个算子的 schema 是什么（输入输出类型）
# 2. 它的行为特征（是否有副作用、是否是 pure function）
# 3. 如何在不同后端执行（CPU/CUDA/Meta）
# 4. 如何计算梯度（autograd formula）
```

### 6.2 注册自定义算子

```python
import torch
from torch.library import Library, impl

# 创建命名空间
mylib = Library("myops", "DEF")

# 定义算子 schema
mylib.define("rms_norm(Tensor input, Tensor weight, float eps=1e-6) -> Tensor")

# CPU 实现
@impl(mylib, "rms_norm", "CPU")
def rms_norm_cpu(input, weight, eps=1e-6):
    variance = input.pow(2).mean(dim=-1, keepdim=True)
    return input * torch.rsqrt(variance + eps) * weight

# CUDA 实现
@impl(mylib, "rms_norm", "CUDA")
def rms_norm_cuda_impl(input, weight, eps=1e-6):
    return rms_norm_cuda(input, weight, eps)  # 调用 CUDA kernel

# Meta 实现（shape inference，torch.compile 需要）
@impl(mylib, "rms_norm", "Meta")
def rms_norm_meta(input, weight, eps=1e-6):
    return torch.empty_like(input)
```

### 6.3 使用 `torch.library.custom_op`（更简洁的 API）

```python
@torch.library.custom_op("myops::rms_norm", mutates_args=())
def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = input.pow(2).mean(dim=-1, keepdim=True)
    return input * torch.rsqrt(variance + eps) * weight

# 注册 Meta/FakeTensor 实现
@rms_norm.register_fake
def rms_norm_fake(input, weight, eps=1e-6):
    return torch.empty_like(input)

# 注册 autograd
def rms_norm_backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    eps = ctx.eps
    variance = input.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    x_hat = input * rstd

    grad_weight = (grad_output * x_hat).sum(dim=tuple(range(grad_output.ndim - 1)))
    grad_input = grad_output * weight * rstd  # 简化版，实际更复杂

    return grad_input, grad_weight, None

def rms_norm_setup_context(ctx, inputs, output):
    input, weight, eps = inputs
    ctx.save_for_backward(input, weight)
    ctx.eps = eps

rms_norm.register_autograd(rms_norm_backward, setup_context=rms_norm_setup_context)
```

### 6.4 与 `torch.compile` 配合

```python
# 注册了 torch.library 的算子可以被 torch.compile 正确处理
@torch.compile
def model_forward(x, weight):
    x = rms_norm(x, weight)   # torch.compile 知道这个算子
    x = torch.relu(x)         # 可能被 fuse 到一起
    return x

# torch.compile 能为这个计算图生成优化的代码
# 如果有 Triton/CUDA 实现，会自动选择最优的
```

---

## 7. FlexAttention：PyTorch 的可组合 Attention API

FlexAttention（PyTorch 2.5+）是一个革命性的 API，让你用 Python 定义 attention 变体，自动编译成高效的 fused kernel。

### 7.1 为什么需要 FlexAttention？

```
Attention 变体爆炸：
  - Causal Attention
  - Sliding Window Attention
  - Block-sparse Attention
  - Prefix Attention (prefill + decode)
  - Document Attention (多文档拼接)
  - ALiBi / RoPE (位置编码影响 attention)
  - Soft-capping (Gemma 2)

每个变体都需要一个独立的 CUDA kernel？
FlexAttention 说：不，给我一个 Python 函数描述你的变体，我来编译。
```

### 7.2 `score_mod`：修改注意力分数

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# score_mod 接收 (score, batch, head, q_idx, kv_idx) 返回修改后的 score
# 在 softmax 之前应用

# Causal Attention
def causal_mask(score, batch, head, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, float("-inf"))

# ALiBi 位置编码
def alibi_bias(score, batch, head, q_idx, kv_idx):
    slope = 1.0 / (2 ** (head + 1))
    bias = -slope * torch.abs(q_idx - kv_idx)
    return score + bias

# Soft-capping (Gemma 2 style)
def soft_cap(score, batch, head, q_idx, kv_idx):
    cap = 50.0
    return cap * torch.tanh(score / cap)

# Sliding window attention
def sliding_window(score, batch, head, q_idx, kv_idx):
    window_size = 1024
    return torch.where(
        (q_idx - kv_idx).abs() <= window_size,
        score,
        float("-inf"),
    )

# 使用
output = flex_attention(query, key, value, score_mod=causal_mask)
```

### 7.3 `block_mask`：稀疏 attention

`block_mask` 告诉 FlexAttention 哪些 Q-K block 完全不需要计算（全是 `-inf`），避免无效计算。

```python
# 定义 mask function
def causal_mask_fn(batch, head, q_idx, kv_idx):
    return q_idx >= kv_idx

# 创建 block mask（编译期计算，不在运行时执行）
block_mask = create_block_mask(
    causal_mask_fn,
    B=batch_size,
    H=num_heads,
    Q_LEN=seq_len,
    KV_LEN=seq_len,
    _compile=True,
)

# block_mask 会自动跳过全 -inf 的 block
output = flex_attention(query, key, value, block_mask=block_mask)
```

### 7.4 组合使用

```python
# 实际场景：causal + sliding window + soft capping
def my_attention_mod(score, batch, head, q_idx, kv_idx):
    score = soft_cap(score, batch, head, q_idx, kv_idx)
    return score

def my_mask_fn(batch, head, q_idx, kv_idx):
    causal = q_idx >= kv_idx
    window = (q_idx - kv_idx) <= 4096
    return causal & window

block_mask = create_block_mask(my_mask_fn, B=B, H=H, Q_LEN=S, KV_LEN=S)
output = flex_attention(Q, K, V, score_mod=my_attention_mod, block_mask=block_mask)
```

### 7.5 FlexAttention vs 手写 Kernel

| 维度 | FlexAttention | 手写 Triton/CUDA |
|---|---|---|
| 开发时间 | 分钟 | 天-周 |
| 性能 | ~90-100% FlashAttention | 100%（如果写对了） |
| 灵活性 | 用 Python 组合 | 每个变体重写 |
| 维护成本 | PyTorch 团队维护 | 自己维护 |
| 梯度计算 | 自动 | 手动实现 backward |
| torch.compile | 原生支持 | 需要 torch.library |

**推荐策略**：优先用 FlexAttention，性能不够再手写。

---

## 8. 测试自定义算子

### 8.1 数值正确性

```python
import torch
import pytest


def test_fused_bias_relu():
    torch.manual_seed(42)
    x = torch.randn(32, 768, device="cuda")
    bias = torch.randn(768, device="cuda")

    y_custom = fused_ops.fused_bias_relu(x, bias)
    y_ref = torch.relu(x + bias)

    assert torch.allclose(y_custom, y_ref, atol=1e-5), \
        f"Max diff: {(y_custom - y_ref).abs().max()}"


def test_rms_norm():
    torch.manual_seed(42)
    for shape in [(1, 768), (32, 768), (4, 128, 4096)]:
        x = torch.randn(*shape, device="cuda")
        weight = torch.randn(shape[-1], device="cuda")

        y_custom = rms_norm(x, weight)
        y_ref = rms_norm_pytorch(x, weight)

        assert torch.allclose(y_custom, y_ref, atol=1e-4), \
            f"Failed for shape {shape}, max diff: {(y_custom - y_ref).abs().max()}"


def test_rms_norm_dtypes():
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = torch.randn(32, 768, device="cuda", dtype=dtype)
        weight = torch.randn(768, device="cuda", dtype=dtype)
        y = rms_norm(x, weight)
        assert y.dtype == dtype
        assert y.shape == x.shape
```

### 8.2 梯度检查

```python
def test_gradient():
    x = torch.randn(4, 16, device="cuda", dtype=torch.float64, requires_grad=True)
    weight = torch.randn(16, device="cuda", dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda x, w: rms_norm(x, w),
        (x, weight),
        eps=1e-6, atol=1e-4, rtol=1e-3,
    )
```

### 8.3 边界条件

```python
def test_edge_cases():
    weight = torch.randn(768, device="cuda")

    x_zeros = torch.zeros(1, 768, device="cuda")
    y = rms_norm(x_zeros, weight)
    assert torch.isfinite(y).all(), "Should handle zero input"

    x_large = torch.randn(1, 768, device="cuda") * 1e6
    y = rms_norm(x_large, weight)
    assert torch.isfinite(y).all(), "Should handle large values"

    x_tiny = torch.randn(1, 768, device="cuda") * 1e-8
    y = rms_norm(x_tiny, weight)
    assert torch.isfinite(y).all(), "Should handle tiny values"
```

---

## 9. 实战：把扩展集成到模型中

```python
import torch
import torch.nn as nn


class FusedRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self._use_fused = self._check_cuda_extension()

    def _check_cuda_extension(self) -> bool:
        try:
            import my_cuda_ops
            self._fused_fn = my_cuda_ops.rms_norm
            return True
        except ImportError:
            return False

    def _pytorch_fallback(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_fused and x.is_cuda:
            return self._fused_fn(x, self.weight, self.eps)
        return self._pytorch_fallback(x)


# 在模型中使用
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm1 = FusedRMSNorm(hidden_size)
        self.norm2 = FusedRMSNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=12)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
```

---

## 10. 常见坑

### 10.1 忘记 contiguous 检查

```cpp
// ❌ 非 contiguous tensor 的 data_ptr 不连续，直接当连续数组访问会出错
float* data = input.data_ptr<float>();  // 如果 input 是 transpose 的结果...

// ✅ 始终检查或强制 contiguous
TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
// 或者
auto input_c = input.contiguous();
```

### 10.2 dtype 不匹配

```cpp
// ❌ 假设 float32，但传入了 float16
float* data = input.data_ptr<float>();  // float16 tensor → 崩溃

// ✅ 使用 AT_DISPATCH 宏
AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "my_kernel", [&] {
    scalar_t* data = input.data_ptr<scalar_t>();
    // scalar_t 会根据输入自动推导为 float, double, 或 at::Half
});
```

### 10.3 JIT 编译缓存问题

```python
# torch.utils.cpp_extension.load() 有编译缓存
# 修改 .cu 文件后可能仍然使用旧的缓存

# 解决方案 1：清除缓存
import shutil
shutil.rmtree(torch.utils.cpp_extension._get_build_directory("my_ops", verbose=False))

# 解决方案 2：改名字强制重新编译
my_ops = load(name="my_ops_v2", sources=[...])

# 解决方案 3：使用 setup.py build（生产环境推荐）
```

### 10.4 忘记注册 Meta/FakeTensor 实现

```python
# 如果自定义算子没有 Meta 实现，torch.compile 会报错
# "NotImplementedError: Could not run 'myops::rms_norm' with arguments
#  from the 'Meta' backend"

# 解决：注册 Meta 实现（只做 shape inference，不做计算）
@impl(mylib, "rms_norm", "Meta")
def rms_norm_meta(input, weight, eps=1e-6):
    return torch.empty_like(input)
```

### 10.5 CUDA architecture 不匹配

```bash
# 编译时指定的 compute capability 必须匹配运行时的 GPU
# A100 = sm_80, H100 = sm_90, RTX 4090 = sm_89

# ❌ 在 H100 上编译 -gencode=arch=compute_80,code=sm_80
# 可能会错过 H100 特有的优化

# ✅ 编译多个 arch
extra_compile_args = {
    "nvcc": [
        "-gencode=arch=compute_80,code=sm_80",  # A100
        "-gencode=arch=compute_89,code=sm_89",  # RTX 4090
        "-gencode=arch=compute_90,code=sm_90",  # H100
    ]
}

# 或者用 PTX（兼容未来架构，但可能略慢）
# "-gencode=arch=compute_80,code=compute_80"
```

### 10.6 backward 实现不正确但不自知

```python
# 自定义 backward 最容易出 bug 的地方
# 一定要用 gradcheck 验证！

# 常见错误：
# 1. 忘记某个输入的梯度返回 None
# 2. 维度搞错（squeeze/unsqueeze）
# 3. 就地修改了 saved tensor
# 4. 精度不够（用 float64 做 gradcheck）

# 黄金法则：gradcheck 通过了再用到训练里
```

---

## 小结

| 方案 | 适用场景 | 复杂度 | torch.compile 兼容 |
|---|---|---|---|
| `torch.compile` | 自动优化 | ★☆☆ | ✅ 原生 |
| Triton kernel | memory-bound fused kernel | ★★☆ | ✅（通过 autograd.Function） |
| C++ extension | CPU 端自定义逻辑 | ★★☆ | ⚠️ 需要 torch.library |
| CUDA extension | 极致性能 kernel | ★★★ | ⚠️ 需要 torch.library |
| `torch.library` | 注册自定义算子 | ★★☆ | ✅ 原生 |
| FlexAttention | Attention 变体 | ★☆☆ | ✅ 原生 |

**一句话总结**：2025 年写自定义算子的优先级是 **FlexAttention > torch.compile > Triton > CUDA extension**。只有当上面的方案搞不定时，才需要往下走。

至此，模块 10 完成。你已经拥有了从 GPU 架构理解到动手写 kernel 的全栈能力。
