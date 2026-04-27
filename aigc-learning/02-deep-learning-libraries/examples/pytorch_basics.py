"""PyTorch 基础示例：Tensor / autograd / device / 基础模型。
运行：python pytorch_basics.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def demo_tensor():
    print("== Tensor 基础 ==")
    x = torch.randn(2, 3, 4)
    print(f"shape={x.shape}, dtype={x.dtype}, device={x.device}")

    print("view:", x.view(2, 12).shape)
    print("permute:", x.permute(2, 0, 1).shape)
    print("unsqueeze:", x.unsqueeze(0).shape)
    print("squeeze:", x.unsqueeze(0).squeeze().shape)

    mat = torch.randn(3, 4) @ torch.randn(4, 5)
    print("matmul:", mat.shape)

    a = torch.randn(5, 3)
    b = torch.randn(3)
    print("broadcast a-b:", (a - b).shape)


def demo_autograd():
    print("\n== Autograd ==")
    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 3 + 2 * x ** 2 + x
    y.backward()
    print(f"dy/dx at x=2: {x.grad.item()}  (expected 3x^2+4x+1 = 21)")

    w = torch.randn(3, 4, requires_grad=True)
    b = torch.randn(4, requires_grad=True)
    X = torch.randn(10, 3)
    y = (X @ w + b).sum()
    y.backward()
    print(f"grad shapes: w={w.grad.shape}, b={b.grad.shape}")


def demo_no_grad():
    print("\n== no_grad vs require_grad ==")
    x = torch.randn(1000, 1000, requires_grad=True)
    y = (x ** 2).sum()
    print(f"with grad: requires_grad={y.requires_grad}")

    with torch.no_grad():
        z = (x ** 2).sum()
    print(f"no_grad: requires_grad={z.requires_grad}")


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def demo_module():
    print("\n== nn.Module ==")
    model = TinyMLP(10, 32, 3)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params}")

    x = torch.randn(4, 10)
    out = model(x)
    print(f"input: {x.shape}  ->  output: {out.shape}")

    print(f"module: {model}")


class TinySelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(attn)


def demo_attention():
    print("\n== 自己写一个 Self-Attention ==")
    attn = TinySelfAttention(d_model=64, n_heads=4)
    x = torch.randn(2, 10, 64)
    y = attn(x)
    print(f"attention input: {x.shape}  ->  output: {y.shape}")
    print(f"params: {sum(p.numel() for p in attn.parameters())}")


def demo_device():
    print("\n== Device ==")
    if torch.cuda.is_available():
        print(f"CUDA available, {torch.cuda.device_count()} device(s)")
        x = torch.randn(3, 3, device="cuda")
        print(f"x.device: {x.device}")
    else:
        print("CUDA not available; using CPU")


if __name__ == "__main__":
    torch.manual_seed(42)
    demo_tensor()
    demo_autograd()
    demo_no_grad()
    demo_module()
    demo_attention()
    demo_device()
