"""einops 综合示例：rearrange / reduce / repeat + ViT patch + Multi-Head Attention。
运行：python einops_demo.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


def demo_rearrange():
    print("== 1. rearrange ==")
    x = torch.randn(2, 3, 4, 4)
    print("CHW -> HWC:", rearrange(x, "b c h w -> b h w c").shape)
    print("spatial flatten:", rearrange(x, "b c h w -> b (h w) c").shape)
    print("全部展平:", rearrange(x, "b c h w -> b (c h w)").shape)

    y = rearrange(x, "b (g c) h w -> b g c h w", g=3)
    print("split channel into groups:", y.shape)


def demo_reduce():
    print("\n== 2. reduce ==")
    x = torch.randn(2, 3, 8, 8)
    print("global avg pool:", reduce(x, "b c h w -> b c", "mean").shape)
    print("2x2 max pool:", reduce(x, "b c (h 2) (w 2) -> b c h w", "max").shape)
    print("keepdim avg:", reduce(x, "b c h w -> b c 1 1", "mean").shape)


def demo_repeat():
    print("\n== 3. repeat ==")
    x = torch.randn(3, 4, 4)
    print("add batch dim:", repeat(x, "c h w -> b c h w", b=8).shape)

    mask = torch.arange(5)
    print("broadcast mask:", repeat(mask, "t -> b t", b=2))


def demo_vit_patch():
    print("\n== 4. ViT: image -> patches ==")
    img = torch.randn(1, 3, 224, 224)
    patches = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=16, p2=16)
    print(f"image: {img.shape}  ->  patches: {patches.shape}")
    print(f"num patches = {(224 // 16) ** 2} = {patches.shape[1]}")
    print(f"patch embed dim = 16*16*3 = {patches.shape[-1]}")


def demo_mha_einops():
    print("\n== 5. Multi-Head Attention：einops 一行搞定 QKV ==")
    B, T, C, H = 2, 10, 64, 4
    x = torch.randn(B, T, C)
    qkv_proj = nn.Linear(C, 3 * C)

    qkv = qkv_proj(x)
    q, k, v = rearrange(qkv, "b t (three h d) -> three b h t d", three=3, h=H)
    print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")

    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = rearrange(attn, "b h t d -> b t (h d)")
    print(f"attn output: {out.shape}")


def demo_einops_layers():
    print("\n== 6. einops Layers：塞进 nn.Sequential ==")
    tiny_cnn = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        Reduce("b c (h 2) (w 2) -> b c h w", "max"),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        Reduce("b c h w -> b c", "mean"),
        nn.Linear(32, 10),
    )
    x = torch.randn(4, 3, 32, 32)
    y = tiny_cnn(x)
    print(f"input: {x.shape}  ->  logits: {y.shape}")
    print(f"params: {sum(p.numel() for p in tiny_cnn.parameters())}")


def demo_cfg_split():
    print("\n== 7. Classifier-Free Guidance：条件/无条件 batch 切分 ==")
    cond = torch.randn(4, 4, 8, 8)
    uncond = torch.randn(4, 4, 8, 8)
    both = torch.cat([cond, uncond], dim=0)
    cond_out, uncond_out = rearrange(both, "(two b) c h w -> two b c h w", two=2)
    print(f"cond_out: {cond_out.shape}, uncond_out: {uncond_out.shape}")


if __name__ == "__main__":
    torch.manual_seed(0)
    demo_rearrange()
    demo_reduce()
    demo_repeat()
    demo_vit_patch()
    demo_mha_einops()
    demo_einops_layers()
    demo_cfg_split()
