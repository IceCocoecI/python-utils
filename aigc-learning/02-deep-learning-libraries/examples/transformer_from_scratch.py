"""从零实现一个极小 Transformer LM。

覆盖：RMSNorm / RoPE / causal self-attention / SwiGLU / KV cache / top-k top-p sampling。

运行：
  python transformer_from_scratch.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyConfig:
    vocab_size: int = 32
    max_seq_len: int = 64
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ffn: int = 128
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * rms


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        # q/k: (B, H, T, D). cos/sin broadcast to (1, 1, T, D).
        seq_len = q.size(-2)
        cos = self.cos[start_pos:start_pos + seq_len].repeat_interleave(2, dim=-1)
        sin = self.sin[start_pos:start_pos + seq_len].repeat_interleave(2, dim=-1)
        cos = cos.to(device=q.device, dtype=q.dtype)[None, None, :, :]
        sin = sin.to(device=q.device, dtype=q.dtype)[None, None, :, :]
        return q * cos + self.rotate_half(q) * sin, k * cos + self.rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(self.d_head, cfg.max_seq_len)
        self.register_buffer("k_cache", torch.empty(0), persistent=False)
        self.register_buffer("v_cache", torch.empty(0), persistent=False)

    def reset_cache(self) -> None:
        self.k_cache = torch.empty(0, device=self.out.weight.device)
        self.v_cache = torch.empty(0, device=self.out.weight.device)

    def forward(self, x: torch.Tensor, use_cache: bool = False, start_pos: int = 0) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        q, k = self.rope(q, k, start_pos=start_pos)

        cached_len = 0
        if use_cache:
            if self.k_cache.numel() > 0:
                cached_len = self.k_cache.size(-2)
                k = torch.cat([self.k_cache, k], dim=-2)
                v = torch.cat([self.v_cache, v], dim=-2)
            self.k_cache = k.detach()
            self.v_cache = v.detach()

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=(not use_cache) or (use_cache and cached_len == 0 and seq_len > 1),
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.out(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_model, d_ffn, bias=False)
        self.w3 = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ffn)

    def forward(self, x: torch.Tensor, use_cache: bool = False, start_pos: int = 0) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), use_cache=use_cache, start_pos=start_pos)
        x = x + self.ffn(self.norm2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def reset_cache(self) -> None:
        for block in self.blocks:
            block.attn.reset_cache()

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x, use_cache=use_cache, start_pos=start_pos)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def make_pattern_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    offsets = torch.arange(seq_len + 1, device=device).view(1, -1)
    tokens = (starts + offsets) % vocab_size
    return tokens[:, :-1], tokens[:, 1:]


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    logits = logits.clone()
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[..., -1, None]
        logits[logits < kth] = -float("inf")
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        remove = sorted_probs.cumsum(dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits[remove] = -float("inf")
        logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return logits


@torch.no_grad()
def generate(
    model: TinyTransformerLM,
    prompt: torch.Tensor,
    max_new_tokens: int = 12,
    temperature: float = 0.8,
    top_k: int = 8,
    top_p: float = 0.95,
) -> torch.Tensor:
    model.eval()
    model.reset_cache()
    out = prompt.clone()

    logits, _ = model(out, use_cache=True, start_pos=0)
    for _ in range(max_new_tokens):
        logits = top_k_top_p_filter(logits[:, -1, :] / max(temperature, 1e-6), top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_token], dim=1)
        logits, _ = model(next_token, use_cache=True, start_pos=out.size(1) - 1)
    return out


def demo_attention_equivalence() -> None:
    print("== Attention shape check ==")
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    mask = torch.ones(8, 8, dtype=torch.bool).tril()[None, None, :, :]
    naive = torch.softmax(scores.masked_fill(~mask, -float("inf")), dim=-1) @ v
    print("sdpa:", tuple(sdpa.shape), "max_diff:", (sdpa - naive).abs().max().item())


def demo_train_and_generate() -> None:
    print("\n== Tiny Transformer LM training ==")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TinyConfig()
    model = TinyTransformerLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    print("params:", sum(p.numel() for p in model.parameters()))

    model.train()
    for step in range(20):
        x, y = make_pattern_batch(batch_size=32, seq_len=24, vocab_size=cfg.vocab_size, device=device)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {0, 5, 10, 19}:
            print(f"step={step:02d} loss={loss.item():.4f}")

    prompt = torch.tensor([[3, 4, 5, 6]], device=device)
    generated = generate(model, prompt, max_new_tokens=12)
    print("prompt:   ", prompt.cpu().tolist()[0])
    print("generated:", generated.cpu().tolist()[0])


if __name__ == "__main__":
    demo_attention_equivalence()
    demo_train_and_generate()
