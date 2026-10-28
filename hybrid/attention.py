"""Qwen3-compatible self-attention.

Kept deliberately identical in parameterisation to Qwen3's decoder attention
(GQA, per-head RMSNorm on q and k, RoPE with theta=1e6, no qkv bias) so that
Qwen3-8B's attention tensors can be grafted in without reshaping - see
``hybrid/graft.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HybridConfig
from .mamba2 import GatedRMSNorm


@dataclass
class KVCache:
    """Growing KV cache for one attention layer. Unlike :class:`MambaState`
    this scales linearly with sequence length - the cost the hybrid pays for
    keeping a handful of attention layers."""

    k: torch.Tensor | None = None   # (b, kv_heads, t, head_dim)
    v: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.size(2)

    def append(self, k: torch.Tensor, v: torch.Tensor):
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    def index_select(self, idx: torch.Tensor) -> "KVCache":
        if self.k is None:
            return KVCache()
        return KVCache(self.k.index_select(0, idx), self.v.index_select(0, idx))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float, max_positions: int):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.max_positions = max_positions

    def forward(self, positions: torch.Tensor):
        """positions: (b, l) int64 -> (cos, sin) each (b, l, dim)."""
        freqs = positions.float().unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)        # (b, 1, l, d)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, cfg: HybridConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.groups = cfg.num_kv_groups
        self.scale = self.hd ** -0.5
        self.sliding_window = cfg.sliding_window

        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)

        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.hd, cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.hd, cfg.rms_norm_eps)
        else:
            self.q_norm = self.k_norm = None

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, l, _ = hidden.shape
        q = self.q_proj(hidden).view(b, l, self.nh, self.hd)
        k = self.k_proj(hidden).view(b, l, self.nkv, self.hd)
        v = self.v_proj(hidden).view(b, l, self.nkv, self.hd)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))     # (b, heads, l, hd)
        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            k, v = cache.append(k, v)
        kv_len = k.size(2)

        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)

        mask = self._build_mask(l, kv_len, hidden.device, hidden.dtype, attention_mask)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        out = out.transpose(1, 2).reshape(b, l, self.nh * self.hd)
        return self.o_proj(out)

    def _build_mask(self, q_len, kv_len, device, dtype, attention_mask):
        """Causal (optionally sliding-window) float mask, merged with a padding mask."""
        q_pos = torch.arange(kv_len - q_len, kv_len, device=device).view(-1, 1)
        k_pos = torch.arange(kv_len, device=device).view(1, -1)
        allowed = k_pos <= q_pos
        if self.sliding_window:
            allowed &= (q_pos - k_pos) < self.sliding_window
        mask = torch.zeros(q_len, kv_len, device=device, dtype=dtype)
        mask = mask.masked_fill(~allowed, torch.finfo(dtype).min).view(1, 1, q_len, kv_len)
        if attention_mask is not None:                       # (b, kv_len), 1 = keep
            pad = (1 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
            mask = mask + pad
        return mask


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


__all__ = ["Attention", "SwiGLUMLP", "RMSNorm", "GatedRMSNorm", "KVCache", "RotaryEmbedding"]
