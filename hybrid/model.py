"""The hybrid SSM/attention decoder stack.

Layout follows NVIDIA's Nemotron-H / Mamba2-Hybrid: a pre-norm residual stack
whose sequence mixer is Mamba-2 almost everywhere, with a small number of
self-attention blocks at regular depths and SwiGLU MLPs after them.

    [Mamba] x11 -> [Attn] -> [MLP] -> [Mamba] x11 -> [Attn] -> [MLP] -> ...

Every block is ``x = x + f(norm(x))``, so blocks are individually removable or
replaceable - which is what makes grafting Qwen3-8B weights tractable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from .attention import Attention, KVCache, RMSNorm, RotaryEmbedding, SwiGLUMLP
from .config import HybridConfig
from .mamba2 import Mamba2Mixer, MambaState


@dataclass
class HybridCache:
    """Per-layer cache. Mamba entries are fixed size; attention entries grow."""

    entries: List[object] = field(default_factory=list)
    seen_tokens: int = 0

    def index_select(self, idx: torch.Tensor) -> "HybridCache":
        return HybridCache([e.index_select(idx) for e in self.entries], self.seen_tokens)

    def memory_bytes(self) -> int:
        total = 0
        for e in self.entries:
            if isinstance(e, MambaState):
                total += e.conv.numel() * e.conv.element_size()
                total += e.ssm.numel() * e.ssm.element_size()
            elif isinstance(e, KVCache) and e.k is not None:
                total += e.k.numel() * e.k.element_size() * 2
        return total


class HybridBlock(nn.Module):
    """Pre-norm residual wrapper around one mixer."""

    def __init__(self, cfg: HybridConfig, kind: str, layer_idx: int):
        super().__init__()
        self.kind = kind
        self.layer_idx = layer_idx
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if kind == "mamba":
            self.mixer = Mamba2Mixer(cfg, layer_idx)
        elif kind == "attention":
            self.mixer = Attention(cfg, layer_idx)
        elif kind == "mlp":
            self.mixer = SwiGLUMLP(cfg)
        else:
            raise ValueError(f"unknown block kind {kind!r}")

    def forward(self, hidden, *, cos=None, sin=None, cache=None, attention_mask=None, decode=False):
        residual = hidden
        h = self.norm(hidden)
        if self.kind == "mamba":
            h = self.mixer.step(h, cache) if decode else self.mixer(h, cache)
        elif self.kind == "attention":
            h = self.mixer(h, cos, sin, cache=cache, attention_mask=attention_mask)
        else:
            h = self.mixer(h)
        return residual + h


class HybridModel(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        self.schedule = cfg.layer_schedule()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            HybridBlock(cfg, kind, i) for i, kind in enumerate(self.schedule)
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings)
        self.apply(self._init)

    def _init(self, m: nn.Module):
        std = self.cfg.initializer_range
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=std)

    # -- cache -------------------------------------------------------------
    def allocate_cache(self, batch: int, device=None, dtype=None) -> HybridCache:
        device = device or self.embed_tokens.weight.device
        dtype = dtype or self.embed_tokens.weight.dtype
        entries = []
        for blk in self.layers:
            if blk.kind == "mamba":
                entries.append(blk.mixer.allocate_state(batch, device, dtype))
            elif blk.kind == "attention":
                entries.append(KVCache())
            else:
                entries.append(None)
        return HybridCache(entries)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Optional[HybridCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decode: bool = False,
    ) -> torch.Tensor:
        b, l = input_ids.shape
        hidden = self.embed_tokens(input_ids)

        past = cache.seen_tokens if cache is not None else 0
        positions = torch.arange(past, past + l, device=input_ids.device).unsqueeze(0).expand(b, l)
        cos, sin = self.rotary(positions)
        cos, sin = cos.to(hidden.dtype), sin.to(hidden.dtype)

        for i, blk in enumerate(self.layers):
            entry = cache.entries[i] if cache is not None else None
            hidden = blk(
                hidden,
                cos=cos,
                sin=sin,
                cache=entry,
                attention_mask=attention_mask,
                decode=decode and blk.kind == "mamba",
            )

        if cache is not None:
            cache.seen_tokens = past + l
        return self.norm(hidden)


class HybridForCausalLM(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        self.model = HybridModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, cache=None, attention_mask=None, labels=None, decode=False):
        hidden = self.model(input_ids, cache=cache, attention_mask=attention_mask, decode=decode)
        logits = self.lm_head(hidden)
        if labels is None:
            return logits
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return logits, loss

    # -- introspection ------------------------------------------------------
    def allocate_cache(self, batch: int, device=None, dtype=None) -> HybridCache:
        return self.model.allocate_cache(batch, device, dtype)

    def parameter_summary(self) -> dict:
        buckets = {"embed": 0, "mamba": 0, "attention": 0, "mlp": 0, "norm": 0, "lm_head": 0}
        buckets["embed"] = self.model.embed_tokens.weight.numel()
        for blk in self.model.layers:
            buckets[blk.kind] += sum(p.numel() for p in blk.mixer.parameters())
            buckets["norm"] += blk.norm.weight.numel()
        buckets["norm"] += self.model.norm.weight.numel()
        if not self.cfg.tie_word_embeddings:
            buckets["lm_head"] = self.lm_head.weight.numel()
        buckets["total"] = sum(p.numel() for p in self.parameters())
        return buckets
