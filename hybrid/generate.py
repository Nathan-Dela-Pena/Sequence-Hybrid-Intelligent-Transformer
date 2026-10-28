"""Autoregressive decoding for the hybrid stack.

Prefill runs the chunked SSD scan over the whole prompt; decoding then uses the
O(1) recurrent step for every Mamba layer and a normal KV cache for the few
attention layers.
"""

from __future__ import annotations

from typing import Callable, Iterator, Optional

import torch

from .model import HybridForCausalLM


def _sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        drop = probs - sorted_logits.softmax(dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(drop, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_idx, sorted_logits)
    return torch.multinomial(logits.softmax(dim=-1), num_samples=1)


@torch.no_grad()
def generate(
    model: HybridForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    eos_token_id: Optional[int] = None,
    on_token: Optional[Callable[[torch.Tensor], None]] = None,
) -> torch.Tensor:
    model.eval()
    b = input_ids.size(0)
    cache = model.allocate_cache(b, input_ids.device, next(model.parameters()).dtype)

    logits = model(input_ids, cache=cache)                 # prefill
    next_tok = _sample(logits[:, -1], temperature, top_p, top_k)

    out = [next_tok]
    done = torch.zeros(b, 1, dtype=torch.bool, device=input_ids.device)
    for _ in range(max_new_tokens - 1):
        if on_token is not None:
            on_token(next_tok)
        if eos_token_id is not None:
            done |= next_tok == eos_token_id
            if bool(done.all()):
                break
        logits = model(next_tok, cache=cache, decode=True)
        next_tok = _sample(logits[:, -1], temperature, top_p, top_k)
        out.append(next_tok)

    return torch.cat([input_ids] + out, dim=1)


@torch.no_grad()
def stream(
    model: HybridForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    eos_token_id: Optional[int] = None,
) -> Iterator[int]:
    """Yield token ids as they are produced. Batch size 1."""
    assert input_ids.size(0) == 1, "stream() is single-sequence"
    model.eval()
    cache = model.allocate_cache(1, input_ids.device, next(model.parameters()).dtype)
    logits = model(input_ids, cache=cache)
    for _ in range(max_new_tokens):
        tok = _sample(logits[:, -1], temperature, top_p, top_k)
        tid = int(tok)
        if eos_token_id is not None and tid == eos_token_id:
            return
        yield tid
        logits = model(tok, cache=cache, decode=True)
