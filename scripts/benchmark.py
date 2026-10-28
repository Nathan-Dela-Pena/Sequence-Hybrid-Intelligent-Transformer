"""Hybrid vs pure-transformer: decode latency and cache growth.

The pure-transformer baseline is the *same* code with ``attention_period=1``,
so the only difference being measured is the sequence mixer.

    python scripts/benchmark.py --lengths 256 1024 4096
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid import HybridForCausalLM, HybridConfig


def build(kind: str, depth: int, d: int) -> HybridForCausalLM:
    cfg = HybridConfig(
        vocab_size=4096,
        hidden_size=d,
        num_layers=depth,
        attention_period=1 if kind == "transformer" else 4,
        mlp_period=1,
        num_attention_heads=d // 64,
        num_key_value_heads=max(1, d // 256),
        head_dim=64,
        intermediate_size=d * 3,
        mamba_d_state=64,
        mamba_headdim=64,
        mamba_ngroups=max(1, d // 256),
        mamba_chunk_size=64,
        max_position_embeddings=65536,
    )
    return HybridForCausalLM(cfg).eval()


@torch.no_grad()
def measure(model: HybridForCausalLM, prompt_len: int, decode_steps: int) -> dict:
    ids = torch.randint(0, 4096, (1, prompt_len))
    cache = model.allocate_cache(1)

    t0 = time.perf_counter()
    model(ids, cache=cache)
    prefill_s = time.perf_counter() - t0
    cache_after_prefill = cache.memory_bytes()

    tok = torch.randint(0, 4096, (1, 1))
    t0 = time.perf_counter()
    for _ in range(decode_steps):
        model(tok, cache=cache, decode=True)
    decode_s = (time.perf_counter() - t0) / decode_steps

    return {
        "prefill_s": round(prefill_s, 4),
        "ms_per_token": round(decode_s * 1000, 3),
        "cache_mib_at_prompt": round(cache_after_prefill / 2**20, 3),
        "cache_mib_final": round(cache.memory_bytes() / 2**20, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048])
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    models = {k: build(k, args.depth, args.hidden) for k in ("transformer", "hybrid")}
    params = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}

    rows = []
    for L in args.lengths:
        for kind, m in models.items():
            r = measure(m, L, args.decode_steps)
            r.update(kind=kind, prompt_len=L)
            rows.append(r)

    w = "{:<12} {:>8} {:>11} {:>14} {:>16}"
    print(f"depth={args.depth} hidden={args.hidden} params={params}")
    print(w.format("model", "ctx", "prefill s", "ms/token", "KV+SSM cache MiB"))
    for r in rows:
        print(w.format(
            r["kind"], r["prompt_len"], f"{r['prefill_s']:.3f}",
            f"{r['ms_per_token']:.2f}", f"{r['cache_mib_at_prompt']:.2f}",
        ))

    if args.json:
        Path(args.json).write_text(json.dumps({"params": params, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
