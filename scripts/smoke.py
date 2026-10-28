"""End-to-end smoke test: build a tiny hybrid, tokenize with Qwen3, generate.

Falls back to byte-level ids if the Qwen3 tokenizer is not downloaded, so this
runs offline.

    python scripts/smoke.py --prompt "def fib(n):"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid import HybridForCausalLM, HybridConfig, tiny_config, generate


def load_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # offline, or transformers not installed
        print(f"[smoke] Qwen tokenizer unavailable ({type(exc).__name__}); using bytes.")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="def fibonacci(n):")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    if tok is not None:
        ids = torch.tensor([tok(args.prompt)["input_ids"]])
        cfg = HybridConfig(
            vocab_size=len(tok), hidden_size=192, num_layers=6, attention_period=3,
            num_attention_heads=4, num_key_value_heads=2, head_dim=48,
            intermediate_size=384, mamba_d_state=32, mamba_headdim=48,
            mamba_ngroups=2, mamba_chunk_size=32, max_position_embeddings=4096,
        )
    else:
        ids = torch.tensor([[b for b in args.prompt.encode()]])
        cfg = tiny_config(vocab_size=256)

    torch.manual_seed(0)
    model = HybridForCausalLM(cfg).eval()
    glyph = {"mamba": "S", "attention": "A", "mlp": "F"}   # SSM / Attention / FFN
    print(f"[smoke] schedule: {''.join(glyph[k] for k in cfg.layer_schedule())}")
    print(f"[smoke] params:   {model.parameter_summary()}")

    out = generate(model, ids, max_new_tokens=args.max_new_tokens, temperature=0.8, top_p=0.95)
    text = tok.decode(out[0]) if tok is not None else bytes(out[0].tolist()).decode(errors="replace")
    print(f"[smoke] prompt:   {args.prompt!r}")
    print(f"[smoke] continuation (UNTRAINED weights - expect gibberish):\n{text}")

    # cache accounting
    cache = model.allocate_cache(1)
    model(ids, cache=cache)
    print(f"[smoke] cache after prefill of {ids.size(1)} tokens: "
          f"{cache.memory_bytes() / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
