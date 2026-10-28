# Sequence-Hybrid-Intelligent-Transformer

A single-purpose coding agent powered by a hybrid of SSMs & transformers.

This repo contains a from-scratch, CPU-runnable implementation of an
**NVIDIA Nemotron-H / Mamba2-Hybrid style stack** — Mamba-2 state-space mixers
for most layers, a small number of self-attention layers interleaved for exact
recall — plus a grafting path that initialises it from **Qwen3-8B**.

```
S F S F S F S F S F S F S F S F S F S F S F A F S F ...
                                            ^
S = Mamba-2 (SSM)   A = GQA attention   F = SwiGLU FFN
33 Mamba + 3 attention + 36 FFN at the Qwen3-8B scale (~8% attention)
```

## Why

Qwen3-8B's KV cache is ~2 GiB per sequence at 32k context. Swapping 33 of its 36
attention layers for Mamba-2 cuts that ~12× while keeping its embeddings, all 36
MLPs and the remaining attention blocks verbatim. See
[`docs/hybrid-vs-transformer.md`](docs/hybrid-vs-transformer.md) for the
measured trade-offs — including where the hybrid is *worse*.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

> **Apple Intel note:** PyTorch's last x86_64 macOS wheel is 2.2.2 (Python ≤3.11).
> This repo is developed and tested on that combination:
> `/usr/bin/python3 -m venv .venv && .venv/bin/pip install torch==2.2.2 'numpy<2'`.

## Quick start

```bash
.venv/bin/python -m pytest tests -q          # 9 tests, ~3s
.venv/bin/python scripts/smoke.py            # tiny model + real Qwen3 tokenizer
.venv/bin/python -m hybrid.graft --dry-run   # layer plan & param budget at 8B
.venv/bin/python scripts/benchmark.py --lengths 128 512 2048 8192
```

```python
from hybrid import HybridForCausalLM, qwen3_8b_hybrid_config, generate

model = HybridForCausalLM(qwen3_8b_hybrid_config())
out = generate(model, input_ids, max_new_tokens=128, temperature=0.7)
```

## Hooking up Qwen3-8B

```bash
.venv/bin/python -m hybrid.graft --source Qwen/Qwen3-8B --out ./checkpoints/hybrid-8b
```

This downloads Qwen3-8B, builds a dimensionally-matched hybrid, and copies
across everything that transfers:

| Component | Source | Status |
|---|---|---|
| `embed_tokens`, `lm_head`, final norm | Qwen3-8B | copied verbatim |
| 36 SwiGLU MLPs | Qwen3-8B | copied verbatim |
| 3 attention blocks (Qwen layers 11, 23, 35) | Qwen3-8B | copied verbatim, incl. q/k RMSNorm |
| 33 Mamba-2 mixers | — | **randomly initialised** |

~65% of parameters transfer. **The grafted checkpoint is not usable as-is** —
the Mamba layers are random, so it must be distilled against Qwen3-8B as the
teacher (MOHAWK-style) before it produces sensible text. `graft_report.json`
records the exact reuse fraction.

The attention config is kept bit-compatible with Qwen3 on purpose: GQA 32/8,
`head_dim` 128, per-head RMSNorm on q and k, RoPE θ=1e6, no qkv bias.

## Layout

```
hybrid/
  config.py     HybridConfig + the periodic layer schedule
  mamba2.py     Mamba-2 SSD mixer: chunked parallel scan + O(1) recurrent step
  attention.py  Qwen3-compatible GQA attention, RoPE, SwiGLU, RMSNorm
  model.py      Pre-norm residual stack, HybridCache (fixed SSM state + growing KV)
  generate.py   Prefill-then-decode sampling loop
  graft.py      Qwen3-8B -> hybrid weight transfer
scripts/
  smoke.py      End-to-end run with the real Qwen3 tokenizer
  benchmark.py  Hybrid vs pure transformer (attention_period=1) latency & cache
tests/          Equivalence, causality and cache-scaling tests
docs/           Hybrid vs pure transformer: pros, cons, measurements
```

## What the tests actually check

The two that matter most:

* **`test_ssd_scan_matches_naive_recurrence`** — the chunked parallel scan used
  for training/prefill and the single-token recurrence used for decoding agree
  to 1e-8 in float64. If these ever diverge, generation silently degrades.
* **`test_prefill_then_decode_matches_full_forward`** — incremental decoding
  with the cache produces the same logits as one full forward pass.

Plus causality, chunk-size invariance, and the scaling property the whole design
rests on: SSM state is constant in sequence length while KV cache is linear.

## Status / not done

* The scan is pure PyTorch — correct, not fast. Swap in `mamba-ssm`'s fused
  kernels for GPU use (`requirements.txt` has the pins, commented).
* No distillation loop yet; the graft produces an initialisation, not a model.
* No training script, no agent harness. This is the architecture layer only.
