"""Graft Qwen3-8B weights onto the hybrid stack.

Why this works at all
---------------------
A pre-norm residual transformer is a sequence of independent read-modify-write
operations on a shared residual stream. Replacing a *subset* of the sequence
mixers changes what is written at those depths, but the residual basis - the
embedding, the MLPs, the final norm, the LM head - stays meaningful. So we:

* copy verbatim:   embed_tokens, lm_head, final norm
* copy verbatim:   every SwiGLU MLP (they dominate the parameter count)
* copy verbatim:   the attention blocks we keep, at their Qwen depths
* initialise fresh: the Mamba-2 mixers that replace the dropped attention

The result is *not* a trained model - the Mamba layers are random and the
network needs distillation to recover quality. This is exactly the procedure
NVIDIA used for Mamba2-Hybrid / MOHAWK-style conversion: initialise from a
transformer, then distill the attention layers' behaviour into the SSM layers.
``plan_graft`` reports what gets reused so you can see the ratio before
committing GPU hours.

Usage
-----
    python -m hybrid.graft --source Qwen/Qwen3-8B --out ./checkpoints/hybrid-8b
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .config import HybridConfig, qwen3_8b_hybrid_config
from .model import HybridForCausalLM


# Qwen3 names -> our names, for a block we keep as attention.
ATTN_MAP = {
    "self_attn.q_proj.weight": "mixer.q_proj.weight",
    "self_attn.k_proj.weight": "mixer.k_proj.weight",
    "self_attn.v_proj.weight": "mixer.v_proj.weight",
    "self_attn.o_proj.weight": "mixer.o_proj.weight",
    "self_attn.q_norm.weight": "mixer.q_norm.weight",
    "self_attn.k_norm.weight": "mixer.k_norm.weight",
    "input_layernorm.weight": "norm.weight",
}
MLP_MAP = {
    "mlp.gate_proj.weight": "mixer.gate_proj.weight",
    "mlp.up_proj.weight": "mixer.up_proj.weight",
    "mlp.down_proj.weight": "mixer.down_proj.weight",
    "post_attention_layernorm.weight": "norm.weight",
}


@dataclass
class GraftPlan:
    """Which Qwen depth feeds which hybrid block."""

    #: hybrid block index -> (kind, source qwen layer index or None)
    assignments: List[Tuple[int, str, int | None]]
    kept_attention: List[int]
    dropped_attention: List[int]

    def describe(self) -> str:
        lines = [
            f"hybrid blocks: {len(self.assignments)}",
            f"attention kept from Qwen layers: {self.kept_attention}",
            f"attention dropped (replaced by Mamba-2): {len(self.dropped_attention)} layers",
        ]
        return "\n".join(lines)


def plan_graft(cfg: HybridConfig, source_layers: int) -> GraftPlan:
    """Map each hybrid block to a Qwen source layer.

    Mixing block i of N_hybrid maps to Qwen layer round(i * N_src / N_hybrid),
    so a hybrid that is shallower or deeper than the source still lines up
    monotonically. The MLP that follows an attention block inherits the same
    source layer's MLP.
    """
    schedule = cfg.layer_schedule()
    mixer_positions = [i for i, k in enumerate(schedule) if k in ("mamba", "attention")]
    n_mix = len(mixer_positions)

    assignments: List[Tuple[int, str, int | None]] = []
    kept, dropped = [], []
    src_for_block: Dict[int, int] = {}
    for mix_i, blk_i in enumerate(mixer_positions):
        src = min(source_layers - 1, round(mix_i * source_layers / n_mix))
        src_for_block[blk_i] = src
        kind = schedule[blk_i]
        assignments.append((blk_i, kind, src))
        (kept if kind == "attention" else dropped).append(src)

    last_src = 0
    for i, kind in enumerate(schedule):
        if kind in ("mamba", "attention"):
            last_src = src_for_block[i]
        else:  # mlp inherits the preceding mixer's source layer
            assignments.append((i, "mlp", last_src))

    assignments.sort(key=lambda t: t[0])
    return GraftPlan(assignments, kept, dropped)


def _load_source_state(source: str) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Load a Qwen3 checkpoint as a flat state dict (safetensors, lazily)."""
    from safetensors.torch import load_file

    if os.path.isdir(source):
        path = source
    else:
        from huggingface_hub import snapshot_download

        path = snapshot_download(source, allow_patterns=["*.safetensors*", "*.json", "*.txt"])

    with open(os.path.join(path, "config.json")) as fh:
        src_cfg = json.load(fh)

    index = os.path.join(path, "model.safetensors.index.json")
    state: Dict[str, torch.Tensor] = {}
    if os.path.exists(index):
        with open(index) as fh:
            shards = sorted(set(json.load(fh)["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    for shard in shards:
        state.update(load_file(os.path.join(path, shard)))
    return state, src_cfg


def config_from_qwen(src_cfg: dict, **overrides) -> HybridConfig:
    """Derive a hybrid config that is dimensionally compatible with a Qwen3 config."""
    base = dict(
        vocab_size=src_cfg["vocab_size"],
        hidden_size=src_cfg["hidden_size"],
        num_layers=src_cfg["num_hidden_layers"],
        num_attention_heads=src_cfg["num_attention_heads"],
        num_key_value_heads=src_cfg["num_key_value_heads"],
        head_dim=src_cfg.get("head_dim", src_cfg["hidden_size"] // src_cfg["num_attention_heads"]),
        intermediate_size=src_cfg["intermediate_size"],
        rope_theta=src_cfg.get("rope_theta", 1e6),
        rms_norm_eps=src_cfg.get("rms_norm_eps", 1e-6),
        max_position_embeddings=src_cfg.get("max_position_embeddings", 32768),
        tie_word_embeddings=src_cfg.get("tie_word_embeddings", False),
        qk_norm=True,
    )
    base.update(overrides)
    return HybridConfig(**base)


def graft(
    model: HybridForCausalLM,
    source_state: Dict[str, torch.Tensor],
    plan: GraftPlan,
    strict_shapes: bool = True,
) -> dict:
    """Copy reusable Qwen tensors into ``model`` in place. Returns a report."""
    own = model.state_dict()
    copied, skipped = [], []

    def put(dst: str, src: str):
        if src not in source_state:
            skipped.append((dst, f"missing source {src}"))
            return
        t = source_state[src]
        if dst not in own:
            skipped.append((dst, "no such target"))
            return
        if own[dst].shape != t.shape:
            msg = f"shape {tuple(t.shape)} != {tuple(own[dst].shape)}"
            if strict_shapes:
                raise ValueError(f"{dst}: {msg}")
            skipped.append((dst, msg))
            return
        own[dst].copy_(t.to(own[dst].dtype))
        copied.append(dst)

    put("model.embed_tokens.weight", "model.embed_tokens.weight")
    put("model.norm.weight", "model.norm.weight")
    if not model.cfg.tie_word_embeddings:
        put("lm_head.weight", "lm_head.weight")

    for blk_i, kind, src in plan.assignments:
        if src is None or kind == "mamba":
            continue
        mapping = ATTN_MAP if kind == "attention" else MLP_MAP
        for s, d in mapping.items():
            put(f"model.layers.{blk_i}.{d}", f"model.layers.{src}.{s}")

    model.load_state_dict(own)
    mamba_params = sum(
        p.numel()
        for blk in model.model.layers
        if blk.kind == "mamba"
        for p in blk.mixer.parameters()
    )
    total = sum(p.numel() for p in model.parameters())
    return {
        "tensors_copied": len(copied),
        "tensors_skipped": skipped,
        "params_total": total,
        "params_randomly_initialised": mamba_params,
        "reuse_fraction": round(1 - mamba_params / total, 4),
    }


def build_qwen3_hybrid(source: str = "Qwen/Qwen3-8B", **cfg_overrides):
    """Load Qwen3, build the hybrid, graft. Returns (model, config, report)."""
    state, src_cfg = _load_source_state(source)
    cfg = config_from_qwen(src_cfg, **cfg_overrides)
    model = HybridForCausalLM(cfg)
    plan = plan_graft(cfg, src_cfg["num_hidden_layers"])
    report = graft(model, state, plan)
    return model, cfg, report


def analytic_param_counts(cfg: HybridConfig) -> dict:
    """Parameter counts without allocating the model - the 8B config does not
    fit comfortably in RAM just to be counted."""
    sched = cfg.layer_schedule()
    d, n = cfg.hidden_size, cfg.mamba_d_state

    attn = (
        cfg.num_attention_heads * cfg.head_dim * d
        + 2 * cfg.num_key_value_heads * cfg.head_dim * d
        + cfg.num_attention_heads * cfg.head_dim * d
        + (2 * cfg.head_dim if cfg.qk_norm else 0)
    )
    mlp = 3 * d * cfg.intermediate_size
    conv_dim = cfg.d_inner + 2 * cfg.mamba_ngroups * n
    mamba = (
        d * (2 * cfg.d_inner + 2 * cfg.mamba_ngroups * n + cfg.mamba_nheads)  # in_proj
        + conv_dim * cfg.mamba_d_conv + (conv_dim if cfg.mamba_conv_bias else 0)
        + 3 * cfg.mamba_nheads                                                # dt_bias, A_log, D
        + cfg.d_inner                                                         # gated norm
        + cfg.d_inner * d                                                     # out_proj
    )

    counts = {
        "embed": cfg.vocab_size * d,
        "mamba": sched.count("mamba") * mamba,
        "attention": sched.count("attention") * attn,
        "mlp": sched.count("mlp") * mlp,
        "norm": (len(sched) + 1) * d,
        "lm_head": 0 if cfg.tie_word_embeddings else cfg.vocab_size * d,
    }
    counts["total"] = sum(counts.values())
    counts["blocks"] = {k: sched.count(k) for k in ("mamba", "attention", "mlp")}
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="Qwen/Qwen3-8B", help="HF repo id or local dir")
    ap.add_argument("--out", default="./checkpoints/hybrid-qwen3-8b")
    ap.add_argument("--attention-period", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, load nothing")
    args = ap.parse_args()

    if args.dry_run:
        cfg = qwen3_8b_hybrid_config(attention_period=args.attention_period)
        plan = plan_graft(cfg, cfg.num_layers)
        print(plan.describe())
        print(json.dumps(analytic_param_counts(cfg), indent=2))
        return

    model, cfg, report = build_qwen3_hybrid(
        args.source, attention_period=args.attention_period
    )
    os.makedirs(args.out, exist_ok=True)
    cfg.to_json(os.path.join(args.out, "hybrid_config.json"))
    torch.save(model.state_dict(), os.path.join(args.out, "pytorch_model.bin"))
    with open(os.path.join(args.out, "graft_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
