"""Configuration for the hybrid SSM/attention stack.

The layer schedule follows NVIDIA's Nemotron-H / Mamba2-Hybrid recipe: the
backbone is overwhelmingly Mamba-2, with a small number of self-attention
layers interleaved at regular depths to recover the exact-recall behaviour that
a fixed-size recurrent state cannot provide.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Literal
import json

LayerKind = Literal["mamba", "attention", "mlp"]


@dataclass
class HybridConfig:
    # --- vocabulary / embeddings -------------------------------------------
    vocab_size: int = 151936          # Qwen3 tokenizer
    hidden_size: int = 4096
    tie_word_embeddings: bool = False

    # --- depth / schedule ---------------------------------------------------
    num_layers: int = 36
    #: One attention block every ``attention_period`` blocks. Nemotron-H uses
    #: roughly 8% attention; 1-in-12 at depth 36 gives 3 attention layers.
    attention_period: int = 12
    #: Insert a SwiGLU MLP after every attention block (Nemotron-H style).
    #: Mamba-2 blocks are used bare, as in the original Mamba stack.
    mlp_after_attention: bool = True
    #: Also interleave MLPs between Mamba blocks at this period (0 = never).
    #: 1 reproduces Nemotron-H, where every mixer is followed by an FFN - and
    #: lets every one of Qwen3's MLPs be reused verbatim by the graft.
    mlp_period: int = 1

    # --- attention ----------------------------------------------------------
    num_attention_heads: int = 32
    num_key_value_heads: int = 8      # GQA, matches Qwen3-8B
    head_dim: int = 128
    qk_norm: bool = True              # Qwen3 normalises q and k per head
    rope_theta: float = 1_000_000.0
    attention_dropout: float = 0.0
    sliding_window: int | None = None

    # --- mamba-2 ------------------------------------------------------------
    mamba_expand: int = 2             # d_inner = expand * hidden_size
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_headdim: int = 64
    mamba_ngroups: int = 8            # B/C groups, mirrors GQA grouping
    mamba_chunk_size: int = 64        # SSD chunk length
    mamba_dt_min: float = 0.001
    mamba_dt_max: float = 0.1
    mamba_dt_init_floor: float = 1e-4
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False

    # --- mlp ------------------------------------------------------------------
    intermediate_size: int = 12288
    hidden_act: str = "silu"

    # --- norm / init -----------------------------------------------------------
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    # --- misc ------------------------------------------------------------------
    max_position_embeddings: int = 32768
    pad_token_id: int | None = 151643
    bos_token_id: int | None = 151643
    eos_token_id: int | None = 151645

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.d_inner % self.mamba_headdim != 0:
            raise ValueError(
                f"d_inner ({self.d_inner}) must be divisible by mamba_headdim ({self.mamba_headdim})"
            )
        if self.mamba_nheads % self.mamba_ngroups != 0:
            raise ValueError("mamba head count must be divisible by mamba_ngroups")

    # --- derived ----------------------------------------------------------------
    @property
    def d_inner(self) -> int:
        return self.mamba_expand * self.hidden_size

    @property
    def mamba_nheads(self) -> int:
        return self.d_inner // self.mamba_headdim

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def layer_schedule(self) -> List[LayerKind]:
        """Expand the periodic recipe into a concrete per-layer block list.

        ``num_layers`` counts *sequence-mixing* blocks (mamba or attention);
        MLP blocks are inserted on top of that count, which is how Nemotron-H
        reports its depth.
        """
        schedule: List[LayerKind] = []
        for i in range(self.num_layers):
            is_attention = (
                self.attention_period > 0
                and (i + 1) % self.attention_period == 0
            )
            if is_attention:
                schedule.append("attention")
                if self.mlp_after_attention:
                    schedule.append("mlp")
            else:
                schedule.append("mamba")
                if self.mlp_period > 0 and (i + 1) % self.mlp_period == 0:
                    schedule.append("mlp")
        return schedule

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "HybridConfig":
        with open(path) as fh:
            return cls(**json.load(fh))


def tiny_config(**overrides) -> HybridConfig:
    """A CPU-sized config used by the tests and the smoke script."""
    base = dict(
        vocab_size=512,
        hidden_size=128,
        num_layers=6,
        attention_period=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        mamba_d_state=16,
        mamba_headdim=32,
        mamba_ngroups=2,
        mamba_chunk_size=8,
        intermediate_size=256,
        max_position_embeddings=512,
    )
    base.update(overrides)
    return HybridConfig(**base)


def qwen3_8b_hybrid_config(**overrides) -> HybridConfig:
    """Hybrid stack dimensioned to accept grafted Qwen3-8B tensors.

    Qwen3-8B is 36 layers, hidden 4096, 32 q-heads / 8 kv-heads, head_dim 128,
    SwiGLU intermediate 12288, rope_theta 1e6. We keep all of that and swap
    11 of every 12 attention blocks for Mamba-2.
    """
    base = dict(
        vocab_size=151936,
        hidden_size=4096,
        num_layers=36,
        attention_period=12,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=12288,
        mlp_period=1,
        rope_theta=1_000_000.0,
        qk_norm=True,
        max_position_embeddings=40960,
    )
    base.update(overrides)
    return HybridConfig(**base)
