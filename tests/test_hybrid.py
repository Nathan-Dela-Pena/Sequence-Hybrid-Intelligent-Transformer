import torch
import pytest

from hybrid import HybridForCausalLM, tiny_config, generate
from hybrid.config import qwen3_8b_hybrid_config
from hybrid.mamba2 import Mamba2Mixer
from hybrid.graft import plan_graft, config_from_qwen, graft

torch.manual_seed(0)


@pytest.fixture(scope="module")
def model():
    m = HybridForCausalLM(tiny_config()).double().eval()
    return m


def test_schedule_is_mostly_mamba():
    cfg = qwen3_8b_hybrid_config()
    sched = cfg.layer_schedule()
    assert sched.count("attention") == 3
    assert sched.count("mamba") == 33
    # every attention block is immediately followed by an MLP
    for i, k in enumerate(sched):
        if k == "attention":
            assert sched[i + 1] == "mlp"


def test_forward_shapes(model):
    ids = torch.randint(0, 512, (2, 17))
    assert model(ids).shape == (2, 17, 512)


def test_causality(model):
    """Changing token t must not move logits at positions < t."""
    ids = torch.randint(0, 512, (1, 24))
    a = model(ids)
    ids2 = ids.clone()
    ids2[0, 12] = (ids2[0, 12] + 7) % 512
    b = model(ids2)
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-9)
    assert not torch.allclose(a[:, 12], b[:, 12], atol=1e-6)


def test_ssd_scan_matches_naive_recurrence():
    cfg = tiny_config()
    mixer = Mamba2Mixer(cfg, 0).double().eval()
    x = torch.randn(2, 20, cfg.hidden_size, dtype=torch.float64)

    parallel = mixer(x)

    state = mixer.allocate_state(2, x.device, torch.float64)
    steps = [mixer.step(x[:, t : t + 1], state) for t in range(x.size(1))]
    recurrent = torch.cat(steps, dim=1)

    assert torch.allclose(parallel, recurrent, atol=1e-8), (
        (parallel - recurrent).abs().max().item()
    )


def test_prefill_then_decode_matches_full_forward(model):
    """The whole point of the cache: incremental == one-shot."""
    ids = torch.randint(0, 512, (1, 19))
    full = model(ids)

    cache = model.allocate_cache(1, ids.device, torch.float64)
    model(ids[:, :-1], cache=cache)
    last = model(ids[:, -1:], cache=cache, decode=True)

    assert torch.allclose(full[:, -1:], last, atol=1e-8), (full[:, -1:] - last).abs().max().item()


def test_chunk_size_invariance():
    """The chunked scan must be independent of chunk_size."""
    outs = []
    for cs in (4, 8, 32):
        torch.manual_seed(1)
        mixer = Mamba2Mixer(tiny_config(mamba_chunk_size=cs), 0).double().eval()
        torch.manual_seed(2)
        x = torch.randn(1, 21, 128, dtype=torch.float64)
        outs.append(mixer(x))
    assert torch.allclose(outs[0], outs[1], atol=1e-9)
    assert torch.allclose(outs[0], outs[2], atol=1e-9)


def test_greedy_generate_is_deterministic(model):
    ids = torch.randint(0, 512, (1, 6))
    a = generate(model, ids, max_new_tokens=8, temperature=0.0)
    b = generate(model, ids, max_new_tokens=8, temperature=0.0)
    assert a.shape == (1, 14)
    assert torch.equal(a, b)


def test_mamba_cache_is_constant_size():
    """Mamba state must not grow with sequence length; attention KV must."""
    cfg = tiny_config()
    m = HybridForCausalLM(cfg).eval()
    sizes = []
    for n in (8, 64):
        cache = m.allocate_cache(1)
        m(torch.randint(0, 512, (1, n)), cache=cache)
        mamba = sum(
            e.conv.numel() + e.ssm.numel()
            for e in cache.entries
            if hasattr(e, "ssm")
        )
        kv = sum(e.k.numel() for e in cache.entries if hasattr(e, "k") and e.k is not None)
        sizes.append((mamba, kv))
    assert sizes[0][0] == sizes[1][0]          # SSM state: constant
    assert sizes[1][1] == sizes[0][1] * 8      # KV cache: linear in length


def test_graft_plan_and_copy():
    src_cfg = dict(
        vocab_size=512, hidden_size=128, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, intermediate_size=256, rope_theta=1e6,
        tie_word_embeddings=False,
    )
    cfg = config_from_qwen(
        src_cfg, attention_period=3, mamba_d_state=16, mamba_headdim=32,
        mamba_ngroups=2, mamba_chunk_size=8,
    )
    model = HybridForCausalLM(cfg)
    plan = plan_graft(cfg, 6)
    assert len(plan.kept_attention) == 2
    assert len(plan.dropped_attention) == 4

    # Build a fake Qwen state dict with the right shapes.
    state = {
        "model.embed_tokens.weight": torch.randn(512, 128),
        "model.norm.weight": torch.randn(128),
        "lm_head.weight": torch.randn(512, 128),
    }
    for i in range(6):
        state[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(128, 128)
        state[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(64, 128)
        state[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(64, 128)
        state[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(128, 128)
        state[f"model.layers.{i}.self_attn.q_norm.weight"] = torch.randn(32)
        state[f"model.layers.{i}.self_attn.k_norm.weight"] = torch.randn(32)
        state[f"model.layers.{i}.input_layernorm.weight"] = torch.randn(128)
        state[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.randn(128)
        state[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(256, 128)
        state[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(256, 128)
        state[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(128, 256)

    report = graft(model, state, plan)
    assert report["tensors_skipped"] == []
    assert torch.equal(model.model.embed_tokens.weight, state["model.embed_tokens.weight"])
    assert 0.0 < report["reuse_fraction"] < 1.0

    # a kept attention block really got Qwen's q_proj
    attn_blocks = [(i, b) for i, b in enumerate(model.model.layers) if b.kind == "attention"]
    blk_i, blk = attn_blocks[0]
    src = dict((i, s) for i, k, s in plan.assignments)[blk_i]
    assert torch.equal(blk.mixer.q_proj.weight, state[f"model.layers.{src}.self_attn.q_proj.weight"])
