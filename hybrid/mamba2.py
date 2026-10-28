"""Pure-PyTorch Mamba-2 (SSD) mixer.

This is a readable reference implementation of the state-space duality form
used by Mamba-2 and by NVIDIA's Nemotron-H hybrid. It runs on CPU with no
custom CUDA kernels, so it is correct-but-slow; the fused `mamba-ssm` kernels
are a drop-in replacement for :meth:`Mamba2Mixer.forward` on a GPU box.

Two execution paths are provided and are numerically equivalent:

* :meth:`forward`   - chunked parallel scan, O(L) work with O(L/chunk) sequential
                      steps. Used for training and prefill.
* :meth:`step`      - single-token recurrent update against a fixed-size state.
                      Used for decoding; this is where the hybrid's constant
                      memory / constant per-token cost comes from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HybridConfig


@dataclass
class MambaState:
    """Recurrent cache for one Mamba-2 layer.

    conv: (batch, conv_channels, d_conv - 1) rolling window for the depthwise conv.
    ssm:  (batch, nheads, headdim, d_state) the SSM hidden state.

    Note both are *independent of sequence length* - this is the whole point.
    """

    conv: torch.Tensor
    ssm: torch.Tensor

    def index_select(self, idx: torch.Tensor) -> "MambaState":
        return MambaState(self.conv.index_select(0, idx), self.ssm.index_select(0, idx))


def ssm_dtype(dtype: torch.dtype) -> torch.dtype:
    """Accumulation dtype for the recurrence.

    The SSM state is always held in at least fp32 regardless of the model's
    parameter dtype - a bf16 state loses the long-range decay after a few
    hundred steps. float64 models (used by the equivalence tests) keep f64.
    """
    return torch.float64 if dtype == torch.float64 else torch.float32


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment-sum: out[..., i, j] = sum(x[..., j+1 : i+1]) for i >= j, else -inf."""
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)                      # (..., T, T)
    strict = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~strict, 0)
    out = torch.cumsum(x, dim=-2)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    return out.masked_fill(~causal, -float("inf"))


def ssd_chunked(
    x: torch.Tensor,          # (b, l, h, p)
    dt: torch.Tensor,         # (b, l, h)
    A: torch.Tensor,          # (h,)   negative scalars
    B: torch.Tensor,          # (b, l, h, n)
    C: torch.Tensor,          # (b, l, h, n)
    chunk_size: int,
    initial_state: torch.Tensor | None = None,   # (b, h, p, n)
):
    """Chunked state-space-duality scan. Returns (y, final_state)."""
    b, l, h, p = x.shape
    n = B.size(-1)

    pad = (-l) % chunk_size
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))
    L = x.size(1)
    nc = L // chunk_size
    cs = chunk_size

    x = (x * dt.unsqueeze(-1)).view(b, nc, cs, h, p)
    B = B.reshape(b, nc, cs, h, n)
    C = C.reshape(b, nc, cs, h, n)
    dA = (dt * A).permute(0, 2, 1).reshape(b, h, nc, cs)          # (b, h, nc, cs)
    dA_cum = dA.cumsum(dim=-1)

    # 1. intra-chunk ("diagonal") contribution - the quadratic-attention form.
    decay = torch.exp(_segsum(dA))                                 # (b, h, nc, cs, cs)
    y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, decay, x)

    # 2. each chunk's contribution to the state carried out of it.
    decay_out = torch.exp(dA_cum[..., -1:] - dA_cum)               # (b, h, nc, cs)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_out, x)

    # 3. sequential recurrence over the (few) chunks.
    if initial_state is None:
        initial_state = torch.zeros_like(states[:, :1]).squeeze(1)
    states = torch.cat([initial_state.unsqueeze(1), states], dim=1)       # (b, nc+1, h, p, n)
    chunk_decay = torch.exp(_segsum(F.pad(dA_cum[..., -1], (1, 0))))      # (b, h, nc+1, nc+1)
    states = torch.einsum("bhzc,bchpn->bzhpn", chunk_decay, states)
    states, final_state = states[:, :-1], states[:, -1]

    # 4. inter-chunk ("off-diagonal") contribution.
    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, torch.exp(dA_cum))

    y = (y_diag + y_off).reshape(b, L, h, p)
    return y[:, :l], final_state


class Mamba2Mixer(nn.Module):
    """Mamba-2 sequence mixer, parameterised to drop into a transformer stack."""

    def __init__(self, cfg: HybridConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.d_model = cfg.hidden_size
        self.d_inner = cfg.d_inner
        self.d_state = cfg.mamba_d_state
        self.d_conv = cfg.mamba_d_conv
        self.headdim = cfg.mamba_headdim
        self.nheads = cfg.mamba_nheads
        self.ngroups = cfg.mamba_ngroups
        self.chunk_size = cfg.mamba_chunk_size

        # z | x | B | C | dt, produced in one GEMM.
        self.conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=cfg.mamba_proj_bias)

        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.d_conv,
            groups=self.conv_dim,
            padding=self.d_conv - 1,
            bias=cfg.mamba_conv_bias,
        )

        # dt bias initialised so softplus(dt_bias) spans [dt_min, dt_max].
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(cfg.mamba_dt_max) - math.log(cfg.mamba_dt_min))
            + math.log(cfg.mamba_dt_min)
        ).clamp(min=cfg.mamba_dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        # A is a per-head negative scalar (the "scalar-identity" structure of Mamba-2).
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.nheads + 1, dtype=torch.float32)))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

        self.norm = GatedRMSNorm(self.d_inner, eps=cfg.rms_norm_eps)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=cfg.mamba_proj_bias)

    # -- helpers ---------------------------------------------------------------
    def _split(self, zxbcdt: torch.Tensor):
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.conv_dim, self.nheads], dim=-1
        )
        return z, xBC, dt

    def _split_xBC(self, xBC: torch.Tensor):
        gn = self.ngroups * self.d_state
        return torch.split(xBC, [self.d_inner, gn, gn], dim=-1)

    def _expand_groups(self, t: torch.Tensor, b: int, l: int) -> torch.Tensor:
        """(b, l, ngroups*d_state) -> (b, l, nheads, d_state) by repeating groups."""
        t = t.view(b, l, self.ngroups, self.d_state)
        return t.repeat_interleave(self.nheads // self.ngroups, dim=2)

    def allocate_state(self, batch: int, device, dtype) -> MambaState:
        return MambaState(
            conv=torch.zeros(batch, self.conv_dim, self.d_conv - 1, device=device, dtype=dtype),
            ssm=torch.zeros(
                batch, self.nheads, self.headdim, self.d_state,
                device=device, dtype=ssm_dtype(dtype),
            ),
        )

    # -- parallel path -----------------------------------------------------------
    def forward(self, hidden: torch.Tensor, state: MambaState | None = None) -> torch.Tensor:
        """hidden: (b, l, d_model). If ``state`` is given it is updated in place
        (used so a prefill can be followed by :meth:`step` decoding)."""
        b, l, _ = hidden.shape
        z, xBC, dt = self._split(self.in_proj(hidden))

        conv_in = xBC.transpose(1, 2)                                   # (b, conv_dim, l)
        if state is not None:
            conv_in = torch.cat([state.conv.to(conv_in.dtype), conv_in], dim=-1)
            offset = self.d_conv - 1
        else:
            offset = 0
        conv_out = self.conv1d(conv_in)[..., offset : offset + l]
        xBC = F.silu(conv_out).transpose(1, 2)

        x, B, C = self._split_xBC(xBC)
        x = x.view(b, l, self.nheads, self.headdim)
        B = self._expand_groups(B, b, l)
        C = self._expand_groups(C, b, l)
        dt = F.softplus(dt + self.dt_bias)                              # (b, l, h)
        acc = ssm_dtype(hidden.dtype)
        A = -torch.exp(self.A_log.to(acc))                              # (h,)

        init = state.ssm.to(acc) if state is not None else None
        y, final = ssd_chunked(
            x.to(acc), dt.to(acc), A, B.to(acc), C.to(acc), self.chunk_size, init
        )
        y = y + x.to(acc) * self.D.to(acc).view(1, 1, -1, 1)            # D skip connection
        y = y.to(hidden.dtype).reshape(b, l, self.d_inner)

        if state is not None:
            state.ssm = final.to(state.ssm.dtype)
            tail = conv_in[..., -(self.d_conv - 1) :]
            if tail.size(-1) < self.d_conv - 1:
                tail = F.pad(tail, (self.d_conv - 1 - tail.size(-1), 0))
            state.conv = tail.to(state.conv.dtype)

        return self.out_proj(self.norm(y, z))

    # -- recurrent path ----------------------------------------------------------
    @torch.no_grad()
    def step(self, hidden: torch.Tensor, state: MambaState) -> torch.Tensor:
        """Single-token update. hidden: (b, 1, d_model). O(1) in sequence length."""
        b = hidden.size(0)
        z, xBC, dt = self._split(self.in_proj(hidden))
        z, xBC, dt = z.squeeze(1), xBC.squeeze(1), dt.squeeze(1)

        window = torch.cat([state.conv.to(xBC.dtype), xBC.unsqueeze(-1)], dim=-1)  # (b, cd, d_conv)
        state.conv = window[..., 1:].to(state.conv.dtype)
        conv_w = self.conv1d.weight.squeeze(1)                           # (conv_dim, d_conv)
        xBC = (window * conv_w).sum(dim=-1)
        if self.conv1d.bias is not None:
            xBC = xBC + self.conv1d.bias
        xBC = F.silu(xBC)

        x, B, C = self._split_xBC(xBC)
        acc = state.ssm.dtype
        x = x.view(b, self.nheads, self.headdim).to(acc)
        B = self._expand_groups(B.unsqueeze(1), b, 1).squeeze(1).to(acc)  # (b, h, n)
        C = self._expand_groups(C.unsqueeze(1), b, 1).squeeze(1).to(acc)
        dt = F.softplus(dt + self.dt_bias).to(acc)                        # (b, h)
        A = -torch.exp(self.A_log.to(acc))

        dA = torch.exp(dt * A).unsqueeze(-1).unsqueeze(-1)                # (b, h, 1, 1)
        dBx = torch.einsum("bh,bhn,bhp->bhpn", dt, B, x)
        state.ssm = state.ssm * dA + dBx
        y = torch.einsum("bhpn,bhn->bhp", state.ssm, C)
        y = y + x * self.D.to(acc).view(1, -1, 1)

        y = y.to(hidden.dtype).reshape(b, self.d_inner)
        return self.out_proj(self.norm(y, z)).unsqueeze(1)


class GatedRMSNorm(nn.Module):
    """RMSNorm with a multiplicative SiLU gate, as used inside the Mamba block."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        if gate is not None:
            x = x * F.silu(gate)
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight
