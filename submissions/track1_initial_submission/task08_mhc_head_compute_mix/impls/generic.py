"""mhc_head_compute_mix: split a mix vector into pre/post gates and a Sinkhorn
doubly-stochastic comb matrix.

Semantics (matches base.py exactly):
  pre  = sigmoid(x[:, :hc]  * s0 + base[:hc]) + eps
  post = 2 * sigmoid(x[:, hc:2hc] * s1 + base[hc:2hc])
  comb = raw.view(-1, hc, hc) * s2 + base[2hc:].view(1, hc, hc)
  comb = exp(comb - row_max)
  comb = comb / row_sum + eps
  comb = comb / (col_sum + eps)
  repeat (sinkhorn_iters - 1) times: row normalize, col normalize

Each (batch, seq) row is independent; a single Triton program handles one row
(4x4 comb + 4 pre + 4 post). hc is a compile-time constant (4), so the comb is
fully unrolled and the Sinkhorn loop is a bounded static loop.
"""

from __future__ import annotations

# Vendor-neutral generic Triton fallback.

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mhc_mix_kernel(
    mixes_ptr,      # [b*s, mix_hc] fp32
    hc_scale_ptr,   # [3] fp32
    hc_base_ptr,    # [mix_hc] fp32
    pre_ptr,        # [b*s, hc] fp32
    post_ptr,       # [b*s, hc] fp32
    comb_ptr,       # [b*s, hc, hc] fp32
    eps,
    hc: tl.constexpr,
    mix_hc: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, hc)           # [hc]
    ij = tl.arange(0, hc * hc)     # [hc*hc]

    s0 = tl.load(hc_scale_ptr + 0)
    s1 = tl.load(hc_scale_ptr + 1)
    s2 = tl.load(hc_scale_ptr + 2)

    # --- pre / post gates ---
    x_pre = tl.load(mixes_ptr + row * mix_hc + i)
    x_post = tl.load(mixes_ptr + row * mix_hc + hc + i)
    b_pre = tl.load(hc_base_ptr + i)
    b_post = tl.load(hc_base_ptr + hc + i)

    pre = 1.0 / (1.0 + tl.exp(-(x_pre * s0 + b_pre))) + eps
    post = 2.0 * (1.0 / (1.0 + tl.exp(-(x_post * s1 + b_post))))

    # --- comb: [hc, hc] doubly-stochastic via Sinkhorn ---
    x_comb = tl.load(mixes_ptr + row * mix_hc + 2 * hc + ij)   # [hc*hc]
    b_comb = tl.load(hc_base_ptr + 2 * hc + ij)                 # [hc*hc]
    comb = (x_comb * s2 + b_comb).reshape(hc, hc)               # [hc, hc]

    row_max = tl.max(comb, axis=1)                              # [hc]
    comb = tl.exp(comb - row_max[:, None])

    row_sum = tl.sum(comb, axis=1)                              # [hc]
    comb = comb / row_sum[:, None] + eps
    col_sum = tl.sum(comb, axis=0)                              # [hc]
    comb = comb / (col_sum[None, :] + eps)

    for _ in range(sinkhorn_iters - 1):
        row_sum = tl.sum(comb, axis=1)
        comb = comb / (row_sum[:, None] + eps)
        col_sum = tl.sum(comb, axis=0)
        comb = comb / (col_sum[None, :] + eps)

    tl.store(pre_ptr + row * hc + i, pre)
    tl.store(post_ptr + row * hc + i, post)
    tl.store(comb_ptr + row * hc * hc + ij, comb.reshape(hc * hc))


class ModelNew(nn.Module):
    def __init__(self, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
        super().__init__()
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

    def forward(
        self,
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        b, s, mix_hc = mixes.shape
        hc = self.hc_mult
        n = b * s

        pre = torch.empty((n, hc), dtype=torch.float32, device=mixes.device)
        post = torch.empty((n, hc), dtype=torch.float32, device=mixes.device)
        comb = torch.empty((n, hc, hc), dtype=torch.float32, device=mixes.device)

        mixes_f = mixes.reshape(-1, mix_hc).to(torch.float32)
        hc_scale_f = hc_scale.to(torch.float32)
        hc_base_f = hc_base.to(torch.float32)

        _mhc_mix_kernel[(n,)](
            mixes_f,
            hc_scale_f,
            hc_base_f,
            pre,
            post,
            comb,
            self.eps,
            hc=hc,
            mix_hc=mix_hc,
            sinkhorn_iters=self.sinkhorn_iters,
            num_warps=1,
        )

        return pre.view(b, s, hc), post.view(b, s, hc), comb.view(b, s, hc, hc)


def get_init_inputs():
    return [4, 20, 1e-6]


def get_inputs():
    hc = 4
    mix_hc = (2 + hc) * hc
    mixes = torch.randn(2, 8, mix_hc, dtype=torch.float32, device="cuda")
    hc_scale = torch.tensor([0.5, 0.25, 1.0], dtype=torch.float32, device="cuda")
    hc_base = torch.randn(mix_hc, dtype=torch.float32, device="cuda") * 0.1
    return [mixes, hc_scale, hc_base]


if __name__ == "__main__":
    model = ModelNew(*get_init_inputs()).eval()
    with torch.no_grad():
        outs = model(*get_inputs())
    for o in outs:
        print(o.shape)
