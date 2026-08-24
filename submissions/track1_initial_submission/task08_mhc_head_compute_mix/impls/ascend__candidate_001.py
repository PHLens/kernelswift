"""Round 001 candidate: fused single-kernel Sinkhorn head-compute-mix.

Collapses the entire forward (sigmoid gates, row-stabilized softmax, and the
20-iteration Sinkhorn row/column normalization loop) into one Triton kernel with
an internal ``tl.static_range`` loop, reducing 136 per-call kernel launches to 1.

Grid: 16 programs, one per (b, s) row (b*s = 2*8 = 16). Each program loads its
24-float ``mixes`` row and the shared 24-float ``hc_base`` / 3-float ``hc_scale``,
computes ``pre`` and ``post`` gates, builds and row-stabilizes the 4x4 combination
matrix, and runs the 20 row + 20 column normalizations inside a single compile-time
loop. Numerical semantics match base.py exactly.
"""

import torch
import torch.nn as nn
import torch_npu  # noqa: F401 - registers the Ascend NPU device
import triton
import triton.language as tl


@triton.jit
def _mhc_head_compute_mix_kernel(
    mixes_ptr,      # [B*S, 24] fp32 (row-major, contiguous)
    hc_scale_ptr,   # [3] fp32 (s0, s1, s2)
    hc_base_ptr,    # [24] fp32
    out_pre_ptr,    # [B*S, 4] fp32
    out_post_ptr,   # [B*S, 4] fp32
    out_comb_ptr,   # [B*S, 4, 4] fp32 (row-major, contiguous)
    HC: tl.constexpr,            # 4
    SINKHORN_ITERS: tl.constexpr,  # 20
    EPS: tl.constexpr,           # 1e-6
):
    pid = tl.program_id(0)

    hc_idx = tl.arange(0, HC)              # [4]
    hc2_idx = tl.arange(0, HC * HC)        # [16]

    # Load the per-row mixes slice: x[pid, :] -> [24]
    x = tl.load(mixes_ptr + pid * (2 * HC + HC * HC) + tl.arange(0, 2 * HC + HC * HC))
    x = x.to(tl.float32)

    # Shared scale/base.
    base = tl.load(hc_base_ptr + tl.arange(0, 2 * HC + HC * HC)).to(tl.float32)
    s0 = tl.load(hc_scale_ptr + 0).to(tl.float32)
    s1 = tl.load(hc_scale_ptr + 1).to(tl.float32)
    s2 = tl.load(hc_scale_ptr + 2).to(tl.float32)

    # --- pre gate: sigmoid(x[:,:4]*s0 + base[:4]) + eps ---
    x_pre = tl.load(mixes_ptr + pid * (2 * HC + HC * HC) + tl.arange(0, HC)).to(tl.float32)
    b_pre = tl.load(hc_base_ptr + tl.arange(0, HC)).to(tl.float32)
    pre = tl.sigmoid(x_pre * s0 + b_pre) + EPS

    # --- post gate: 2 * sigmoid(x[:,4:8]*s1 + base[4:8])  (no +eps) ---
    x_post = tl.load(mixes_ptr + pid * (2 * HC + HC * HC) + HC + tl.arange(0, HC)).to(tl.float32)
    b_post = tl.load(hc_base_ptr + HC + tl.arange(0, HC)).to(tl.float32)
    post = 2.0 * tl.sigmoid(x_post * s1 + b_post)

    # --- comb matrix: reshape(x[:,8:24], 4,4) * s2 + base[8:24].view(1,4,4) ---
    # Load raw 16 floats [8:24] and base[8:24].
    raw = tl.load(
        mixes_ptr + pid * (2 * HC + HC * HC) + 2 * HC + tl.arange(0, HC * HC)
    ).to(tl.float32)  # [16]
    b_comb = tl.load(hc_base_ptr + 2 * HC + tl.arange(0, HC * HC)).to(tl.float32)  # [16]

    # Reshape to (4,4) and broadcast base (4,4) -> no explicit broadcast needed.
    comb = tl.reshape(raw, (HC, HC)) * s2 + tl.reshape(b_comb, (HC, HC))

    # Row-stabilized softmax over last axis (axis=1 -> row sums).
    row_max = tl.max(comb, axis=1)                       # [4] per-row max
    comb = tl.exp(comb - row_max[:, None])
    comb = comb / tl.sum(comb, axis=1)[:, None] + EPS     # softmax row + eps

    # First column normalization (axis=0 -> column sums).
    comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)

    # Sinkhorn iterations: sinkhorn_iters - 1 = 19 more row/column pairs.
    for _ in tl.static_range(0, SINKHORN_ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + EPS)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)

    # Store outputs.
    tl.store(out_pre_ptr + pid * HC + hc_idx, pre)
    tl.store(out_post_ptr + pid * HC + hc_idx, post)
    # comb flat store: row-major (i*HC + j).
    tl.store(out_comb_ptr + pid * HC * HC + hc2_idx, tl.reshape(comb, (HC * HC,)))


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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, mix_hc = mixes.shape
        hc = self.hc_mult
        eps = self.eps
        expected = (2 + hc) * hc
        if mix_hc != expected:
            raise ValueError(f"expected mix dim {expected}, got {mix_hc}")

        # mixes -> [b*s, mix_hc] fp32 contiguous.
        x = mixes.reshape(-1, mix_hc).to(dtype=torch.float32).contiguous()
        scale = hc_scale.to(dtype=torch.float32).contiguous()
        base = hc_base.to(dtype=torch.float32).contiguous()

        n_rows = b * s
        out_pre = torch.empty((n_rows, hc), dtype=torch.float32, device=x.device)
        out_post = torch.empty((n_rows, hc), dtype=torch.float32, device=x.device)
        out_comb = torch.empty((n_rows, hc, hc), dtype=torch.float32, device=x.device)

        grid = (n_rows,)
        _mhc_head_compute_mix_kernel[grid](
            x,
            scale,
            base,
            out_pre,
            out_post,
            out_comb,
            HC=hc,
            SINKHORN_ITERS=self.sinkhorn_iters,
            EPS=eps,
            num_warps=1,
        )

        return (
            out_pre.view(b, s, hc),
            out_post.view(b, s, hc),
            out_comb.view(b, s, hc, hc),
        )


def get_init_inputs():
    """Returns positional args for Model.__init__: (hc_mult, sinkhorn_iters, eps)."""
    return [4, 20, 1e-6]


def get_inputs():
    """Returns positional args for Model.forward: (mixes, hc_scale, hc_base)."""
    hc = 4
    mix_hc = (2 + hc) * hc
    torch.manual_seed(0)
    mixes = torch.randn(2, 8, mix_hc, dtype=torch.float32, device="npu")
    hc_scale = torch.tensor([0.5, 0.25, 1.0], dtype=torch.float32, device="npu")
    hc_base = torch.randn(mix_hc, dtype=torch.float32, device="npu") * 0.1
    return [mixes, hc_scale, hc_base]
