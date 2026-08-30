"""mhc_post_layer_mix: combine per-head post layer mix with a residual comb.

Semantics (matches base.py exactly):
  term2 = einsum('abmn,abmc->abnc', comb_res_mix, residual.float())
  out   = (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()

Shapes: x[n0,n1,h] bf16, residual[n0,n1,m,h] bf16,
        post_layer_mix[n0,n1,m,1] fp32, comb_res_mix[n0,n1,m,m] fp32,
        out[n0,n1,m,h] bf16.

For each (a,b) position, out[m,h] = x[h]*post[m] + sum_k comb[m,k]*residual[k,h].
m (=mhc_mult=4) is a compile-time constant; h is tiled. The [m,m]@[m,BLOCK_H]
matmul is done with elementwise tl.sum (no tl.dot, which is Unknown on GCU).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_gcu  # noqa: F401 - registers the GCU device
import triton_gcu  # noqa: F401


@triton.jit
def _post_layer_mix_kernel(
    x_ptr,               # [n0,n1,h] bf16
    residual_ptr,        # [n0,n1,m,h] bf16
    post_layer_mix_ptr,  # [n0,n1,m,1] fp32
    comb_res_mix_ptr,    # [n0,n1,m,m] fp32
    out_ptr,             # [n0,n1,m,h] bf16
    h,
    m: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    ab = tl.program_id(0)
    k = tl.arange(0, m)          # [m]
    mm = k[:, None] * m + k[None, :]  # [m,m] comb offset
    hb = tl.arange(0, BLOCK_H)   # [BLOCK_H]

    comb = tl.load(comb_res_mix_ptr + ab * (m * m) + mm)       # [m,m] fp32
    post = tl.load(post_layer_mix_ptr + ab * m + k)            # [m] fp32

    for h0 in range(0, h, BLOCK_H):
        h_idx = h0 + hb
        h_mask = h_idx < h

        xv = tl.load(x_ptr + ab * h + h_idx, mask=h_mask, other=0.0).to(tl.float32)
        # residual block: [m, BLOCK_H]
        r_off = ab * m * h + k[:, None] * h + h_idx[None, :]
        r_block = tl.load(residual_ptr + r_off, mask=h_mask[None, :], other=0.0).to(tl.float32)

        # out[n, block] = sum_m comb[m, n] * r_block[m, block]  (comb^T @ residual)
        acc = tl.sum(comb[:, :, None] * r_block[:, None, :], axis=0)  # [m, BLOCK_H]

        out = acc + post[:, None] * xv[None, :]
        tl.store(
            out_ptr + ab * m * h + k[:, None] * h + h_idx[None, :],
            out.to(tl.bfloat16),
            mask=h_mask[None, :],
        )


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        n0, n1, h = x.shape
        m = comb_res_mix.shape[-1]
        num_ab = n0 * n1

        out = torch.empty((n0, n1, m, h), dtype=torch.bfloat16, device=x.device)

        BLOCK_H = 256
        grid = (num_ab,)
        _post_layer_mix_kernel[grid](
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            out,
            h,
            m=m,
            BLOCK_H=BLOCK_H,
            num_warps=1,
        )
        return out


n0 = 2
n1 = 4096
h = 1280
mhc_mult = 4


def get_inputs():
    x = torch.randn((n0, n1, h), dtype=torch.bfloat16, device="cuda")
    residual = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device="cuda")
    post_layer_mix = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float32, device="cuda")
    comb_res_mix = torch.randn((n0, n1, mhc_mult, mhc_mult), dtype=torch.float32, device="cuda")
    return [x, residual, post_layer_mix, comb_res_mix]


def get_init_inputs():
    return []


if __name__ == "__main__":
    model = ModelNew().eval()
    with torch.no_grad():
        out = model(*get_inputs())
    print(out.shape)
