import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mhc_head_compute_mix_kernel(
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    hc: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    eps: tl.constexpr,
):
    # One program per (b, s) position. mixes is [2, 8, 24] -> 16 rows of 24
    # contiguous fp32 elements. hc == 4 (compile-time constant). All small dims
    # are handled with explicit scalar loads and manual reductions.
    pos = tl.program_id(0)
    row_base = pos * (2 * hc + hc * hc)

    s0 = tl.load(hc_scale_ptr + 0)
    s1 = tl.load(hc_scale_ptr + 1)
    s2 = tl.load(hc_scale_ptr + 2)

    # ---- pre: pre[i] = sigmoid(x[i]*s0 + base[i]) + eps ----
    # ---- post: post[i] = 2 * sigmoid(x[hc+i]*s1 + base[hc+i]) ----
    for i in tl.static_range(hc):
        xp = tl.load(mixes_ptr + row_base + i)
        bp = tl.load(hc_base_ptr + i)
        sig = 1.0 / (1.0 + tl.exp(-(xp * s0 + bp)))
        tl.store(pre_ptr + pos * hc + i, sig + eps)

        xq = tl.load(mixes_ptr + row_base + hc + i)
        bq = tl.load(hc_base_ptr + hc + i)
        sig2 = 1.0 / (1.0 + tl.exp(-(xq * s1 + bq)))
        tl.store(post_ptr + pos * hc + i, 2.0 * sig2)

    # ---- comb[j*4+i] = x[8 + j*4+i]*s2 + base[8 + j*4+i], then exp(-row_max) ----
    c00 = tl.load(mixes_ptr + row_base + 8 + 0) * s2 + tl.load(hc_base_ptr + 8 + 0)
    c01 = tl.load(mixes_ptr + row_base + 8 + 1) * s2 + tl.load(hc_base_ptr + 8 + 1)
    c02 = tl.load(mixes_ptr + row_base + 8 + 2) * s2 + tl.load(hc_base_ptr + 8 + 2)
    c03 = tl.load(mixes_ptr + row_base + 8 + 3) * s2 + tl.load(hc_base_ptr + 8 + 3)

    c10 = tl.load(mixes_ptr + row_base + 8 + 4) * s2 + tl.load(hc_base_ptr + 8 + 4)
    c11 = tl.load(mixes_ptr + row_base + 8 + 5) * s2 + tl.load(hc_base_ptr + 8 + 5)
    c12 = tl.load(mixes_ptr + row_base + 8 + 6) * s2 + tl.load(hc_base_ptr + 8 + 6)
    c13 = tl.load(mixes_ptr + row_base + 8 + 7) * s2 + tl.load(hc_base_ptr + 8 + 7)

    c20 = tl.load(mixes_ptr + row_base + 8 + 8) * s2 + tl.load(hc_base_ptr + 8 + 8)
    c21 = tl.load(mixes_ptr + row_base + 8 + 9) * s2 + tl.load(hc_base_ptr + 8 + 9)
    c22 = tl.load(mixes_ptr + row_base + 8 + 10) * s2 + tl.load(hc_base_ptr + 8 + 10)
    c23 = tl.load(mixes_ptr + row_base + 8 + 11) * s2 + tl.load(hc_base_ptr + 8 + 11)

    c30 = tl.load(mixes_ptr + row_base + 8 + 12) * s2 + tl.load(hc_base_ptr + 8 + 12)
    c31 = tl.load(mixes_ptr + row_base + 8 + 13) * s2 + tl.load(hc_base_ptr + 8 + 13)
    c32 = tl.load(mixes_ptr + row_base + 8 + 14) * s2 + tl.load(hc_base_ptr + 8 + 14)
    c33 = tl.load(mixes_ptr + row_base + 8 + 15) * s2 + tl.load(hc_base_ptr + 8 + 15)

    # row_max and exp
    m0 = tl.maximum(tl.maximum(c00, c01), tl.maximum(c02, c03))
    m1 = tl.maximum(tl.maximum(c10, c11), tl.maximum(c12, c13))
    m2 = tl.maximum(tl.maximum(c20, c21), tl.maximum(c22, c23))
    m3 = tl.maximum(tl.maximum(c30, c31), tl.maximum(c32, c33))

    c00 = tl.exp(c00 - m0)
    c01 = tl.exp(c01 - m0)
    c02 = tl.exp(c02 - m0)
    c03 = tl.exp(c03 - m0)
    c10 = tl.exp(c10 - m1)
    c11 = tl.exp(c11 - m1)
    c12 = tl.exp(c12 - m1)
    c13 = tl.exp(c13 - m1)
    c20 = tl.exp(c20 - m2)
    c21 = tl.exp(c21 - m2)
    c22 = tl.exp(c22 - m2)
    c23 = tl.exp(c23 - m2)
    c30 = tl.exp(c30 - m3)
    c31 = tl.exp(c31 - m3)
    c32 = tl.exp(c32 - m3)
    c33 = tl.exp(c33 - m3)

    # ---- FIRST row normalize: comb = comb / row_sum + eps (eps AFTER division) ----
    rs0 = c00 + c01 + c02 + c03
    rs1 = c10 + c11 + c12 + c13
    rs2 = c20 + c21 + c22 + c23
    rs3 = c30 + c31 + c32 + c33
    c00 = c00 / rs0 + eps
    c01 = c01 / rs0 + eps
    c02 = c02 / rs0 + eps
    c03 = c03 / rs0 + eps
    c10 = c10 / rs1 + eps
    c11 = c11 / rs1 + eps
    c12 = c12 / rs1 + eps
    c13 = c13 / rs1 + eps
    c20 = c20 / rs2 + eps
    c21 = c21 / rs2 + eps
    c22 = c22 / rs2 + eps
    c23 = c23 / rs2 + eps
    c30 = c30 / rs3 + eps
    c31 = c31 / rs3 + eps
    c32 = c32 / rs3 + eps
    c33 = c33 / rs3 + eps

    # ---- FIRST col normalize: comb = comb / (col_sum + eps) ----
    cs0 = c00 + c10 + c20 + c30
    cs1 = c01 + c11 + c21 + c31
    cs2 = c02 + c12 + c22 + c32
    cs3 = c03 + c13 + c23 + c33
    c00 = c00 / (cs0 + eps)
    c10 = c10 / (cs0 + eps)
    c20 = c20 / (cs0 + eps)
    c30 = c30 / (cs0 + eps)
    c01 = c01 / (cs1 + eps)
    c11 = c11 / (cs1 + eps)
    c21 = c21 / (cs1 + eps)
    c31 = c31 / (cs1 + eps)
    c02 = c02 / (cs2 + eps)
    c12 = c12 / (cs2 + eps)
    c22 = c22 / (cs2 + eps)
    c32 = c32 / (cs2 + eps)
    c03 = c03 / (cs3 + eps)
    c13 = c13 / (cs3 + eps)
    c23 = c23 / (cs3 + eps)
    c33 = c33 / (cs3 + eps)

    # ---- remaining (sinkhorn_iters - 1) = 19 row/col pairs ----
    for _ in tl.static_range(sinkhorn_iters - 1):
        rs0 = c00 + c01 + c02 + c03
        rs1 = c10 + c11 + c12 + c13
        rs2 = c20 + c21 + c22 + c23
        rs3 = c30 + c31 + c32 + c33
        c00 = c00 / (rs0 + eps)
        c01 = c01 / (rs0 + eps)
        c02 = c02 / (rs0 + eps)
        c03 = c03 / (rs0 + eps)
        c10 = c10 / (rs1 + eps)
        c11 = c11 / (rs1 + eps)
        c12 = c12 / (rs1 + eps)
        c13 = c13 / (rs1 + eps)
        c20 = c20 / (rs2 + eps)
        c21 = c21 / (rs2 + eps)
        c22 = c22 / (rs2 + eps)
        c23 = c23 / (rs2 + eps)
        c30 = c30 / (rs3 + eps)
        c31 = c31 / (rs3 + eps)
        c32 = c32 / (rs3 + eps)
        c33 = c33 / (rs3 + eps)

        cs0 = c00 + c10 + c20 + c30
        cs1 = c01 + c11 + c21 + c31
        cs2 = c02 + c12 + c22 + c32
        cs3 = c03 + c13 + c23 + c33
        c00 = c00 / (cs0 + eps)
        c10 = c10 / (cs0 + eps)
        c20 = c20 / (cs0 + eps)
        c30 = c30 / (cs0 + eps)
        c01 = c01 / (cs1 + eps)
        c11 = c11 / (cs1 + eps)
        c21 = c21 / (cs1 + eps)
        c31 = c31 / (cs1 + eps)
        c02 = c02 / (cs2 + eps)
        c12 = c12 / (cs2 + eps)
        c22 = c22 / (cs2 + eps)
        c32 = c32 / (cs2 + eps)
        c03 = c03 / (cs3 + eps)
        c13 = c13 / (cs3 + eps)
        c23 = c23 / (cs3 + eps)
        c33 = c33 / (cs3 + eps)

    # ---- store comb ----
    tl.store(comb_ptr + pos * 16 + 0, c00)
    tl.store(comb_ptr + pos * 16 + 1, c01)
    tl.store(comb_ptr + pos * 16 + 2, c02)
    tl.store(comb_ptr + pos * 16 + 3, c03)
    tl.store(comb_ptr + pos * 16 + 4, c10)
    tl.store(comb_ptr + pos * 16 + 5, c11)
    tl.store(comb_ptr + pos * 16 + 6, c12)
    tl.store(comb_ptr + pos * 16 + 7, c13)
    tl.store(comb_ptr + pos * 16 + 8, c20)
    tl.store(comb_ptr + pos * 16 + 9, c21)
    tl.store(comb_ptr + pos * 16 + 10, c22)
    tl.store(comb_ptr + pos * 16 + 11, c23)
    tl.store(comb_ptr + pos * 16 + 12, c30)
    tl.store(comb_ptr + pos * 16 + 13, c31)
    tl.store(comb_ptr + pos * 16 + 14, c32)
    tl.store(comb_ptr + pos * 16 + 15, c33)


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

        x = mixes.reshape(-1, mix_hc).to(dtype=torch.float32)
        base = hc_base.to(dtype=torch.float32)
        s0, s1, s2 = hc_scale[0], hc_scale[1], hc_scale[2]

        # Fast path: benchmark shape, fp32 contiguous cuda, no grad, hc==4.
        if (
            hc == 4
            and mix_hc == 24
            and self.sinkhorn_iters == 20
            and abs(float(eps) - 1e-6) < 1e-12
            and x.dtype == torch.float32
            and base.dtype == torch.float32
            and x.is_cuda
            and x.is_contiguous()
            and base.is_contiguous()
            and hc_scale.is_cuda
            and hc_scale.dtype == torch.float32
            and hc_scale.is_contiguous()
            and not torch.is_grad_enabled()
        ):
            n = x.shape[0]
            pre = torch.empty((n, hc), dtype=torch.float32, device=x.device)
            post = torch.empty((n, hc), dtype=torch.float32, device=x.device)
            comb = torch.empty((n, hc, hc), dtype=torch.float32, device=x.device)
            grid = (n,)
            _mhc_head_compute_mix_kernel[grid](
                x,
                hc_scale,
                base,
                pre,
                post,
                comb,
                hc=hc,
                sinkhorn_iters=self.sinkhorn_iters,
                eps=eps,
                num_warps=1,
            )
            return pre.view(b, s, hc), post.view(b, s, hc), comb.view(b, s, hc, hc)

        # Fallback: unchanged PyTorch reference path.
        pre = torch.sigmoid(x[:, :hc] * s0 + base[:hc].unsqueeze(0)) + eps
        post = 2 * torch.sigmoid(x[:, hc : 2 * hc] * s1 + base[hc : 2 * hc].unsqueeze(0))
        raw = x[:, 2 * hc : 2 * hc + hc * hc]
        comb = raw.view(-1, hc, hc) * s2 + base[2 * hc : 2 * hc + hc * hc].view(1, hc, hc)

        row_max = comb.amax(dim=-1, keepdim=True)
        comb = torch.exp(comb - row_max)
        comb = comb / comb.sum(dim=-1, keepdim=True) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

        return pre.view(b, s, hc), post.view(b, s, hc), comb.view(b, s, hc, hc)


def get_init_inputs():
    """Returns positional args for Model.__init__: (hc_mult, sinkhorn_iters, eps)."""
    return [4, 20, 1e-6]


def get_inputs():
    """Returns positional args for Model.forward: (mixes, hc_scale, hc_base)."""
    hc = 4
    mix_hc = (2 + hc) * hc
    torch.manual_seed(0)
    mixes = torch.randn(2, 8, mix_hc, dtype=torch.float32, device="cuda")
    hc_scale = torch.tensor([0.5, 0.25, 1.0], dtype=torch.float32, device="cuda")
    hc_base = torch.randn(mix_hc, dtype=torch.float32, device="cuda") * 0.1
    return [mixes, hc_scale, hc_base]
