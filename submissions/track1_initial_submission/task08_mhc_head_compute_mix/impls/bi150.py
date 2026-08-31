import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mhc_head_compute_mix_kernel(
    mixes_ptr,          # [2,8,24] fp32, contiguous, viewed as [16,24]
    hc_scale_ptr,       # [3] fp32
    hc_base_ptr,        # [24] fp32
    pre_ptr,            # [16,4] fp32 output (pre)
    post_ptr,           # [16,4] fp32 output (post)
    comb_ptr,           # [16,4,4] fp32 output (comb)
    eps: tl.constexpr,  # float
    HC: tl.constexpr,   # 4
    ITERS: tl.constexpr,  # sinkhorn_iters - 1 = 19
):
    n = tl.program_id(0)  # 0..15, one program per (b,s) position

    idx = tl.arange(0, 4)  # [4]

    # ---- load hc_scale scalars ----
    s0 = tl.load(hc_scale_ptr + 0)
    s1 = tl.load(hc_scale_ptr + 1)
    s2 = tl.load(hc_scale_ptr + 2)

    # ---- load the [24]-vector for this element: x = mixes[n, :] ----
    base_off = n * (2 * HC + HC * HC)  # = n * 24
    x_pre = tl.load(mixes_ptr + base_off + idx)                    # [0:4]
    x_post = tl.load(mixes_ptr + base_off + HC + idx)              # [4:8]

    # base slices
    base_pre = tl.load(hc_base_ptr + idx)                          # base[0:4]
    base_post = tl.load(hc_base_ptr + HC + idx)                    # base[4:8]

    # ---- pre = sigmoid(x0*s0 + base0) + eps ----
    pre = tl.sigmoid(x_pre * s0 + base_pre) + eps
    tl.store(pre_ptr + n * HC + idx, pre)

    # ---- post = 2 * sigmoid(x1*s1 + base1) ----
    post = 2.0 * tl.sigmoid(x_post * s1 + base_post)
    tl.store(post_ptr + n * HC + idx, post)

    # ---- comb [4,4] tile loaded into registers ----
    rows = idx[:, None]          # [4,1]
    cols = idx[None, :]          # [1,4]
    comb_base_off = 2 * HC       # = 8
    comb_x = tl.load(mixes_ptr + base_off + comb_base_off + rows * HC + cols)   # [4,4]
    comb_b = tl.load(hc_base_ptr + comb_base_off + rows * HC + cols)            # [4,4]
    comb = comb_x * s2 + comb_b                                                 # [4,4]

    # ---- stable softmax: exp(comb - row_max) ----
    row_max = tl.max(comb, axis=1, keep_dims=True)      # [4,1]
    comb = tl.exp(comb - row_max)                       # [4,4]

    # ---- first explicit normalization pair (eps asymmetric placement) ----
    # row-normalize: comb = comb / row_sum + eps  (eps added to MATRIX)
    row_sum = tl.sum(comb, axis=1, keep_dims=True)      # [4,1]
    comb = comb / row_sum + eps
    # col-normalize: comb = comb / (col_sum + eps)  (eps added to DENOMINATOR)
    col_sum = tl.sum(comb, axis=0, keep_dims=True)      # [1,4]
    comb = comb / (col_sum + eps)

    # ---- looped normalization (ITERS = 19 rounds), eps always in denominator ----
    # NOTE: tl.static_range(ITERS=19) is a compile-time unroll the BI150 / CoreX
    # Triton 3.1.0 compiler cannot complete in bounded time (4 iters ~instant,
    # 8 iters ~100s, 19 iters >300s). We use the dynamic tl.range(ITERS) loop,
    # which preserves the exact 19-round semantics (same body, same eps placement,
    # same count) while keeping the [4,4] comb tile in registers.
    for _ in tl.range(ITERS):
        row_sum = tl.sum(comb, axis=1, keep_dims=True)  # [4,1]
        comb = comb / (row_sum + eps)
        col_sum = tl.sum(comb, axis=0, keep_dims=True)  # [1,4]
        comb = comb / (col_sum + eps)

    # ---- store comb [4,4] ----
    tl.store(comb_ptr + n * HC * HC + rows * HC + cols, comb)


class ModelNew(nn.Module):
    def __init__(self, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
        super().__init__()
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

    def forward(self, mixes, hc_scale, hc_base):
        b, s, mix_hc = mixes.shape
        hc = self.hc_mult
        eps = self.eps
        expected = (2 + hc) * hc
        if mix_hc != expected:
            raise ValueError(f"expected mix dim {expected}, got {mix_hc}")

        x = mixes.reshape(-1, mix_hc).to(dtype=torch.float32)
        base = hc_base.to(dtype=torch.float32)
        scale = hc_scale.to(dtype=torch.float32)

        n = x.shape[0]  # b * s = 16

        pre = torch.empty((n, hc), dtype=torch.float32, device=x.device)
        post = torch.empty((n, hc), dtype=torch.float32, device=x.device)
        comb = torch.empty((n, hc, hc), dtype=torch.float32, device=x.device)

        iters = self.sinkhorn_iters - 1
        _mhc_head_compute_mix_kernel[(n,)](
            x, scale, base, pre, post, comb,
            eps=eps, HC=hc, ITERS=iters,
        )

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
