import torch
import torch.nn as nn
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _mhc_mix_bwd_fused_kernel(
    input_mix_ptr,
    mhc_scale_ptr,
    mhc_base_ptr,
    grad_out_ptr,
    grad_input_mix_ptr,
    grad_mhc_base_ptr,
    grad_mhc_scale_ptr,
    NROWS,
    BLOCK_R: tl.constexpr,
):
    # 1D grid over row tiles of the [2048, 4] view (2*1024 = 2048 rows).
    # Each program loads a [BLOCK_R, 4] tile, computes the full sigmoid backward
    # elementwise chain, stores grad_input_mix, then reduces the block-local
    # partials (per-column sum -> [4]; full sum -> scalar) and atomically
    # accumulates them into the tiny [4] and [1] outputs.
    pid = tl.program_id(0)
    row_start = pid * BLOCK_R
    rows = row_start + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, 4)

    mask = rows < NROWS

    # Row-major [2048, 4]: flat index = row * 4 + col.
    offs = rows[:, None] * 4 + cols[None, :]

    input_mix = tl.load(input_mix_ptr + offs, mask=mask[:, None], other=0.0)
    grad_out = tl.load(grad_out_ptr + offs, mask=mask[:, None], other=0.0)

    # mhc_scale: [1] scalar broadcast; mhc_base: [4] broadcast along columns.
    mhc_scale = tl.load(mhc_scale_ptr)
    mhc_base = tl.load(mhc_base_ptr + cols)

    z = input_mix * mhc_scale + mhc_base
    sig = tl.sigmoid(z)
    grad_z = grad_out * sig * (1.0 - sig)
    grad_input_mix = grad_z * mhc_scale

    tl.store(grad_input_mix_ptr + offs, grad_input_mix, mask=mask[:, None])

    # Block-local partials. Both masked lanes are zero (other=0.0) so the sum is
    # exact regardless of mask.
    base_partial = tl.sum(grad_z, axis=0)          # [4]  per-column over rows
    scale_partial = tl.sum(grad_z * input_mix)     # scalar full sum

    tl.atomic_add(grad_mhc_base_ptr + cols, base_partial)
    tl.atomic_add(grad_mhc_scale_ptr, scale_partial)


class ModelNew(nn.Module):
    """
    Model that computes manual backward of mhc_head_compute_mix (fused Triton).
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        input_mix: torch.Tensor,
        mhc_scale: torch.Tensor,
        mhc_base: torch.Tensor,
        grad_out: torch.Tensor,
    ):
        device = input_mix.device

        input_mix_c = input_mix.contiguous()
        grad_out_c = grad_out.contiguous()
        mhc_scale_c = mhc_scale.contiguous()
        mhc_base_c = mhc_base.contiguous()

        n0, n1, mhc_mult = input_mix_c.shape
        nrows = n0 * n1

        grad_input_mix = torch.empty_like(input_mix_c)
        grad_mhc_base = torch.zeros((mhc_mult,), dtype=torch.float32, device=device)
        grad_mhc_scale = torch.zeros((1,), dtype=torch.float32, device=device)

        BLOCK_R = 64
        grid = (triton.cdiv(nrows, BLOCK_R),)
        _mhc_mix_bwd_fused_kernel[grid](
            input_mix_c,
            mhc_scale_c,
            mhc_base_c,
            grad_out_c,
            grad_input_mix,
            grad_mhc_base,
            grad_mhc_scale,
            nrows,
            BLOCK_R=BLOCK_R,
            num_warps=4,
        )

        return (
            grad_input_mix.reshape(n0, n1, mhc_mult),
            grad_mhc_scale,
            grad_mhc_base,
        )


batch0 = 2
batch1 = 1024
mhc_mult = 4


def get_inputs():
    input_mix = torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32, device="npu")
    mhc_scale = torch.randn(1, dtype=torch.float32, device="npu")
    mhc_base = torch.randn(mhc_mult, dtype=torch.float32, device="npu")
    grad_out = torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32, device="npu")

    return [input_mix, mhc_scale, mhc_base, grad_out]


def get_init_inputs():
    return []
