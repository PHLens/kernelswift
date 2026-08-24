import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mhc_head_compute_mix_backward_kernel(
    input_mix_ptr,    # [R, 4] fp32 contiguous (R = batch0 * batch1)
    grad_out_ptr,     # [R, 4] fp32 contiguous
    mhc_scale_ptr,    # [1] fp32
    mhc_base_ptr,     # [4] fp32
    grad_input_mix_ptr,  # [R, 4] fp32 output
    grad_mhc_base_ptr,   # [4] fp32 output (atomic accumulate)
    grad_mhc_scale_ptr,  # [1] fp32 output (atomic accumulate)
    R,                # total rows = batch0 * batch1
    BLOCK: tl.constexpr,  # rows per program
):
    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)   # [BLOCK]
    rm = rows < R
    cols = tl.arange(0, 4)                      # [4]

    offs = rows[:, None] * 4 + cols[None, :]    # [BLOCK, 4]
    m = rm[:, None]

    # ---- load ----
    im = tl.load(input_mix_ptr + offs, mask=m, other=0.0)
    go = tl.load(grad_out_ptr + offs, mask=m, other=0.0)
    scale = tl.load(mhc_scale_ptr + 0)                          # scalar
    base = tl.load(mhc_base_ptr + cols[None, :])                # [1,4] broadcast

    # ---- sigmoid backward chain ----
    z = im * scale + base
    sig = tl.sigmoid(z)
    gz = go * sig * (1.0 - sig)
    gim = gz * scale
    tl.store(grad_input_mix_ptr + offs, gim, mask=m)

    # ---- on-chip reductions ----
    # grad_mhc_base = sum(gz, dim=(0,1)) -> [4] (sum over rows, keep last dim)
    base_partial = tl.sum(gz, axis=0)        # [4]
    # grad_mhc_scale = sum(gz * input_mix) full reduction -> scalar
    scale_partial = tl.sum(gz * im)          # scalar

    tl.atomic_add(grad_mhc_base_ptr + cols, base_partial)
    tl.atomic_add(grad_mhc_scale_ptr + 0, scale_partial)


class ModelNew(nn.Module):
    """
    Manual backward of mhc_head_compute_mix (sigmoid-gated affine modulation),
    fused into a single Triton kernel.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input_mix, mhc_scale, mhc_base, grad_out):
        b0, b1, mult = input_mix.shape
        R = b0 * b1

        im = input_mix.contiguous().view(R, mult)
        go = grad_out.contiguous().view(R, mult)
        scale = mhc_scale.contiguous().view(-1)
        base = mhc_base.contiguous().view(-1)

        grad_input_mix = torch.empty((R, mult), dtype=torch.float32, device=im.device)
        grad_mhc_base = torch.zeros(mult, dtype=torch.float32, device=im.device)
        grad_mhc_scale = torch.zeros(1, dtype=torch.float32, device=im.device)

        BLOCK = 128
        grid = (triton.cdiv(R, BLOCK),)
        _mhc_head_compute_mix_backward_kernel[grid](
            im, go, scale, base,
            grad_input_mix, grad_mhc_base, grad_mhc_scale,
            R, BLOCK=BLOCK,
        )

        return (
            grad_input_mix.view(b0, b1, mult),
            grad_mhc_scale.view(1),
            grad_mhc_base.view(mult),
        )


def get_inputs():
    batch0 = 2
    batch1 = 1024
    mhc_mult = 4
    input_mix = torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32, device="cuda")
    mhc_scale = torch.randn(1, dtype=torch.float32, device="cuda")
    mhc_base = torch.randn(mhc_mult, dtype=torch.float32, device="cuda")
    grad_out = torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32, device="cuda")
    return [input_mix, mhc_scale, mhc_base, grad_out]


def get_init_inputs():
    return []
