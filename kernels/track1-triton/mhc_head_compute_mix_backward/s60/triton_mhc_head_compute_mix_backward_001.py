"""mhc_head_compute_mix_backward: manual backward of mhc_head_compute_mix.

Elementwise sigmoid-backward is fused into a single Triton kernel; the two
reductions (grad_mhc_base, grad_mhc_scale) are computed with library torch ops
(the GCU CNNL reductions are already well-optimized and exact).

Semantics (matches base.py exactly):
  z = input_mix * mhc_scale + mhc_base
  sigmoid = sigmoid(z)
  grad_z = grad_out * sigmoid * (1 - sigmoid)
  grad_input_mix = grad_z * mhc_scale
  grad_mhc_base  = grad_z.sum(dim=(0,1)).view(-1)              # [mhc_mult]
  grad_mhc_scale = (grad_z * input_mix).sum().view(1)          # [1]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_gcu  # noqa: F401 - registers the GCU device
import triton_gcu  # noqa: F401


@triton.jit
def _sigmoid_backward_kernel(
    input_mix_ptr,
    mhc_scale_ptr,
    mhc_base_ptr,
    grad_out_ptr,
    grad_input_mix_ptr,
    grad_z_ptr,
    mhc_mult: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs % mhc_mult
    scale = tl.load(mhc_scale_ptr)  # scalar [1]
    base = tl.load(mhc_base_ptr + m)
    x = tl.load(input_mix_ptr + offs)
    go = tl.load(grad_out_ptr + offs)

    z = x * scale + base
    sig = 1.0 / (1.0 + tl.exp(-z))
    grad_z = go * sig * (1.0 - sig)

    grad_input_mix = grad_z * scale

    tl.store(grad_input_mix_ptr + offs, grad_input_mix)
    tl.store(grad_z_ptr + offs, grad_z)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        input_mix: torch.Tensor,
        mhc_scale: torch.Tensor,
        mhc_base: torch.Tensor,
        grad_out: torch.Tensor,
    ):
        numel = input_mix.numel()
        mhc_mult = mhc_base.numel()

        grad_input_mix = torch.empty_like(input_mix)
        grad_z = torch.empty_like(input_mix)

        BLOCK = 1024
        grid = ((numel + BLOCK - 1) // BLOCK,)
        _sigmoid_backward_kernel[grid](
            input_mix,
            mhc_scale,
            mhc_base,
            grad_out,
            grad_input_mix,
            grad_z,
            mhc_mult=mhc_mult,
            BLOCK=BLOCK,
            num_warps=1,
        )

        grad_mhc_base = grad_z.sum(dim=(0, 1), keepdim=True).view(-1)
        grad_mhc_scale = (grad_z * input_mix).sum(dim=(0, 1, 2), keepdim=True).view(1)

        return grad_input_mix, grad_mhc_scale, grad_mhc_base


batch0 = 2
batch1 = 1024
mhc_mult = 4


def get_inputs():
    input_mix = torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32, device="cuda")
    mhc_scale = torch.randn(1, dtype=torch.float32, device="cuda")
    mhc_base = torch.randn(mhc_mult, dtype=torch.float32, device="cuda")
    grad_out = torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32, device="cuda")

    return [input_mix, mhc_scale, mhc_base, grad_out]


def get_init_inputs():
    return []


if __name__ == "__main__":
    model = ModelNew().eval()
    with torch.no_grad():
        outs = model(*get_inputs())
    for o in outs:
        print(o.shape)
