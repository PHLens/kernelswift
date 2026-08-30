"""Centre Random Augmentation (diffusion rigid-body random augmentation).

Semantics (matches base.py exactly):
  center = masked mean of x_input_coords
  x = (x_input_coords - center).unsqueeze(0).expand(n_sample, -1, -1)
  R = random_rotation_matrices(n_sample)  (from host random quaternions u1,u2,u3)
  T = s_trans * randn(n_sample, 3)
  x = rot_vec_mul(R, x) + T
  if mask is not None: x = x * mask

The random numbers (u1,u2,u3,T) are generated on the HOST in exactly the same
order and shapes as base.py (torch.rand x3 then torch.randn), so that after
auto_bench's set_seed(42) the sequences are identical. The 3x3 matvec rotation
+ translation + masking is fused into a single Triton kernel.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_gcu  # noqa: F401 - registers the GCU device
import triton_gcu  # noqa: F401


@triton.jit
def _rotate_translate_kernel(
    x_ptr,          # [n_sample, N_atom, 3] fp32 (centered & expanded)
    r_ptr,          # [n_sample, 3, 3] fp32
    t_ptr,          # [n_sample, 3] fp32
    mask_ptr,       # [N_atom] fp32 (ones or 0/1)
    out_ptr,        # [n_sample, N_atom, 3] fp32
    N_atom,
    n_sample: tl.constexpr,
    has_mask: tl.constexpr,
):
    atom = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.arange(0, 4)
    c = tl.arange(0, 4)
    valid = (d < 3) & (c < 3)

    # load R[s]: [4,4] with only [0:3,0:3] valid
    r = tl.load(
        r_ptr + s * 9 + c[:, None] * 3 + d[None, :],
        mask=valid,
        other=0.0,
    )
    t = tl.load(t_ptr + s * 3 + d, mask=d < 3, other=0.0)
    xv = tl.load(x_ptr + s * N_atom * 3 + atom * 3 + c, mask=c < 3, other=0.0)

    # out[d] = sum_c r[d, c] * x[c] + t[d]
    acc = tl.sum(r * xv[None, :], axis=1) + t

    if has_mask:
        m = tl.load(mask_ptr + atom)
        acc = acc * m

    tl.store(out_ptr + s * N_atom * 3 + atom * 3 + d, acc, mask=d < 3)


class ModelNew(nn.Module):
    def __init__(self, n_sample: int = 1, s_trans: float = 1.0, centre_only: bool = False):
        super().__init__()
        self.n_sample = n_sample
        self.s_trans = s_trans
        self.centre_only = centre_only

    def forward(self, x_input_coords: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = x_input_coords.device
        dtype = x_input_coords.dtype
        n_sample = self.n_sample
        n_atom = x_input_coords.shape[0]

        if mask is None:
            center = x_input_coords.mean(dim=-2, keepdim=True)
        else:
            m = mask.to(dtype=dtype).unsqueeze(-1)
            center = (x_input_coords * m).sum(dim=-2, keepdim=True) / (m.sum(dim=-2, keepdim=True) + 1e-12)
        x = x_input_coords - center
        x = x.unsqueeze(0).expand(n_sample, -1, -1).contiguous()

        if self.centre_only:
            return x

        # --- random rotation matrices (same order/shapes as base.py) ---
        u1 = torch.rand(n_sample, device=device, dtype=dtype)
        u2 = torch.rand(n_sample, device=device, dtype=dtype)
        u3 = torch.rand(n_sample, device=device, dtype=dtype)

        q1 = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2)
        q2 = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
        q3 = torch.sqrt(u1) * torch.sin(2 * math.pi * u3)
        q4 = torch.sqrt(u1) * torch.cos(2 * math.pi * u3)
        xx, yy, zz = q1 * q1, q2 * q2, q3 * q3
        xy, xz, yz = q1 * q2, q1 * q3, q2 * q3
        wx, wy, wz = q4 * q1, q4 * q2, q4 * q3

        R = torch.stack(
            [
                1 - 2 * (yy + zz),
                2 * (xy - wz),
                2 * (xz + wy),
                2 * (xy + wz),
                1 - 2 * (xx + zz),
                2 * (yz - wx),
                2 * (xz - wy),
                2 * (yz + wx),
                1 - 2 * (xx + yy),
            ],
            dim=-1,
        ).reshape(n_sample, 3, 3)

        T = self.s_trans * torch.randn(n_sample, 3, device=device, dtype=dtype)

        out = torch.empty((n_sample, n_atom, 3), dtype=dtype, device=device)

        if mask is None:
            mask_ptr = torch.ones(n_atom, dtype=dtype, device=device)
            has_mask = False
        else:
            mask_ptr = mask.to(dtype=dtype).contiguous()
            has_mask = True

        _rotate_translate_kernel[(n_atom, n_sample)](
            x,
            R,
            T,
            mask_ptr,
            out,
            n_atom,
            n_sample=n_sample,
            has_mask=has_mask,
            num_warps=1,
        )
        return out


N_ATOM = 256
N_SAMPLE = 4
S_TRANS = 1.0
CENTRE_ONLY = False


def get_inputs():
    device = "cuda"
    x_input_coords = torch.randn(N_ATOM, 3, device=device)
    mask = torch.ones(N_ATOM, device=device, dtype=torch.float32)
    return [x_input_coords, mask]


def get_init_inputs():
    return [N_SAMPLE, S_TRANS, CENTRE_ONLY]


if __name__ == "__main__":
    model = ModelNew(*get_init_inputs()).eval()
    with torch.no_grad():
        out = model(*get_inputs())
    print(out.shape)
