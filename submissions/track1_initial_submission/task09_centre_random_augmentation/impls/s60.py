"""Centre Random Augmentation (diffusion rigid-body random augmentation) — epoch2 Round 001.

Semantics (matches base.py exactly):
  center = masked mean of x_input_coords  (or unbiased mean when mask is None)
  x_centered = x_input_coords - center                       # [N_atom, 3]
  u1/u2/u3 = torch.rand(n_sample) x3  (host, base order)
  T = s_trans * torch.randn(n_sample, 3)  (host, base order)
  out[s,a,:] = R(u1[s],u2[s],u3[s]) @ x_centered[a,:] + T[s,:]
  if mask is not None: out = out * mask

The whole per-sample path (quaternion -> 9-element rotation matrix -> 3x3 matvec
-> translation -> optional mask) is fused into ONE direct-launched Triton kernel
grid=(n_sample,)=4, num_warps=1.  The random sources u1/u2/u3/T are generated on
the HOST in exactly the base order/count/shape so the seed-42 random sequence is
bit-identical to base, keeping correctness exact-match under the allclose comparator.

Legality: primary contract math.elementwise (tl.sqrt/tl.sin/tl.cos/mul/add/sub);
NO tl.dot (the 3x3 matvec is 3 statically-unrolled fp32 dot products); tl.arange
extent is BLOCK=256 (power-of-2); num_warps=1; no torch.compile/graph/capture;
no .contiguous() anywhere in the forward host path.
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


_BLOCK = 256


@triton.jit
def _centre_random_aug_kernel(
    x_ptr,          # [N_atom, 3] fp32 (x_centered = x_input_coords - center)
    u1_ptr,         # [n_sample] fp32
    u2_ptr,         # [n_sample] fp32
    u3_ptr,         # [n_sample] fp32
    t_ptr,          # [n_sample, 3] fp32
    mask_ptr,       # [N_atom] fp32 (only read when has_mask)
    out_ptr,        # [n_sample, N_atom, 3] fp32
    N_atom: tl.constexpr,
    BLOCK: tl.constexpr,
    has_mask: tl.constexpr,
):
    s = tl.program_id(0)

    # --- random sources for this sample (scalars) ---
    u1 = tl.load(u1_ptr + s)
    u2 = tl.load(u2_ptr + s)
    u3 = tl.load(u3_ptr + s)

    # --- quaternion (uniform) -> rotation matrix, arithmetic order = base ---
    q1 = tl.sqrt(1.0 - u1) * tl.sin(2.0 * math.pi * u2)
    q2 = tl.sqrt(1.0 - u1) * tl.cos(2.0 * math.pi * u2)
    q3 = tl.sqrt(u1) * tl.sin(2.0 * math.pi * u3)
    q4 = tl.sqrt(u1) * tl.cos(2.0 * math.pi * u3)
    x, y, z, w = q1, q2, q3, q4

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    # --- translation for this sample (3 scalars) ---
    t0 = tl.load(t_ptr + s * 3 + 0)
    t1 = tl.load(t_ptr + s * 3 + 1)
    t2 = tl.load(t_ptr + s * 3 + 2)

    # --- loop atoms in power-of-2 BLOCK tiles (BLOCK=256 => 1 iteration) ---
    for a_start in range(0, N_atom, BLOCK):
        offs = a_start + tl.arange(0, BLOCK)
        mask_a = offs < N_atom

        x0 = tl.load(x_ptr + offs * 3 + 0, mask=mask_a, other=0.0)
        x1 = tl.load(x_ptr + offs * 3 + 1, mask=mask_a, other=0.0)
        x2 = tl.load(x_ptr + offs * 3 + 2, mask=mask_a, other=0.0)

        # --- 3x3 matvec (3 statically-unrolled fp32 dot products) + translation ---
        o0 = r00 * x0 + r01 * x1 + r02 * x2 + t0
        o1 = r10 * x0 + r11 * x1 + r12 * x2 + t1
        o2 = r20 * x0 + r21 * x1 + r22 * x2 + t2

        if has_mask:
            m = tl.load(mask_ptr + offs, mask=mask_a, other=0.0)
            o0 = o0 * m
            o1 = o1 * m
            o2 = o2 * m

        out_base = s * N_atom * 3 + offs * 3
        tl.store(out_ptr + out_base + 0, o0, mask=mask_a)
        tl.store(out_ptr + out_base + 1, o1, mask=mask_a)
        tl.store(out_ptr + out_base + 2, o2, mask=mask_a)


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

        # --- center (torch, identical to base) ---
        if mask is None:
            center = x_input_coords.mean(dim=-2, keepdim=True)
        else:
            m = mask.to(dtype=dtype).unsqueeze(-1)
            center = (x_input_coords * m).sum(dim=-2, keepdim=True) / (m.sum(dim=-2, keepdim=True) + 1e-12)
        x_centered = x_input_coords - center  # [N_atom, 3]

        if self.centre_only:
            return x_centered.unsqueeze(0).expand(n_sample, -1, -1)

        # --- random sources (host, base order/count/shape) ---
        u1 = torch.rand(n_sample, device=device, dtype=dtype)
        u2 = torch.rand(n_sample, device=device, dtype=dtype)
        u3 = torch.rand(n_sample, device=device, dtype=dtype)
        T = self.s_trans * torch.randn(n_sample, 3, device=device, dtype=dtype)

        out = torch.empty((n_sample, n_atom, 3), dtype=dtype, device=device)

        if mask is None:
            has_mask = False
            mask_ptr = x_centered  # dummy; never read when has_mask=False
        else:
            has_mask = True
            mask_ptr = mask.to(dtype=dtype)

        _centre_random_aug_kernel[(n_sample,)](
            x_centered,
            u1,
            u2,
            u3,
            T,
            mask_ptr,
            out,
            N_atom=n_atom,
            BLOCK=_BLOCK,
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
    torch.manual_seed(42)

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
