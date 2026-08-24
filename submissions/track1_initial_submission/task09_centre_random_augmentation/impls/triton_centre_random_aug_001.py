"""
Centre Random Augmentation (扩散采样里用于随机刚体变换)

From: protenix/model/utils.py:centre_random_augmentation

Triton candidate (round 001): fuse the deterministic linear tail
(centering mean + x - center, 3x3 rotation matvec, translation add,
mask multiply) into a single Triton kernel over [4,256,3].

R/T generation (3x torch.rand(4) + 1x torch.randn(4,3)) and the
quaternion-to-matrix Sin/Cos/Sqrt stack REMAIN in torch so the seeded
RNG draw order and the resulting R/T are bitwise identical to base.py.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _centre_aug_linear_kernel(
    x_ptr,        # x_input_coords [256,3] fp32 contiguous
    r_ptr,        # R [4,3,3] fp32 contiguous
    t_ptr,        # T [4,3] fp32 contiguous
    m_ptr,        # mask [256] fp32 contiguous
    center_ptr,   # center [3] fp32 contiguous (precomputed in torch)
    out_ptr,      # out [4,256,3] fp32 contiguous
    N_ATOM: tl.constexpr,   # 256
    N_SAMPLE: tl.constexpr,  # 4
    BLOCK: tl.constexpr,    # number of (sample, atom) rows per program
):
    # Grid is 1D over (sample * atom) rows; each row writes 3 contiguous floats.
    pid = tl.program_id(0)
    row = pid * BLOCK + tl.arange(0, BLOCK)   # flat row index in [0, N_SAMPLE*N_ATOM)
    row_mask = row < (N_SAMPLE * N_ATOM)

    sample = row // N_ATOM
    atom = row % N_ATOM

    # Load center (3 scalars)
    cx = tl.load(center_ptr + 0)
    cy = tl.load(center_ptr + 1)
    cz = tl.load(center_ptr + 2)

    # Load x_input_coords[atom, 0:3]
    x0 = tl.load(x_ptr + atom * 3 + 0, mask=row_mask, other=0.0)
    x1 = tl.load(x_ptr + atom * 3 + 1, mask=row_mask, other=0.0)
    x2 = tl.load(x_ptr + atom * 3 + 2, mask=row_mask, other=0.0)

    # Centre: (x - center)
    c0 = x0 - cx
    c1 = x1 - cy
    c2 = x2 - cz

    # Load R[sample, 0:3, 0:3] (9 contiguous floats per sample)
    r_base = sample * 9
    r00 = tl.load(r_ptr + r_base + 0, mask=row_mask, other=0.0)
    r01 = tl.load(r_ptr + r_base + 1, mask=row_mask, other=0.0)
    r02 = tl.load(r_ptr + r_base + 2, mask=row_mask, other=0.0)
    r10 = tl.load(r_ptr + r_base + 3, mask=row_mask, other=0.0)
    r11 = tl.load(r_ptr + r_base + 4, mask=row_mask, other=0.0)
    r12 = tl.load(r_ptr + r_base + 5, mask=row_mask, other=0.0)
    r20 = tl.load(r_ptr + r_base + 6, mask=row_mask, other=0.0)
    r21 = tl.load(r_ptr + r_base + 7, mask=row_mask, other=0.0)
    r22 = tl.load(r_ptr + r_base + 8, mask=row_mask, other=0.0)

    # Load T[sample, 0:3]
    t_base = sample * 3
    t0 = tl.load(t_ptr + t_base + 0, mask=row_mask, other=0.0)
    t1 = tl.load(t_ptr + t_base + 1, mask=row_mask, other=0.0)
    t2 = tl.load(t_ptr + t_base + 2, mask=row_mask, other=0.0)

    # 3x3 matvec: o = R @ c + T
    o0 = r00 * c0 + r01 * c1 + r02 * c2 + t0
    o1 = r10 * c0 + r11 * c1 + r12 * c2 + t1
    o2 = r20 * c0 + r21 * c1 + r22 * c2 + t2

    # Mask multiply: o *= mask[atom]
    m = tl.load(m_ptr + atom, mask=row_mask, other=1.0)
    o0 = o0 * m
    o1 = o1 * m
    o2 = o2 * m

    # Store out[sample, atom, 0:3]
    out_base = row * 3
    tl.store(out_ptr + out_base + 0, o0, mask=row_mask)
    tl.store(out_ptr + out_base + 1, o1, mask=row_mask)
    tl.store(out_ptr + out_base + 2, o2, mask=row_mask)


def random_rotation_matrices(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    生成 n 个随机旋转矩阵 [n,3,3]，基于随机四元数（均匀分布）。
    """
    u1 = torch.rand(n, device=device, dtype=dtype)
    u2 = torch.rand(n, device=device, dtype=dtype)
    u3 = torch.rand(n, device=device, dtype=dtype)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * math.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * math.pi * u3)
    # quaternion (x,y,z,w)
    x, y, z, w = q1, q2, q3, q4

    # convert to rotation matrix
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

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
    ).reshape(n, 3, 3)
    return R


def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    r: [...,3,3], t: [...,3]
    """
    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


class ModelNew(nn.Module):
    def __init__(self, n_sample: int = 1, s_trans: float = 1.0, centre_only: bool = False):
        super().__init__()
        self.n_sample = n_sample
        self.s_trans = s_trans
        self.centre_only = centre_only

    def forward(self, x_input_coords: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = x_input_coords.device
        dtype = x_input_coords.dtype

        n_atom = x_input_coords.shape[0]
        n_sample = self.n_sample

        # ---- centre_only fast path (identical semantics to base.py) ----
        if mask is None:
            center = x_input_coords.mean(dim=-2, keepdim=True)
        else:
            m = mask.to(dtype=dtype).unsqueeze(-1)
            center = (x_input_coords * m).sum(dim=-2, keepdim=True) / (m.sum(dim=-2, keepdim=True) + 1e-12)

        if self.centre_only:
            x = x_input_coords - center
            x = x.unsqueeze(0).expand(n_sample, -1, -1).contiguous()
            return x

        # ---- R/T generation stays in torch (preserve seeded RNG stream) ----
        R = random_rotation_matrices(n_sample, device=device, dtype=dtype)  # [n,3,3]
        T = self.s_trans * torch.randn(n_sample, 3, device=device, dtype=dtype)

        # ---- fused linear tail in Triton over [n_sample, n_atom, 3] ----
        center_c = center.reshape(-1).contiguous().to(torch.float32)

        out = torch.empty((n_sample, n_atom, 3), dtype=torch.float32, device=device)

        if mask is None:
            m = torch.ones(n_atom, device=device, dtype=dtype)
        else:
            m = mask.to(dtype=dtype).contiguous()

        xc = x_input_coords.contiguous()
        Rc = R.contiguous()
        Tc = T.contiguous()

        total_rows = n_sample * n_atom
        BLOCK = 256
        grid = (triton.cdiv(total_rows, BLOCK),)
        _centre_aug_linear_kernel[grid](
            xc, Rc, Tc, m, center_c, out,
            N_ATOM=n_atom,
            N_SAMPLE=n_sample,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out


# ==========================================
# Hyperparameters & Data Generation
# ==========================================

N_ATOM = 256
N_SAMPLE = 4
S_TRANS = 1.0
CENTRE_ONLY = False


def get_inputs():
    device = 'cuda'
    torch.manual_seed(42)

    x_input_coords = torch.randn(N_ATOM, 3, device=device)
    mask = torch.ones(N_ATOM, device=device, dtype=torch.float32)

    return [x_input_coords, mask]


def get_init_inputs():
    return [N_SAMPLE, S_TRANS, CENTRE_ONLY]
