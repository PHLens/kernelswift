"""
Centre Random Augmentation (扩散采样里用于随机刚体变换)

From: protenix/model/utils.py:centre_random_augmentation

Round 001 candidate: kernel-fusion.

The deterministic rigid-transform chain (centering subtraction, the 3x3-by-3
vector product `rot_vec_mul`, the `+ T` translation, and the `* mask` multiply)
is fused into a single Triton kernel. The random number draws
(`random_rotation_matrices` -> 3x `torch.rand`, and `torch.randn` for the
translation) remain ordinary host-dispatched torch calls inside `forward`, in
the exact reference order, so the RNG consumption order is byte-identical to the
reference and `R`/`T` stay bit-comparable under the harness's per-call `set_seed`.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl


def random_rotation_matrices(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    生成 n 个随机旋转矩阵 [n,3,3]，基于随机四元数（均匀分布）。

    Byte-identical to the reference: three uniform draws `u1,u2,u3 = torch.rand(n)`
    followed by the quaternion -> rotation-matrix construction. Kept on the host
    so the RNG consumption order matches the reference exactly.
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


@triton.jit
def _centre_aug_kernel(
    xc_ptr,        # [N_ATOM, 3] fp32 row-major
    mask_ptr,      # [N_ATOM] fp32
    R_ptr,         # [N_SAMPLE, 9] fp32 row-major (flattened [N_SAMPLE,3,3])
    T_ptr,         # [N_SAMPLE, 3] fp32 row-major
    out_ptr,       # [N_SAMPLE, N_ATOM, 3] fp32 row-major
    N_ATOM,
    eps,
    BLOCK_N: tl.constexpr,  # power-of-two >= N_ATOM
):
    # One program per sample; the centering reduction is recomputed per program
    # (tiny data, correctness-first) and is sample-independent.
    s = tl.program_id(0)

    offs = tl.arange(0, BLOCK_N)
    valid = offs < N_ATOM

    m = tl.load(mask_ptr + offs, mask=valid, other=0.0)
    msum = tl.sum(m)

    x0 = tl.load(xc_ptr + offs * 3 + 0, mask=valid, other=0.0)
    x1 = tl.load(xc_ptr + offs * 3 + 1, mask=valid, other=0.0)
    x2 = tl.load(xc_ptr + offs * 3 + 2, mask=valid, other=0.0)

    # centering: center_j = sum_i(xc[i,j] * m[i]) / (sum_i(m[i]) + eps)
    c0 = tl.sum(x0 * m) / (msum + eps)
    c1 = tl.sum(x1 * m) / (msum + eps)
    c2 = tl.sum(x2 * m) / (msum + eps)

    # centered coordinates
    cx0 = x0 - c0
    cx1 = x1 - c1
    cx2 = x2 - c2

    # load R[s] (9 scalar entries) and T[s] (3 scalar entries)
    r00 = tl.load(R_ptr + s * 9 + 0)
    r01 = tl.load(R_ptr + s * 9 + 1)
    r02 = tl.load(R_ptr + s * 9 + 2)
    r10 = tl.load(R_ptr + s * 9 + 3)
    r11 = tl.load(R_ptr + s * 9 + 4)
    r12 = tl.load(R_ptr + s * 9 + 5)
    r20 = tl.load(R_ptr + s * 9 + 6)
    r21 = tl.load(R_ptr + s * 9 + 7)
    r22 = tl.load(R_ptr + s * 9 + 8)

    t0 = tl.load(T_ptr + s * 3 + 0)
    t1 = tl.load(T_ptr + s * 3 + 1)
    t2 = tl.load(T_ptr + s * 3 + 2)

    # rot_vec_mul + translation + mask multiply
    o0 = (r00 * cx0 + r01 * cx1 + r02 * cx2 + t0) * m
    o1 = (r10 * cx0 + r11 * cx1 + r12 * cx2 + t1) * m
    o2 = (r20 * cx0 + r21 * cx1 + r22 * cx2 + t2) * m

    base = s * N_ATOM * 3
    tl.store(out_ptr + base + offs * 3 + 0, o0, mask=valid)
    tl.store(out_ptr + base + offs * 3 + 1, o1, mask=valid)
    tl.store(out_ptr + base + offs * 3 + 2, o2, mask=valid)


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
        eps = 1e-12

        if self.centre_only:
            # Reference early-return path: no RNG consumed.
            if mask is None:
                center = x_input_coords.mean(dim=-2, keepdim=True)
            else:
                m = mask.to(dtype=dtype).unsqueeze(-1)
                center = (x_input_coords * m).sum(dim=-2, keepdim=True) / (m.sum(dim=-2, keepdim=True) + eps)
            x = x_input_coords - center
            return x.unsqueeze(0).expand(n_sample, -1, -1).contiguous()

        # Random draws: identical order and count to the reference
        # (3x torch.rand inside random_rotation_matrices, then 1x torch.randn).
        R = random_rotation_matrices(n_sample, device=device, dtype=dtype)  # [n,3,3]
        T = self.s_trans * torch.randn(n_sample, 3, device=device, dtype=dtype)

        if mask is None:
            # Reference mask-None fallback (not exercised by the harness): keep
            # the exact reference dataflow, no fusion.
            center = x_input_coords.mean(dim=-2, keepdim=True)
            x = x_input_coords - center
            x = x.unsqueeze(0).expand(n_sample, -1, -1).contiguous()
            return rot_vec_mul(R[:, None, :, :].expand(-1, x.shape[1], -1, -1), x) + T[:, None, :]

        # mask is not None (harness path): fuse centering + rot_vec_mul +
        # translation + mask multiply into a single Triton kernel.
        N_ATOM = x_input_coords.shape[0]
        m = mask.to(dtype=dtype)  # [N_ATOM] fp32
        R_flat = R.reshape(n_sample, 9).contiguous()  # [n,9] row-major
        out = torch.empty((n_sample, N_ATOM, 3), device=device, dtype=dtype)

        BLOCK_N = triton.next_power_of_2(N_ATOM)
        _centre_aug_kernel[(n_sample,)](
            x_input_coords, m, R_flat, T, out,
            N_ATOM, eps,
            BLOCK_N=BLOCK_N,
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
