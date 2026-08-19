"""
Centre Random Augmentation (扩散采样里用于随机刚体变换)

From: protenix/model/utils.py:centre_random_augmentation

Round 002 candidate: kernel-fusion (extends Round 001).

Extends the fused Triton kernel to also absorb the quaternion-to-rotation-matrix
construction (sqrt/sin/cos and the 9-entry matrix arithmetic) from u1/u2/u3.
The random draws (3x `torch.rand` for u1/u2/u3, 1x `torch.randn` for T) remain
ordinary host-dispatched torch calls inside `forward`, in the exact reference
order, so the RNG consumption order is byte-identical to the reference. The
fused kernel only reads u1/u2/u3/T and performs deterministic math; it never
draws random numbers.

A local probe confirmed `tl.sqrt`/`tl.sin`/`tl.cos` lower on the CoreX Triton
3.1.0 BI150 backend and produce bit-identical results to `torch.sqrt/sin/cos`
(max abs diff 0.0), so the transcendental construction is not a capability-miss.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _centre_aug_kernel(
    xc_ptr,        # [N_ATOM, 3] fp32 row-major
    mask_ptr,      # [N_ATOM] fp32
    u1_ptr,        # [N_SAMPLE] fp32
    u2_ptr,        # [N_SAMPLE] fp32
    u3_ptr,        # [N_SAMPLE] fp32
    T_ptr,         # [N_SAMPLE, 3] fp32 row-major
    out_ptr,       # [N_SAMPLE, N_ATOM, 3] fp32 row-major
    N_ATOM,
    eps,
    BLOCK_N: tl.constexpr,  # power-of-two >= N_ATOM
):
    # One program per sample.
    s = tl.program_id(0)

    # --- quaternion -> rotation matrix construction (deterministic) ---
    u1s = tl.load(u1_ptr + s)
    u2s = tl.load(u2_ptr + s)
    u3s = tl.load(u3_ptr + s)

    s1 = tl.sqrt(1.0 - u1s)   # sqrt(1 - u1)
    su = tl.sqrt(u1s)         # sqrt(u1)

    # 2 * math.pi as a literal (Triton JIT cannot resolve module globals)
    q1 = s1 * tl.sin(6.283185307179586 * u2s)
    q2 = s1 * tl.cos(6.283185307179586 * u2s)
    q3 = su * tl.sin(6.283185307179586 * u3s)
    q4 = su * tl.cos(6.283185307179586 * u3s)

    # quaternion (x, y, z, w)
    x = q1
    y = q2
    z = q3
    w = q4

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    # --- centering ---
    offs = tl.arange(0, BLOCK_N)
    valid = offs < N_ATOM

    m = tl.load(mask_ptr + offs, mask=valid, other=0.0)
    msum = tl.sum(m)

    x0 = tl.load(xc_ptr + offs * 3 + 0, mask=valid, other=0.0)
    x1 = tl.load(xc_ptr + offs * 3 + 1, mask=valid, other=0.0)
    x2 = tl.load(xc_ptr + offs * 3 + 2, mask=valid, other=0.0)

    c0 = tl.sum(x0 * m) / (msum + eps)
    c1 = tl.sum(x1 * m) / (msum + eps)
    c2 = tl.sum(x2 * m) / (msum + eps)

    cx0 = x0 - c0
    cx1 = x1 - c1
    cx2 = x2 - c2

    # --- rot_vec_mul + translation + mask multiply ---
    t0 = tl.load(T_ptr + s * 3 + 0)
    t1 = tl.load(T_ptr + s * 3 + 1)
    t2 = tl.load(T_ptr + s * 3 + 2)

    o0 = (r00 * cx0 + r01 * cx1 + r02 * cx2 + t0) * m
    o1 = (r10 * cx0 + r11 * cx1 + r12 * cx2 + t1) * m
    o2 = (r20 * cx0 + r21 * cx1 + r22 * cx2 + t2) * m

    base = s * N_ATOM * 3
    tl.store(out_ptr + base + offs * 3 + 0, o0, mask=valid)
    tl.store(out_ptr + base + offs * 3 + 1, o1, mask=valid)
    tl.store(out_ptr + base + offs * 3 + 2, o2, mask=valid)


def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    r: [...,3,3], t: [...,3]  (used only by the mask-None fallback path)
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


def _random_rotation_matrices_host(n: int, device: torch.device, dtype: torch.dtype):
    """Host-side reference quaternion->matrix construction, used only by the
    mask-None fallback path (not the fused harness path)."""
    u1 = torch.rand(n, device=device, dtype=dtype)
    u2 = torch.rand(n, device=device, dtype=dtype)
    u3 = torch.rand(n, device=device, dtype=dtype)

    q1 = torch.sqrt(1 - u1) * torch.sin(6.283185307179586 * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(6.283185307179586 * u2)
    q3 = torch.sqrt(u1) * torch.sin(6.283185307179586 * u3)
    q4 = torch.sqrt(u1) * torch.cos(6.283185307179586 * u3)
    x, y, z, w = q1, q2, q3, q4

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

        if mask is None:
            # Reference mask-None fallback (not exercised by the harness): full
            # host-side reference dataflow, no fusion.
            R = _random_rotation_matrices_host(n_sample, device=device, dtype=dtype)
            T = self.s_trans * torch.randn(n_sample, 3, device=device, dtype=dtype)
            center = x_input_coords.mean(dim=-2, keepdim=True)
            x = x_input_coords - center
            x = x.unsqueeze(0).expand(n_sample, -1, -1).contiguous()
            return rot_vec_mul(R[:, None, :, :].expand(-1, x.shape[1], -1, -1), x) + T[:, None, :]

        # mask is not None (harness path): random draws in the exact reference
        # order (3x torch.rand then 1x torch.randn), then the fused kernel
        # performs quaternion->matrix, centering, rot_vec_mul, translation, and
        # mask multiply.
        u1 = torch.rand(n_sample, device=device, dtype=dtype)
        u2 = torch.rand(n_sample, device=device, dtype=dtype)
        u3 = torch.rand(n_sample, device=device, dtype=dtype)
        T = self.s_trans * torch.randn(n_sample, 3, device=device, dtype=dtype)

        N_ATOM = x_input_coords.shape[0]
        m = mask.to(dtype=dtype)  # [N_ATOM] fp32
        out = torch.empty((n_sample, N_ATOM, 3), device=device, dtype=dtype)

        BLOCK_N = triton.next_power_of_2(N_ATOM)
        _centre_aug_kernel[(n_sample,)](
            x_input_coords, m, u1, u2, u3, T, out,
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
