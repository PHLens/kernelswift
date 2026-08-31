"""Vendor-neutral Triton fallback for the official benchmark shape.

This fallback intentionally imports only standard torch/triton modules.
It is used when the detected accelerator has no task-specific implementation.
"""

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_rotary_embedding_kernel(
    timestamps_ptr,      # [B, S] fp32 contiguous
    inv_freq_ptr,        # [D2] fp32 contiguous, D2 = dim // 2
    position_angles_ptr, # [MAXS, D] fp32 contiguous, D = dim (already interleaved)
    cos_out_ptr,         # [B, S, D2X] fp32 contiguous, D2X = 2 * D
    sin_out_ptr,         # [B, S, D2X] fp32 contiguous
    B, S, D2, D, MAXS, D2X,
    D2X_POW2: tl.constexpr,  # power-of-two >= D2X, used as column extent
    BLOCK: tl.constexpr,     # number of (b, s) pairs per program
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < B * S
    b = offs // S
    s = offs % S

    ts = tl.load(timestamps_ptr + offs, mask=mask, other=0.0)
    angle = -ts * 6.283185307179586  # -timestamps * 2 * pi

    batch_pos = b.to(tl.float32) / MAXS  # (BLOCK,)

    col = tl.arange(0, D2X_POW2)  # (D2X_POW2,) column indices over the padded dim

    # Column c maps into the 128-dim as follows:
    #   c < 64  (batch half):  freq = batch_pos * inv_freq[c // 2]
    #   c >= 64 (time half):   freq = position_angles[s, c - 64]
    # We gather per-column frequencies for every (b, s) row.
    is_batch = col < D  # D == 64, the batch half occupies columns [0, 64)
    k_batch = col // 2  # inv_freq index for batch half

    # Load batch-half source value per column: inv_freq[c//2] for c < 64
    inv_idx = k_batch
    inv_val = tl.load(inv_freq_ptr + inv_idx, mask=is_batch, other=0.0)  # (D2X_POW2,)
    bf = batch_pos[:, None] * inv_val[None, :]  # (BLOCK, D2X_POW2)

    # Load time-half source value per column: position_angles[s, c - 64] for c >= 64
    time_idx = col - D  # (D2X_POW2,)
    time_valid = (col >= D) & (col < D2X)  # only real columns within [64, 128)
    # per-row gather: ptr = s * D + time_idx, mask also requires time_valid
    tf = tl.load(
        position_angles_ptr + s[:, None] * D + time_idx[None, :],
        mask=time_valid[None, :],
        other=0.0,
    )  # (BLOCK, D2X_POW2)

    freq = tl.where(is_batch[None, :], bf, tf)  # (BLOCK, D2X_POW2)

    x = freq * angle[:, None]
    cos_vals = tl.cos(x)
    sin_vals = tl.sin(x)

    store_mask = mask[:, None] & (col[None, :] < D2X)
    out_off = offs[:, None] * D2X + col[None, :]
    tl.store(cos_out_ptr + out_off, cos_vals, mask=store_mask)
    tl.store(sin_out_ptr + out_off, sin_vals, mask=store_mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


class ModelNew(nn.Module):
    """MusicFlamingoRotaryEmbedding: batch (song) + time positional embedding.

    Fused Triton implementation of the reference forward elementwise chain.
    Returns (cos, sin) where cos/sin combines batch and time frequencies.
    """

    def __init__(self, dim: int = 64, max_seq_len: int = 256, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        positions = torch.arange(max_seq_len, dtype=torch.float)
        positions_norm = positions / max_seq_len * (2 * math.pi)
        position_angles = positions_norm.unsqueeze(-1) * inv_freq
        position_angles = position_angles.repeat_interleave(2, dim=-1)
        self.register_buffer("position_angles", position_angles)

    def forward(self, timestamps: torch.Tensor, seq_len: int):
        B = timestamps.shape[0]
        S = timestamps.shape[1]
        D2 = self.inv_freq.shape[0]  # dim // 2
        D = self.position_angles.shape[1]  # dim (already interleaved)
        D2X = 2 * D  # output inner dim

        device = timestamps.device
        dtype = timestamps.dtype

        cos_out = torch.empty((B, S, D2X), device=device, dtype=dtype)
        sin_out = torch.empty((B, S, D2X), device=device, dtype=dtype)

        D2X_POW2 = _next_pow2(D2X)

        total = B * S
        BLOCK = 64
        grid = (triton.cdiv(total, BLOCK),)

        _fused_rotary_embedding_kernel[grid](
            timestamps,
            self.inv_freq,
            self.position_angles,
            cos_out,
            sin_out,
            B,
            S,
            D2,
            D,
            self.max_seq_len,
            D2X,
            D2X_POW2=D2X_POW2,
            BLOCK=BLOCK,
        )
        return cos_out, sin_out


def get_inputs():
    B, SEQ = 4, 32
    timestamps = torch.rand(B, SEQ, device="cuda")
    return [timestamps, SEQ]


def get_init_inputs():
    return [64, 256, 10000.0]


if __name__ == "__main__":
    model = ModelNew(*get_init_inputs()).cuda().eval()
    with torch.no_grad():
        cos, sin = model(*get_inputs())
    print(cos.shape, sin.shape)
