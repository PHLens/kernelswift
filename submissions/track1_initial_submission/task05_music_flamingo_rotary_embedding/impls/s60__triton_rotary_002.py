import math

import torch
import torch.nn as nn
import torch_gcu
import triton
import triton.language as tl
import triton_gcu


@triton.jit
def _rotary_embedding_kernel(
    timestamps_ptr,
    inv_freq_ptr,
    position_angles_ptr,
    cos_ptr,
    sin_ptr,
    max_seq_len,
    seq_len,
    D: tl.constexpr,  # 64 (per-branch freq width)
    BLOCK: tl.constexpr,  # tile width per program (128)
):
    # Flattened elementwise map over [B=4, T=32, D=128] = 16384 elements,
    # partitioned across grid=(16384 // BLOCK,) programs. Each program handles
    # BLOCK consecutive flattened elements, computes cos(theta) and sin(theta),
    # and writes both output buffers in the single launch.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < 16384

    b = offs // (seq_len * (2 * D))          # batch index [0,4)
    rem = offs % (seq_len * (2 * D))
    t = rem // (2 * D)                        # time index [0, seq_len)
    d = rem % (2 * D)                         # freq column [0, 128)

    ts = tl.load(timestamps_ptr + b * seq_len + t, mask=mask, other=0.0).to(
        tl.float32
    )
    angle = -ts * 6.283185307179586

    # branch select: d < D uses batch_freq (b/max_seq_len * inv_freq[d//2]),
    # d >= D uses time_freq (position_angles[t, d-D]).
    is_time = d >= D
    inv_idx = (d // 2) % D                    # d//2 for batch branch
    time_d = d - D                             # column into position_angles [256,64]

    inv = tl.load(inv_freq_ptr + inv_idx, mask=mask, other=0.0).to(tl.float32)
    batch_freq = (b.to(tl.float32) / max_seq_len) * inv

    pa = tl.load(
        position_angles_ptr + t * D + time_d, mask=mask, other=0.0
    ).to(tl.float32)

    freq = tl.where(is_time, pa, batch_freq)
    theta = freq * angle

    c = tl.cos(theta)
    s = tl.sin(theta)

    tl.store(cos_ptr + offs, c, mask=mask)
    tl.store(sin_ptr + offs, s, mask=mask)


class ModelNew(nn.Module):
    """MusicFlamingoRotaryEmbedding: batch (song) + time positional embedding.

    Fuses the forward elementwise/view ops (arange, mul/div, repeat_interleave,
    broadcast, cat, angle multiply, cos, sin) into a single Triton elementwise
    kernel that writes both cos and sin output buffers in one launch.
    Returns (cos, sin) where cos/sin combines batch and time frequencies.
    """

    def __init__(self, dim: int = 64, max_seq_len: int = 256, base: float = 10000.0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.dim = dim
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        self.register_buffer("inv_freq", inv_freq)
        positions = torch.arange(max_seq_len, dtype=torch.float)
        positions_norm = positions / max_seq_len * (2 * math.pi)
        position_angles = positions_norm.unsqueeze(-1) * inv_freq
        position_angles = position_angles.repeat_interleave(2, dim=-1)
        self.register_buffer("position_angles", position_angles)

    def forward(self, timestamps: torch.Tensor, seq_len: int):
        B = timestamps.shape[0]
        D = self.dim
        out_shape = (B, seq_len, 2 * D)
        total = B * seq_len * 2 * D

        cos_out = torch.empty(out_shape, dtype=torch.float32, device=timestamps.device)
        sin_out = torch.empty(out_shape, dtype=torch.float32, device=timestamps.device)

        # grid-parallelism repair: partition the 16384-element flat map across
        # 128 programs of BLOCK=128 each (1-D grid, tl.program_id axis 0 only).
        BLOCK = 128
        grid = (total // BLOCK,)

        _rotary_embedding_kernel[grid](
            timestamps,
            self.inv_freq,
            self.position_angles,
            cos_out,
            sin_out,
            self.max_seq_len,
            seq_len,
            D=D,
            BLOCK=BLOCK,
            num_warps=1,
        )
        return (cos_out, sin_out)


def get_inputs():
    B, SEQ = (4, 32)
    timestamps = torch.rand(B, SEQ, device="cuda")
    return [timestamps, SEQ]


def get_init_inputs():
    return [64, 256, 10000.0]


if __name__ == "__main__":
    model = ModelNew(*get_init_inputs()).eval()
    with torch.no_grad():
        cos, sin = model(*get_inputs())
    print(cos.shape, sin.shape)
