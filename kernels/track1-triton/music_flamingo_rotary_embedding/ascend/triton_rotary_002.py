import math
import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_npu


@triton.jit
def _rotary_embedding_row_kernel(
    timestamps_ptr,
    batch_freq_base_ptr,
    position_angles_ptr,
    cos_out_ptr,
    sin_out_ptr,
    SEQ,
    MAX_SEQ,
    TWO_PI,
    DIM: tl.constexpr,
):
    # Row-per-program grid over (B, SEQ). Each program computes b/t/ts/angle once
    # as scalars, loads two contiguous DIM-wide frequency halves exactly once, and
    # writes the batch half [0:DIM] and time half [DIM:2DIM] of cos/sin directly
    # (no tl.where branch, no per-lane integer division).
    pid = tl.program_id(0)
    b = pid // SEQ
    t = pid % SEQ

    ts = tl.load(timestamps_ptr + b * SEQ + t)
    angle = -ts * TWO_PI

    b_f = b.to(tl.float32)
    scale = b_f / MAX_SEQ.to(tl.float32)

    cols = tl.arange(0, DIM)

    # Batch half: interleaved inv_freq precomputed in __init__ (repeat_interleave(2)).
    half_batch = tl.load(batch_freq_base_ptr + cols)
    row_batch = half_batch * scale
    val_batch = row_batch * angle

    # Time half: position_angles[t, 0:DIM] precomputed in __init__.
    half_time = tl.load(position_angles_ptr + t * DIM + cols)
    val_time = half_time * angle

    cos_b = tl.cos(val_batch)
    sin_b = tl.sin(val_batch)
    cos_t = tl.cos(val_time)
    sin_t = tl.sin(val_time)

    row_base = b * (SEQ * 2 * DIM) + t * (2 * DIM)
    tl.store(cos_out_ptr + row_base + cols, cos_b)
    tl.store(cos_out_ptr + row_base + DIM + cols, cos_t)
    tl.store(sin_out_ptr + row_base + cols, sin_b)
    tl.store(sin_out_ptr + row_base + DIM + cols, sin_t)


class ModelNew(nn.Module):
    """MusicFlamingoRotaryEmbedding: batch (song) + time positional embedding.
    Returns (cos, sin) where cos/sin combines batch and time frequencies."""

    def __init__(self, dim: int = 64, max_seq_len: int = 256, base: float = 10000.0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.dim = dim
        inv_freq = 1.0 / (base ** (
            torch.arange(0, dim, 2, dtype=torch.float) / dim
        ))
        self.register_buffer("inv_freq", inv_freq)

        # Interleaved inv_freq (repeat_interleave(2)) -> [dim], precomputed once to
        # remove the kernel's c // 2 integer division. Derived cache, non-persistent.
        self.register_buffer("batch_freq_base", inv_freq.repeat_interleave(2), persistent=False)

        positions = torch.arange(max_seq_len, dtype=torch.float)
        positions_norm = positions / max_seq_len * (2 * math.pi)
        position_angles = positions_norm.unsqueeze(-1) * inv_freq
        position_angles = position_angles.repeat_interleave(2, dim=-1)
        self.register_buffer("position_angles", position_angles)

    def forward(self, timestamps: torch.Tensor, seq_len: int):
        B = timestamps.shape[0]
        SEQ = seq_len
        DIM = self.dim

        device = timestamps.device
        cos_out = torch.empty((B, SEQ, 2 * DIM), dtype=torch.float32, device=device)
        sin_out = torch.empty((B, SEQ, 2 * DIM), dtype=torch.float32, device=device)

        grid = (B * SEQ,)
        _rotary_embedding_row_kernel[grid](
            timestamps,
            self.batch_freq_base,
            self.position_angles,
            cos_out,
            sin_out,
            SEQ,
            self.max_seq_len,
            2.0 * math.pi,
            DIM=DIM,
            num_warps=4,
        )
        return cos_out, sin_out


def get_inputs():
    # timestamps: [batch_size, seq_len] — normalized song timestamps per time step
    B, SEQ = 4, 32
    timestamps = torch.rand(B, SEQ, device="npu")
    return [timestamps, SEQ]


def get_init_inputs():
    return [64, 256, 10000.0]


if __name__ == "__main__":
    model = ModelNew(*get_init_inputs()).npu().eval()
    with torch.no_grad():
        cos, sin = model(*get_inputs())
    print(cos.shape, sin.shape)
