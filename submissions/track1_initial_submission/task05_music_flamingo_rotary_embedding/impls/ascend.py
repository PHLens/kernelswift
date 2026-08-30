import math
import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_npu


@triton.jit
def _rotary_embedding_fused_kernel(
    timestamps_ptr,
    inv_freq_ptr,
    position_angles_ptr,
    cos_out_ptr,
    sin_out_ptr,
    N,
    SEQ,
    DIM,
    MAX_SEQ,
    TWO_PI,
    BLOCK: tl.constexpr,
):
    # 1D grid over the flattened B*SEQ*2DIM output elements, blocked by BLOCK.
    # Each program decodes a contiguous run of output indices (b, t, c), selects
    # the batch (c < dim) vs time (c >= dim) frequency source, applies the shared
    # -timestamps[b,t]*2pi angle, and writes both cos and sin in one pass.
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N

    b = idx // (SEQ * 2 * DIM)
    t = (idx // (2 * DIM)) % SEQ
    c = idx % (2 * DIM)

    dim = DIM
    half = c // 2

    # Batch frequency source (valid only when c < dim). repeat_interleave(2)
    # maps output column c -> inv_freq[c // 2]. Clamp half into [0, dim/2) so the
    # load stays in-bounds when c >= dim (the value is discarded by tl.where).
    half_safe = tl.minimum(half, dim // 2 - 1)
    freq_batch_raw = tl.load(inv_freq_ptr + half_safe, mask=mask, other=0.0)

    # Time frequency source (valid only when c >= dim). position_angles[t, c-dim]
    # is precomputed in __init__; clamp c-dim into [0, dim) to stay in-bounds.
    c_minus_dim = c - dim
    c_minus_dim_safe = tl.maximum(c_minus_dim, 0)
    freq_time = tl.load(
        position_angles_ptr + t * dim + c_minus_dim_safe, mask=mask, other=0.0
    )

    b_f = b.to(tl.float32)
    max_seq_f = MAX_SEQ.to(tl.float32)
    freq_batch = freq_batch_raw * (b_f / max_seq_f)

    is_time = c >= dim
    freq = tl.where(is_time, freq_time, freq_batch)

    ts = tl.load(timestamps_ptr + b * SEQ + t, mask=mask, other=0.0)
    angle = -ts * TWO_PI
    value = freq * angle

    cosv = tl.cos(value)
    sinv = tl.sin(value)

    tl.store(cos_out_ptr + idx, cosv, mask=mask)
    tl.store(sin_out_ptr + idx, sinv, mask=mask)


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

        positions = torch.arange(max_seq_len, dtype=torch.float)
        positions_norm = positions / max_seq_len * (2 * math.pi)
        position_angles = positions_norm.unsqueeze(-1) * inv_freq
        position_angles = position_angles.repeat_interleave(2, dim=-1)
        self.register_buffer("position_angles", position_angles)

    def forward(self, timestamps: torch.Tensor, seq_len: int):
        B = timestamps.shape[0]
        SEQ = seq_len
        DIM = self.dim
        MAX_SEQ = self.max_seq_len
        TWO_DIM = 2 * DIM

        device = timestamps.device
        cos_out = torch.empty((B, SEQ, TWO_DIM), dtype=torch.float32, device=device)
        sin_out = torch.empty((B, SEQ, TWO_DIM), dtype=torch.float32, device=device)

        N = B * SEQ * TWO_DIM
        BLOCK = 128
        grid = (triton.cdiv(N, BLOCK),)
        _rotary_embedding_fused_kernel[grid](
            timestamps,
            self.inv_freq,
            self.position_angles,
            cos_out,
            sin_out,
            N,
            SEQ,
            DIM,
            MAX_SEQ,
            2.0 * math.pi,
            BLOCK=BLOCK,
            num_warps=1,
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
