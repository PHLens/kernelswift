import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rotary_embed_fused_kernel(
    timestamps_ptr,       # [B, SEQ] fp32
    inv_freq_ptr,         # [dim//2] fp32
    position_angles_ptr,  # [MAX_SEQ, dim] fp32 (already repeat_interleave(2))
    cos_ptr,              # [B, SEQ, 2*dim] fp32
    sin_ptr,              # [B, SEQ, 2*dim] fp32
    total,                # B * SEQ * 2*dim
    D2,                   # dim // 2  (inv_freq length)
    DIM,                  # dim       (width of each cat half)
    OUT_DIM,              # 2 * dim   (output last-dim width)
    MAX_SEQ,              # max_seq_len
    B,                    # batch size
    SEQ,                  # seq_len
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    # Decompose flat index -> (b, s, j) with j in [0, 2*dim)
    b = offs // (SEQ * OUT_DIM)
    rem = offs % (SEQ * OUT_DIM)
    s = rem // OUT_DIM
    j = rem % OUT_DIM

    # batch half j in [0, dim): batch_freqs[b, j] = (b / MAX_SEQ) * inv_freq[j // 2]
    half_mask = j < DIM
    inv_idx = j // 2
    inv_freq_val = tl.load(inv_freq_ptr + inv_idx, mask=mask, other=0.0)
    b_f = (b.to(tl.float32) / MAX_SEQ) * inv_freq_val

    # position half j in [dim, 2*dim): position_angles[s, j - dim]
    pos_col = j - DIM
    pos_idx = s * DIM + pos_col
    tfa = tl.load(position_angles_ptr + pos_idx, mask=mask & (~half_mask), other=0.0)

    freq_val = tl.where(half_mask, b_f, tfa)

    ts = tl.load(timestamps_ptr + b * SEQ + s, mask=mask, other=0.0)
    angle = -ts * (2.0 * math.pi)
    x = freq_val * angle

    tl.store(cos_ptr + offs, tl.cos(x), mask=mask)
    tl.store(sin_ptr + offs, tl.sin(x), mask=mask)


class ModelNew(nn.Module):
    """MusicFlamingoRotaryEmbedding: batch (song) + time positional embedding.
    Returns (cos, sin) where cos/sin combines batch and time frequencies."""

    def __init__(self, dim: int = 64, max_seq_len: int = 256, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        self.register_buffer("inv_freq", inv_freq)

        positions = torch.arange(max_seq_len, dtype=torch.float)
        positions_norm = positions / max_seq_len * (2 * math.pi)
        position_angles = positions_norm.unsqueeze(-1) * inv_freq
        position_angles = position_angles.repeat_interleave(2, dim=-1)
        self.register_buffer("position_angles", position_angles)

    def forward(self, timestamps: torch.Tensor, seq_len: int):
        # Fast path: fused Triton-MACA elementwise kernel. Guard matches the
        # benchmark contract: fp32 contiguous cuda, seq_len==32, dim==64,
        # max_seq_len==256, no grad.
        if (
            self.dim == 64
            and self.max_seq_len == 256
            and seq_len == 32
            and timestamps.is_cuda
            and timestamps.dtype == torch.float32
            and timestamps.is_contiguous()
            and not timestamps.requires_grad
        ):
            return self._forward_fused(timestamps, seq_len)

        # Unchanged PyTorch fallback (copied from base.py).
        batch_positions = torch.arange(
            timestamps.shape[0], device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        batch_positions = batch_positions / self.max_seq_len
        batch_freqs = batch_positions.unsqueeze(-1) * self.inv_freq
        batch_freqs = batch_freqs.repeat_interleave(2, dim=-1)

        batch_freqs = batch_freqs[:, None, :]
        time_freqs = self.position_angles[:seq_len][None, :, :]
        batch_freqs, time_freqs = torch.broadcast_tensors(batch_freqs, time_freqs)
        freqs = torch.cat((batch_freqs, time_freqs), dim=-1)
        angle = (-timestamps * 2 * math.pi).to(freqs)
        freqs = freqs * angle.unsqueeze(-1)
        return freqs.cos(), freqs.sin()

    def _forward_fused(self, timestamps: torch.Tensor, seq_len: int):
        B, SEQ = timestamps.shape
        DIM = self.dim
        D2 = DIM // 2
        OUT_DIM = 2 * DIM
        inv_freq = self.inv_freq
        position_angles = self.position_angles
        device = timestamps.device

        cos = torch.empty((B, SEQ, OUT_DIM), device=device, dtype=torch.float32)
        sin = torch.empty((B, SEQ, OUT_DIM), device=device, dtype=torch.float32)

        total = B * SEQ * OUT_DIM
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)

        _rotary_embed_fused_kernel[grid](
            timestamps,
            inv_freq,
            position_angles,
            cos,
            sin,
            total,
            D2,
            DIM,
            OUT_DIM,
            self.max_seq_len,
            B,
            SEQ,
            BLOCK=BLOCK,
            num_warps=1,
        )
        return cos, sin


def get_inputs():
    # timestamps: [batch_size, seq_len] — normalized song timestamps per time step
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
