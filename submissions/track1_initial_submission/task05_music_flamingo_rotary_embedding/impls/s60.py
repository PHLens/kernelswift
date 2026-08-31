import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rotary_freqs_kernel(
    timestamps_ptr,
    inv_freq_ptr,
    position_angles_ptr,
    freqs_ptr,
    SEQ: tl.constexpr,       # seq_len (32)
    MAX_SEQ_LEN: tl.constexpr,  # max_seq_len (256)
    HALF: tl.constexpr,      # dim // 2 == 32 (power-of-2 tl.arange extent)
    D: tl.constexpr,         # dim == 64 (per-branch freq width)
):
    # grid = (B, seq_len): one program per (b, t) pair.
    b = tl.program_id(0)
    t = tl.program_id(1)

    # HALF=32 power-of-2 arange over the freq axis.
    i = tl.arange(0, HALF)

    # batch_freq = (b / max_seq_len) * inv_freq[i]  (float division, matches base)
    bpos = b.to(tl.float32) / MAX_SEQ_LEN
    bf = bpos * tl.load(inv_freq_ptr + i)

    # time_freq = position_angles[t, 2i] (even column; position_angles is
    # already repeat_interleave(2) so adjacent columns are duplicates).
    # position_angles row stride = 2 * HALF = D.
    tf = tl.load(position_angles_ptr + t * (2 * HALF) + i * 2)

    # angle = -timestamps[b, t] * 2 * pi  (scalar, fp32)
    angle = -tl.load(timestamps_ptr + b * SEQ + t) * 6.283185307179586

    f_bf = bf * angle
    f_tf = tf * angle

    # freqs[b, t, :128] = [batch_freqs (64), time_freqs (64)]; write each f to
    # both adjacent columns 2i / 2i+1 to emulate repeat_interleave(2).
    base_off = b * (SEQ * (2 * D)) + t * (2 * D)
    tl.store(freqs_ptr + base_off + 2 * i, f_bf)
    tl.store(freqs_ptr + base_off + 2 * i + 1, f_bf)
    tl.store(freqs_ptr + base_off + D + 2 * i, f_tf)
    tl.store(freqs_ptr + base_off + D + 2 * i + 1, f_tf)


class ModelNew(nn.Module):
    """MusicFlamingoRotaryEmbedding: batch (song) + time positional embedding.

    PARTIAL fusion: a single direct-launched Triton kernel computes the freqs
    elementwise chain (div/mul/repeat_interleave/broadcast/cat/mul-angle) into
    one intermediate [B, seq_len, 128] fp32 buffer; cos/sin are RETAINED as
    vendor torch.cos/torch.sin on the host side (the kernel contains NO
    device-side trig — the epoch-1 full-fusion lesson).
    """

    def __init__(self, dim: int = 64, max_seq_len: int = 256, base: float = 10000.0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.dim = dim
        inv_freq = 1.0 / base ** (
            torch.arange(0, dim, 2, dtype=torch.float) / dim
        )
        self.register_buffer("inv_freq", inv_freq)

        positions = torch.arange(max_seq_len, dtype=torch.float)
        positions_norm = positions / max_seq_len * (2 * math.pi)
        position_angles = positions_norm.unsqueeze(-1) * inv_freq
        position_angles = position_angles.repeat_interleave(2, dim=-1)
        self.register_buffer("position_angles", position_angles)

    def forward(self, timestamps: torch.Tensor, seq_len: int):
        B = timestamps.shape[0]
        D = self.dim
        freqs = torch.empty(
            (B, seq_len, 2 * D), dtype=torch.float32, device=timestamps.device
        )
        _rotary_freqs_kernel[(B, seq_len)](
            timestamps,
            self.inv_freq,
            self.position_angles,
            freqs,
            SEQ=seq_len,
            MAX_SEQ_LEN=self.max_seq_len,
            HALF=D // 2,
            D=D,
            num_warps=1,
        )
        return freqs.cos(), freqs.sin()


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
