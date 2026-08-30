import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mm_encoder_attn_fwd(q_ptr, k_ptr, v_ptr, o_ptr, scale,
                         B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                         D: tl.constexpr, TP: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % H
    b = pid // H

    offs_t = tl.arange(0, TP)
    offs_d = tl.arange(0, D)
    mask_t = offs_t < S
    base = b * (S * H * D) + h * D

    q = tl.load(q_ptr + base + offs_t[:, None] * (H * D) + offs_d[None, :],
                mask=mask_t[:, None], other=0.0).to(tl.float32)
    k = tl.load(k_ptr + base + offs_t[:, None] * (H * D) + offs_d[None, :],
                mask=mask_t[:, None], other=0.0).to(tl.float32)
    v = tl.load(v_ptr + base + offs_t[:, None] * (H * D) + offs_d[None, :],
                mask=mask_t[:, None], other=0.0).to(tl.float32)

    s = tl.dot(q, tl.trans(k)) * scale
    s = tl.where(mask_t[None, :], s, float('-inf'))
    m = tl.max(s, axis=1)
    p = tl.exp(s - m[:, None])
    l = tl.sum(p, axis=1)
    attn = p / l[:, None]
    out = tl.dot(attn, v)
    out = tl.where(mask_t[:, None], out, 0.0)
    tl.store(o_ptr + base + offs_t[:, None] * (H * D) + offs_d[None, :],
             out.to(tl.float16), mask=mask_t[:, None])


class ModelNew(nn.Module):
    def __init__(self, num_heads: int = 8, head_size: int = 64,
                 num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = 1.0 / head_size ** 0.5

    def _launch(self, query, key, value, out):
        (bsz, seq_len, hidden) = query.shape
        _mm_encoder_attn_fwd[(bsz * self.num_heads,)](
            query, key, value, out, self.scale,
            B=bsz, S=seq_len, H=self.num_heads, D=self.head_size,
            TP=128,
            num_warps=2,
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        out = torch.empty(query.shape, dtype=query.dtype, device=query.device)
        self._launch(query, key, value, out)
        return out

    def run_out(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, out: torch.Tensor) -> None:
        self._launch(query, key, value, out)
        return None


def get_inputs():
    (bsz, seq_len, num_heads, head_size, dtype) = (2, 83, 8, 64, torch.float16)
    hidden = num_heads * head_size
    query = torch.randn(bsz, seq_len, hidden, dtype=dtype, device='cuda')
    key = torch.randn(bsz, seq_len, hidden, dtype=dtype, device='cuda')
    value = torch.randn(bsz, seq_len, hidden, dtype=dtype, device='cuda')
    return [query, key, value]


def get_init_inputs():
    return [8, 64, 8]
