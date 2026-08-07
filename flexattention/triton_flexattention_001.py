"""v1: Single Triton kernel implementing fused SDPA (causal + GQA ratio 1).

Replaces 22 eager kernels (BMM + softmax + cast + transpose + ...) with one
Triton kernel: one program per (query_token, head), online-equivalent softmax
in a single block over the padded sequence length.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_mlu  # noqa: F401


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@triton.jit
def _sdpa_v1_kernel(
    q_ptr,           # [T, H, D] fp16
    k_ptr,           # [T, H, D] fp16
    v_ptr,           # [T, H, D] fp16
    out_ptr,         # [T, H, D] fp16
    scale,
    H: tl.constexpr,
    D: tl.constexpr,
    T_BLOCK: tl.constexpr,  # padded seq len (next_pow2(T))
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)

    d_idx = tl.arange(0, D)
    t_idx = tl.arange(0, T_BLOCK)

    # Causal mask: key index <= token_id (others masked to -inf)
    mask = t_idx <= token_id

    # Load q: [D]
    q_off = token_id * H * D + head_id * D + d_idx
    q = tl.load(q_ptr + q_off).to(tl.float32)

    # Load k, v: [T_BLOCK, D] with masked rows zeroed (so they don't affect sums)
    kv_off = t_idx[:, None] * H * D + head_id * D + d_idx[None, :]
    k = tl.load(k_ptr + kv_off, mask=mask[:, None], other=0.0).to(tl.float32)
    v = tl.load(v_ptr + kv_off, mask=mask[:, None], other=0.0).to(tl.float32)

    # QK^T: [T_BLOCK]
    qk = tl.sum(q[None, :] * k, axis=1) * scale
    qk = tl.where(mask, qk, -float("inf"))

    # Softmax (one pass: T_BLOCK small enough for a single block)
    m_i = tl.max(qk, axis=0)
    p = tl.exp(qk - m_i)
    p = p / tl.sum(p, axis=0)

    # AV: [D]
    out = tl.sum(p[:, None] * v, axis=0)

    tl.store(out_ptr + q_off, out.to(tl.float16))


def _sdpa_v1(query, key, value, scale, num_heads, num_kv_heads):
    T = query.shape[0]
    H = num_heads
    D = query.shape[-1]
    out = torch.empty(T, H, D, dtype=query.dtype, device=query.device)

    if num_kv_heads != num_heads:
        # GQA ratio > 1 path: fall back to repeat_interleave then call kernel.
        r = H // num_kv_heads
        k = key.repeat_interleave(r, dim=1)
        v = value.repeat_interleave(r, dim=1)
    else:
        k = key
        v = value

    T_BLOCK = _next_pow2(T)
    grid = (T, H)
    with torch.mlu.device(query.device):
        _sdpa_v1_kernel[grid](
            query, k, v, out, scale,
            H=H, D=D, T_BLOCK=T_BLOCK,
            num_warps=1, num_stages=1,
        )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_heads: int = 8, head_size: int = 64,
                 scale: float = None, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale or 1.0 / (head_size ** 0.5)
        self.num_kv_heads = num_kv_heads

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        out = _sdpa_v1(query, key, value, self.scale, self.num_heads, self.num_kv_heads)
        T = query.shape[0]
        return out.reshape(T, self.num_heads * self.head_size)


def get_inputs():
    num_tokens, num_heads, head_size = 83, 8, 64
    dtype = torch.float16
    query = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cuda")
    key = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cuda")
    value = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cuda")
    return [query, key, value]


def get_init_inputs():
    return [8, 64, None, 8]


if __name__ == "__main__":
    init_inputs = get_init_inputs()
    model = ModelNew(*init_inputs).cuda().eval()
    inputs = get_inputs()
    with torch.no_grad():
        out = model(*inputs)
    print(out.shape)
