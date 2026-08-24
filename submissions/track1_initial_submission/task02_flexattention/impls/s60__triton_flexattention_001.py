"""flexattention: causal scaled-dot-product attention.

Semantics (matches base.py exactly):
  q = query.unsqueeze(0).transpose(1,2)   -> [1, num_heads, T, D]
  k/v likewise
  out = F.scaled_dot_product_attention(q, k, v, scale=1/sqrt(D), is_causal=True)
  return out.squeeze(0).transpose(0,1).reshape(T, num_heads * D)

For this regime num_heads == num_kv_heads == 8 (no GQA repeat). Each Triton
program handles one (head, query_token): scores against all keys with the
causal mask (key_idx <= query_idx), softmax, weighted value sum. GEMM uses
elementwise tl.sum (no tl.dot, which is Unknown on GCU). A single query token
is always a valid row, so the softmax can never be all -inf; the causal + range
mask is applied with tl.where to keep out-of-range keys at -inf (softmax -> 0).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_gcu  # noqa: F401 - registers the GCU device
import triton_gcu  # noqa: F401


@triton.jit
def _causal_attn_kernel(
    q_ptr,      # [num_heads, T, D] fp16
    k_ptr,      # [num_heads, T, D] fp16
    v_ptr,      # [num_heads, T, D] fp16
    out_ptr,    # [num_heads, T, D] fp16
    T,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    scale,
):
    pid = tl.program_id(0)
    h = pid // T
    q_idx = pid % T
    base = h * T * D
    d = tl.arange(0, D)
    k_idx = tl.arange(0, BLOCK)
    k_mask = k_idx < T

    q_vec = tl.load(q_ptr + base + q_idx * D + d).to(tl.float32)

    K = tl.load(
        k_ptr + base + k_idx[:, None] * D + d[None, :],
        mask=k_mask[:, None], other=0.0,
    ).to(tl.float32)
    V = tl.load(
        v_ptr + base + k_idx[:, None] * D + d[None, :],
        mask=k_mask[:, None], other=0.0,
    ).to(tl.float32)

    scores = tl.sum(q_vec[None, :] * K, axis=1) * scale

    # causal mask: only keys with k_idx <= q_idx participate
    causal = k_mask & (k_idx <= q_idx)
    scores = tl.where(causal, scores, float("-inf"))

    m = tl.max(scores, axis=0)
    e = tl.exp(scores - m)
    e = tl.where(causal, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    out = tl.sum(p[:, None] * V, axis=0)

    tl.store(out_ptr + base + q_idx * D + d, out.to(tl.float16))


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
        num_tokens = query.shape[0]
        head_size = self.head_size
        num_heads = self.num_heads

        # [num_heads, T, D] (matches base transpose)
        q = query.transpose(0, 1).contiguous()
        k = key.transpose(0, 1).contiguous()
        v = value.transpose(0, 1).contiguous()

        out = torch.empty((num_heads, num_tokens, head_size),
                          dtype=query.dtype, device=query.device)

        BLOCK = 128
        grid = (num_heads * num_tokens,)
        _causal_attn_kernel[grid](
            q, k, v, out,
            num_tokens,
            D=head_size,
            BLOCK=BLOCK,
            scale=self.scale,
            num_warps=1,
        )

        return out.transpose(0, 1).reshape(num_tokens, num_heads * head_size)


def get_inputs():
    num_tokens, num_heads, head_size = 83, 8, 64
    dtype = torch.float16
    query = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cuda")
    key   = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cuda")
    value = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cuda")
    return [query, key, value]


def get_init_inputs():
    return [8, 64, None, 8]


if __name__ == "__main__":
    model = ModelNew(*get_init_inputs()).eval()
    with torch.no_grad():
        out = model(*get_inputs())
    print(out.shape)
