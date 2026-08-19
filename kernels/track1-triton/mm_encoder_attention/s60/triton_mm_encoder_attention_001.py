"""mm_encoder_attention: non-causal scaled-dot-product attention (encoder).

Semantics (matches base.py exactly):
  q/k/v reshaped to [bsz, num_heads, seq, head_size]
  out = F.scaled_dot_product_attention(q, k, v, scale=1/sqrt(head_size))
  out reshaped back to [bsz, seq, num_heads*head_size]

For this regime num_heads == num_kv_heads == 8 (no GQA repeat). Each Triton
program handles one query token: it computes scores against all keys (padded
to a power-of-two BLOCK) with elementwise tl.sum (no tl.dot, which is Unknown
on GCU), softmax, and the weighted value sum. This keeps the per-program local
memory small (a [BLOCK] score vector instead of a [BLOCK, BLOCK] tile).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_gcu  # noqa: F401 - registers the GCU device
import triton_gcu  # noqa: F401


@triton.jit
def _attn_kernel(
    q_ptr,      # [bsz, num_heads, seq, D] fp16
    k_ptr,      # [bsz, num_heads, seq, D] fp16
    v_ptr,      # [bsz, num_heads, seq, D] fp16
    out_ptr,    # [bsz, num_heads, seq, D] fp16
    seq,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    scale,
):
    pid = tl.program_id(0)
    bh = pid // seq
    q_idx = pid % seq
    base = bh * seq * D
    d = tl.arange(0, D)
    k_idx = tl.arange(0, BLOCK)
    k_mask = k_idx < seq

    # load query vector [D]
    q_vec = tl.load(q_ptr + base + q_idx * D + d).to(tl.float32)

    # load K/V tiles [BLOCK, D]
    K = tl.load(
        k_ptr + base + k_idx[:, None] * D + d[None, :],
        mask=k_mask[:, None], other=0.0,
    ).to(tl.float32)
    V = tl.load(
        v_ptr + base + k_idx[:, None] * D + d[None, :],
        mask=k_mask[:, None], other=0.0,
    ).to(tl.float32)

    # scores[k] = sum_d q_vec[d] * K[k,d] * scale
    scores = tl.sum(q_vec[None, :] * K, axis=1) * scale

    # softmax over keys
    scores = tl.where(k_mask, scores, float("-inf"))
    m = tl.max(scores, axis=0)
    e = tl.exp(scores - m)
    e = tl.where(k_mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # out[d] = sum_k p[k] * V[k,d]
    out = tl.sum(p[:, None] * V, axis=0)

    tl.store(out_ptr + base + q_idx * D + d, out.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, num_heads: int = 8, head_size: int = 64, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = 1.0 / (head_size ** 0.5)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        head_size = self.head_size
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads

        q = query.view(bsz, q_len, num_heads, head_size).transpose(1, 2).contiguous()
        k = key.view(bsz, kv_len, num_kv_heads, head_size).transpose(1, 2).contiguous()
        v = value.view(bsz, kv_len, num_kv_heads, head_size).transpose(1, 2).contiguous()

        out = torch.empty((bsz, num_heads, q_len, head_size), dtype=query.dtype, device=query.device)

        seq = q_len
        BLOCK = 128
        grid = (bsz * num_heads * seq,)
        _attn_kernel[grid](
            q,
            k,
            v,
            out,
            seq,
            D=head_size,
            BLOCK=BLOCK,
            scale=self.scale,
            num_warps=1,
        )

        return out.transpose(1, 2).reshape(bsz, q_len, -1)


def get_inputs():
    bsz, seq_len, num_heads, head_size, dtype = 2, 83, 8, 64, torch.float16
    hidden = num_heads * head_size
    query = torch.randn(bsz, seq_len, hidden, dtype=dtype, device="cuda")
    key = torch.randn(bsz, seq_len, hidden, dtype=dtype, device="cuda")
    value = torch.randn(bsz, seq_len, hidden, dtype=dtype, device="cuda")
    return [query, key, value]


def get_init_inputs():
    return [8, 64, 8]


if __name__ == "__main__":
    model = ModelNew(*get_init_inputs()).eval()
    with torch.no_grad():
        out = model(*get_inputs())
    print(out.shape)
