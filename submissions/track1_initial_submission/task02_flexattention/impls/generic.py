"""Vendor-neutral Triton fallback for the official benchmark shape.

This fallback intentionally imports only standard torch/triton modules.
It is used when the detected accelerator has no task-specific implementation.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _causal_attention_kernel(
    q_ptr,             # [S, H, D] fp16 contiguous
    k_ptr,             # [S, H, D] fp16 contiguous
    v_ptr,             # [S, H, D] fp16 contiguous
    out_ptr,           # [S, H, D] fp16 contiguous
    scale,             # float (1/sqrt(D))
    S: tl.constexpr,   # real sequence length (83)
    H: tl.constexpr,   # num heads (8)
    D: tl.constexpr,   # head size (64)
    BLOCK_S: tl.constexpr,  # padded sequence length (next power of two >= S)
):
    pid = tl.program_id(0)  # head index h

    offs_s = tl.arange(0, BLOCK_S)   # (BLOCK_S,)
    offs_d = tl.arange(0, D)         # (D,)

    row_mask = offs_s[:, None] < S   # (BLOCK_S, 1)

    # Per-head base offset = h * D (since layout is [S, H, D], element [s,h,d]
    # lives at s*H*D + h*D + d).
    base = pid * D

    # Q: [BLOCK_S, D]
    q_ptrs = q_ptr + base + offs_s[:, None] * H * D + offs_d[None, :]
    q = tl.load(q_ptrs, mask=row_mask, other=0.0).to(tl.float32)

    # K^T: [D, BLOCK_S]
    kt_ptrs = k_ptr + base + offs_s[None, :] * H * D + offs_d[:, None]
    kt = tl.load(kt_ptrs, mask=offs_s[None, :] < S, other=0.0).to(tl.float32)

    # V: [BLOCK_S, D]
    v_ptrs = v_ptr + base + offs_s[:, None] * H * D + offs_d[None, :]
    v = tl.load(v_ptrs, mask=row_mask, other=0.0).to(tl.float32)

    # scores = q @ k^T * scale  -> [BLOCK_S, BLOCK_S]
    scores = tl.dot(q, kt) * scale

    # Causal mask: query position m attends only to key positions n <= m.
    # Also mask padded key columns (n >= S) to -inf.
    causal_mask = offs_s[:, None] >= offs_s[None, :]
    valid_key = offs_s[None, :] < S
    scores = tl.where(causal_mask & valid_key, scores, float("-inf"))

    # Numerically-stable softmax over the key dimension (axis=1).
    # Every valid query row keeps at least its diagonal (m == n) finite, so the
    # per-row max is finite and exp(-inf) == 0 handles masked columns exactly.
    m = tl.max(scores, axis=1)                 # (BLOCK_S,)
    p = tl.exp(scores - m[:, None])            # (BLOCK_S, BLOCK_S)
    denom = tl.sum(p, axis=1)                  # (BLOCK_S,)
    p = p / denom[:, None]

    # out = p @ v  -> [BLOCK_S, D]
    acc = tl.dot(p, v)

    out_ptrs = out_ptr + base + offs_s[:, None] * H * D + offs_d[None, :]
    tl.store(out_ptrs, acc.to(tl.float16), mask=row_mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


class ModelNew(nn.Module):
    def __init__(self, num_heads: int = 8, head_size: int = 64,
                 scale: float = None, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = scale if scale is not None else 1.0 / (head_size ** 0.5)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        S = query.shape[0]
        H = self.num_heads
        D = self.head_size

        # query/key/value are contiguous [S, H, D]; the kernel reads them in
        # place with a per-head stride of H*D, so no transpose materialization
        # is needed.
        out = torch.empty((S, H, D), dtype=query.dtype, device=query.device)

        BLOCK_S = _next_pow2(S)
        grid = (H,)
        _causal_attention_kernel[grid](
            query, key, value, out,
            self.scale,
            S=S,
            H=H,
            D=D,
            BLOCK_S=BLOCK_S,
        )

        # [S, H, D] -> [S, H*D] == [83, 512]
        return out.reshape(S, H * D)


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
