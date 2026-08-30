import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mm_encoder_attention_kernel(
    q_ptr,             # [B, H, S, D] fp16 contiguous
    k_ptr,             # [B, H, S, D] fp16 contiguous
    v_ptr,             # [B, H, S, D] fp16 contiguous
    out_ptr,           # [B, H, S, D] fp16 contiguous
    scale,             # float (1/sqrt(D))
    S: tl.constexpr,   # real sequence length (83)
    D: tl.constexpr,   # head size (64)
    BLOCK_S: tl.constexpr,  # padded sequence length (next power of two >= S)
):
    pid = tl.program_id(0)  # pid = b * H + h

    offs_s = tl.arange(0, BLOCK_S)   # (BLOCK_S,)
    offs_d = tl.arange(0, D)         # (D,)

    row_mask = offs_s[:, None] < S   # (BLOCK_S, 1)

    # Q: [BLOCK_S, D]
    q_ptrs = q_ptr + pid * S * D + offs_s[:, None] * D + offs_d[None, :]
    q = tl.load(q_ptrs, mask=row_mask, other=0.0).to(tl.float32)

    # K^T: [D, BLOCK_S]  (load k transposed directly)
    kt_ptrs = k_ptr + pid * S * D + offs_s[None, :] * D + offs_d[:, None]
    kt = tl.load(kt_ptrs, mask=offs_s[None, :] < S, other=0.0).to(tl.float32)

    # V: [BLOCK_S, D]
    v_ptrs = v_ptr + pid * S * D + offs_s[:, None] * D + offs_d[None, :]
    v = tl.load(v_ptrs, mask=row_mask, other=0.0).to(tl.float32)

    # scores = q @ k^T * scale  -> [BLOCK_S, BLOCK_S]
    scores = tl.dot(q, kt) * scale

    # Mask invalid key columns to -inf so they contribute nothing to softmax.
    scores = tl.where(offs_s[None, :] < S, scores, float("-inf"))

    # Numerically-stable softmax over the key dimension (axis=1).
    m = tl.max(scores, axis=1)                 # (BLOCK_S,)
    p = tl.exp(scores - m[:, None])            # (BLOCK_S, BLOCK_S)
    denom = tl.sum(p, axis=1)                  # (BLOCK_S,)
    p = p / denom[:, None]

    # out = p @ v  -> [BLOCK_S, D]
    acc = tl.dot(p, v)

    out_ptrs = out_ptr + pid * S * D + offs_s[:, None] * D + offs_d[None, :]
    tl.store(out_ptrs, acc.to(tl.float16), mask=row_mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


class ModelNew(nn.Module):
    def __init__(self, num_heads: int = 8, head_size: int = 64, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = 1.0 / (head_size ** 0.5)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        bsz, q_len = query.size()[:2]
        H = self.num_heads
        D = self.head_size

        # [B, S, H*D] -> [B, H, S, D] (contiguous for the kernel)
        q = query.view(bsz, q_len, H, D).transpose(1, 2).contiguous()
        k = key.view(bsz, q_len, H, D).transpose(1, 2).contiguous()
        v = value.view(bsz, q_len, H, D).transpose(1, 2).contiguous()

        out = torch.empty((bsz, H, q_len, D), dtype=query.dtype, device=query.device)

        BLOCK_S = _next_pow2(q_len)
        grid = (bsz * H,)
        _mm_encoder_attention_kernel[grid](
            q, k, v, out,
            self.scale,
            S=q_len,
            D=D,
            BLOCK_S=BLOCK_S,
        )

        return out.transpose(1, 2).reshape(bsz, q_len, H * D)


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
    init_inputs = get_init_inputs()
    model = ModelNew(*init_inputs).cuda().eval()
    inputs = get_inputs()
    with torch.no_grad():
        out = model(*inputs)
    print(out.shape)
