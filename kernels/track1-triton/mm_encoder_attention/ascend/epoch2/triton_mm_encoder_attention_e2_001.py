import torch
import torch_npu  # noqa: F401  must precede any NPU allocation
import triton
import triton.language as tl


@triton.jit
def _fused_attention_kernel(
    Q,
    K,
    V,
    O,
    stride_b,
    stride_s,
    S,
    HEAD_DIM: tl.constexpr,
    NH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    scale,
):
    """One program per (batch, head): full self-attention in a single kernel.

    q/k/v keep their native [B, S, NH*HEAD_DIM] layout and are indexed with
    explicit strides, so no transpose is ever materialized. K is loaded
    transposed as [HEAD_DIM, BLOCK_N] via strides, which avoids tl.trans because
    that primitive is not in the reviewed capability set for this target.
    """
    pid = tl.program_id(0)
    b = pid // NH
    h = pid % NH

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    row_mask = offs_m < S
    col_mask = offs_n < S

    base = b * stride_b + h * HEAD_DIM

    q = tl.load(
        Q + base + offs_m[:, None] * stride_s + offs_d[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    # [HEAD_DIM, BLOCK_N]: element [d, n] is K[b, n, h*HEAD_DIM + d]
    k_t = tl.load(
        K + base + offs_n[None, :] * stride_s + offs_d[:, None],
        mask=col_mask[None, :],
        other=0.0,
    )
    v = tl.load(
        V + base + offs_n[:, None] * stride_s + offs_d[None, :],
        mask=col_mask[:, None],
        other=0.0,
    )

    # (BLOCK_M, HEAD_DIM) @ (HEAD_DIM, BLOCK_N) with fp32 accumulation
    qk = tl.dot(q, k_t) * scale
    qk = tl.where(col_mask[None, :], qk, -1.0e6)

    row_max = tl.max(qk, axis=1)
    p = tl.exp(qk - row_max[:, None])
    l_i = tl.sum(p, axis=1)
    p = p / l_i[:, None]

    acc = tl.dot(p.to(tl.float16), v)

    tl.store(
        O + base + offs_m[:, None] * stride_s + offs_d[None, :],
        acc.to(tl.float16),
        mask=row_mask[:, None],
    )


class ModelNew(torch.nn.Module):
    def __init__(self, num_heads: int = 8, head_size: int = 64, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = 1.0 / (head_size**0.5)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        bsz, q_len, hidden = query.shape
        head_dim = self.head_size
        out = torch.empty_like(query)

        block = 128
        if q_len > block:
            raise ValueError(
                f"this candidate covers S <= {block}; got S={q_len}. "
                "A row-blocked loop is required before this shape can be served."
            )

        grid = (bsz * self.num_heads,)
        _fused_attention_kernel[grid](
            query,
            key,
            value,
            out,
            query.stride(0),
            query.stride(1),
            q_len,
            HEAD_DIM=head_dim,
            NH=self.num_heads,
            BLOCK_M=block,
            BLOCK_N=block,
            scale=self.scale,
            num_warps=4,
            num_stages=1,
        )
        return out


def get_inputs():
    bsz, seq_len, num_heads, head_size, dtype = 2, 83, 8, 64, torch.float16
    hidden = num_heads * head_size
    query = torch.randn(bsz, seq_len, hidden, dtype=dtype, device="npu")
    key = torch.randn(bsz, seq_len, hidden, dtype=dtype, device="npu")
    value = torch.randn(bsz, seq_len, hidden, dtype=dtype, device="npu")
    return [query, key, value]


def get_init_inputs():
    return [8, 64, 8]
