import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _flex_attn_fwd(q_ptr, k_ptr, v_ptr, o_ptr, scale,
                   S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                   TP: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % H
    offs_m = tl.arange(0, TP)
    offs_n = tl.arange(0, TP)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < S
    mask_n = offs_n < S
    qh = tl.load(q_ptr + offs_m[:, None] * (H * D) + h * D + offs_d[None, :],
                 mask=mask_m[:, None], other=0.0)
    kh = tl.load(k_ptr + offs_n[:, None] * (H * D) + h * D + offs_d[None, :],
                 mask=mask_n[:, None], other=0.0)
    vh = tl.load(v_ptr + offs_n[:, None] * (H * D) + h * D + offs_d[None, :],
                 mask=mask_n[:, None], other=0.0).to(tl.float32)
    s = tl.dot(qh, tl.trans(kh)) * scale
    causal = offs_m[:, None] >= offs_n[None, :]
    s = tl.where(causal & mask_n[None, :], s, float('-inf'))
    m = tl.max(s, axis=1)
    p = tl.exp(s - m[:, None])
    l = tl.sum(p, axis=1)
    attn = p / l[:, None]
    out = tl.dot(attn, vh)
    out = tl.where(mask_m[:, None], out, 0.0)
    tl.store(o_ptr + offs_m[:, None] * (H * D) + h * D + offs_d[None, :],
             out.to(tl.float16), mask=mask_m[:, None])


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
        o = torch.empty(query.shape, dtype=query.dtype, device=query.device)
        _flex_attn_fwd[(self.num_heads,)](
            query, key, value, o, self.scale,
            S=num_tokens, H=self.num_heads, D=self.head_size,
            TP=128, num_warps=1,
        )
        return o.reshape(num_tokens, self.num_heads * self.head_size)

    def run_out(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, out: torch.Tensor) -> None:
        num_tokens = query.shape[0]
        _flex_attn_fwd[(self.num_heads,)](
            query, key, value, out, self.scale,
            S=num_tokens, H=self.num_heads, D=self.head_size,
            TP=128, num_warps=1,
        )
        return None


def get_inputs():
    num_tokens, num_heads, head_size = 83, 8, 64
    dtype = torch.float16
    query = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device='cuda')
    key = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device='cuda')
    value = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device='cuda')
    return [query, key, value]


def get_init_inputs():
    return [8, 64, None, 8]
