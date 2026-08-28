import torch
import torch.nn as nn
import triton
import triton.language as tl


_BM = 32
_BN = 32
_BD = 32


@triton.jit
def _mm_encoder_attn_fwd(q_ptr, k_ptr, v_ptr, o_ptr, scale,
                         B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                         D: tl.constexpr, NM: tl.constexpr, NT: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    """One blocked bidirectional full-attention program over (batch, head, q-tile).

    grid = B * H * ceil(S / BM) programs; each program owns BM query rows of
    one (batch, head) pair and loops the ceil(S / BN) key tiles SEQUENTIALLY
    with an online running-max softmax in fp32 (the running state depends on
    the key tiles, so they cannot be parallel programs). Inputs are fp16
    [B, S, H*D] row-major tensors addressed DIRECTLY by their frozen
    constexpr strides (batch stride S*H*D, token stride H*D, head stride D);
    no layout-copying host op exists. Every fp16 tile is WIDENED to fp32
    immediately after its load so all four dot call sites stay
    (BM, BD) @ (BD, BN) and (BM, BN) @ (BN, BD) — i.e. (32,32)@(32,32) with
    fp32 operands and fp32 accumulator, the proven envelope. Keys are loaded
    directly in transposed layout, so no trans op exists. Padding keys
    (token index >= S) are masked to -inf PRE-softmax (exp(-inf) == 0
    exactly, so padded keys contribute exactly zero); the attention is
    bidirectional — every real key tile is visited, no causal skip exists.
    Results are stored DIRECTLY into the final [B, S, H*D] fp16 token-major
    layout — no view, copy, or relayout op exists anywhere.

    The arithmetic is IDENTICAL to the round-001 kernel; only the launch
    warp count differs (two warps per program, 64 threads), the
    round-002 pre-adoption sweep having shown the outputs to be warp-count
    invariant here while the kernel-only device time drops at the target
    shape.
    """
    pid = tl.program_id(0)
    bh = pid % (B * H)
    mtile = pid // (B * H)
    b = bh // H
    h = bh % H

    offs_m = mtile * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, BD)
    mask_m = offs_m < S
    head_off = b * (S * H * D) + h * D
    row_off = offs_m[:, None] * (H * D)

    q_lo = tl.load(q_ptr + head_off + row_off + offs_d[None, :],
                   mask=mask_m[:, None], other=0.0).to(tl.float32)
    q_hi = tl.load(q_ptr + head_off + row_off + BD + offs_d[None, :],
                   mask=mask_m[:, None], other=0.0).to(tl.float32)

    m_run = tl.full([BM], float('-inf'), dtype=tl.float32)
    l_run = tl.zeros([BM], dtype=tl.float32)
    acc_lo = tl.zeros([BM, BD], dtype=tl.float32)
    acc_hi = tl.zeros([BM, BD], dtype=tl.float32)

    for ntile in tl.static_range(NT):
        offs_n = ntile * BN + tl.arange(0, BN)
        mask_n = offs_n < S
        col_off = offs_n[None, :] * (H * D)

        k_lo_t = tl.load(k_ptr + head_off + col_off + offs_d[:, None],
                         mask=mask_n[None, :], other=0.0).to(tl.float32)
        k_hi_t = tl.load(k_ptr + head_off + col_off + BD + offs_d[:, None],
                         mask=mask_n[None, :], other=0.0).to(tl.float32)

        s = tl.dot(q_lo, k_lo_t) + tl.dot(q_hi, k_hi_t)
        s = s * scale
        s = tl.where(mask_n[None, :], s, float('-inf'))

        v_lo = tl.load(v_ptr + head_off + offs_n[:, None] * (H * D) + offs_d[None, :],
                       mask=mask_n[:, None], other=0.0).to(tl.float32)
        v_hi = tl.load(v_ptr + head_off + offs_n[:, None] * (H * D) + BD + offs_d[None, :],
                       mask=mask_n[:, None], other=0.0).to(tl.float32)

        m_new = tl.maximum(m_run, tl.max(s, axis=1))
        alpha = tl.exp(m_run - m_new)
        p = tl.exp(s - m_new[:, None])
        l_run = l_run * alpha + tl.sum(p, axis=1)
        acc_lo = acc_lo * alpha[:, None] + tl.dot(p, v_lo)
        acc_hi = acc_hi * alpha[:, None] + tl.dot(p, v_hi)
        m_run = m_new

    out_lo = (acc_lo / l_run[:, None]).to(tl.float16)
    out_hi = (acc_hi / l_run[:, None]).to(tl.float16)
    tl.store(o_ptr + head_off + row_off + offs_d[None, :], out_lo,
             mask=mask_m[:, None])
    tl.store(o_ptr + head_off + row_off + BD + offs_d[None, :], out_hi,
             mask=mask_m[:, None])


class ModelNew(nn.Module):
    """Bidirectional full MHA encoder attention (mm_encoder_attention), decision-002.

    Kernel-config variant of the banked round-001 deliverable (family
    triton-attention-kernel-config): the ENTIRE round-001 boundary stays —
    TWO python-visible ops per forward (one torch.empty + ONE direct kernel
    launch), the run_out 4-arg surface with a single launch writing the
    caller buffer, grid B*H*ceil(S/BM) = 48 programs at the target regime,
    (32,32) tiles with fp32-widened operands at all four dot call sites,
    online fp32 running-max softmax, direct strided addressing of the fp16
    [B, S, H*D] inputs and direct final-layout stores (zero layout-copying
    host ops). ONLY the kernel execution config varies: the launch pins the
    two-warp configuration (fixed module literal at the single launch site —
    no runtime selection, no config search machinery; the final candidate
    carries exactly ONE config), selected by the round-002
    pre-adoption sweep over warp counts and dot-operand dtypes: fp16-operand
    dots (even at a fp32 accumulator) failed the exactness gate on the
    fp16-extreme suite (one-hot tie-flip divergence, vendor-class signature)
    and warp counts above two bought no further kernel-only time, while the
    two-warp config reproduced the round-001 exactness numbers on every
    suite and cut the kernel-only device time at the target shape. There is
    NO workspace, NO cross-call state of any kind; the only persistent
    objects are this module's config attributes and the framework-owned
    Triton JIT compile cache (one-time compile at first call, absorbed by
    harness warmup).
    """

    def __init__(self, num_heads: int = 8, head_size: int = 64,
                 num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = 1.0 / head_size ** 0.5

    def _launch(self, query, key, value, out):
        """THE single launch site: one direct Triton kernel launch, grid
        (B*H*ceil(S/BM),) = (16*3,) = 48 programs at the target regime,
        two-warp configuration (the selected round-002 execution config as a
        fixed literal), no staging-count knob; shape constexprs derive from
        the live tensor shape, so any non-target shape simply routes to a
        fresh framework JIT specialization (stateless)."""
        (bsz, seq_len, hidden) = query.shape
        nm = (seq_len + _BM - 1) // _BM
        _mm_encoder_attn_fwd[(bsz * self.num_heads * nm,)](
            query, key, value, out, self.scale,
            B=bsz, S=seq_len, H=self.num_heads, D=self.head_size,
            NM=nm, NT=(seq_len + _BN - 1) // _BN,
            BM=_BM, BN=_BN, BD=_BD,
            num_warps=2,
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        out = torch.empty(query.shape, dtype=query.dtype, device=query.device)
        self._launch(query, key, value, out)
        return out

    def run_out(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, out: torch.Tensor) -> None:
        """Preallocated-output surface (project.md public_contract): ONE
        direct kernel launch writing the caller-provided buffer; zero
        allocations; returns None; bitwise-equal to forward for identical
        inputs (same kernel, same bits, deterministic launch configuration)."""
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
