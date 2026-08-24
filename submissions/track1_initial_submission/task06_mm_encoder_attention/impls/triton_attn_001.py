import torch
import torch.nn as nn
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _mm_enc_attn_kernel(
    Q,
    K,
    V,
    Out,
    scale,
    BATCH,
    SEQ,
    HEADS,
    HEAD_SIZE,
    HIDDEN,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program per (batch, head, seq) triple; 1D grid = BATCH * HEADS * SEQ.
    pid = tl.program_id(0)
    b = pid // (HEADS * SEQ)
    rem = pid % (HEADS * SEQ)
    h = rem // SEQ
    s = rem % SEQ

    d_off = tl.arange(0, BLOCK_D)          # [BLOCK_D] = head_size dim
    k_off = tl.arange(0, BLOCK_K)          # [BLOCK_K] = key/value seq dim

    # Native [bsz, seq, hidden] contiguous layout. head h data lives at
    # offset h*HEAD_SIZE + d within each row. Strided load of q row.
    q_base = b * SEQ * HIDDEN + s * HIDDEN + h * HEAD_SIZE
    q = tl.load(Q + q_base + d_off).to(tl.float32)  # [BLOCK_D]

    # Key block K[b, j, h*HEAD_SIZE : (h+1)*HEAD_SIZE] for j in [0, SEQ).
    k_base = b * SEQ * HIDDEN + h * HEAD_SIZE
    k_ptr = K + k_base + k_off[:, None] * HIDDEN + d_off[None, :]
    k_mask = k_off[:, None] < SEQ
    k = tl.load(k_ptr, mask=k_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_D]

    # scores[j] = sum_d q[d] * k[j, d] * scale  -> [BLOCK_K]
    scores = tl.sum(q[None, :] * k, axis=1) * scale

    # Mask padded positions (j >= SEQ) out of the softmax.
    scores = tl.where(k_off < SEQ, scores, float("-inf"))

    # Non-causal softmax over all SEQ positions.
    scores = scores - tl.max(scores, axis=0)
    probs = tl.exp(scores)
    probs = probs / tl.sum(probs, axis=0)

    # Value block V[b, j, h*HEAD_SIZE : (h+1)*HEAD_SIZE].
    v_base = b * SEQ * HIDDEN + h * HEAD_SIZE
    v_ptr = V + v_base + k_off[:, None] * HIDDEN + d_off[None, :]
    v = tl.load(v_ptr, mask=k_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_D]

    # acc[d] = sum_j probs[j] * v[j, d]  -> [BLOCK_D]
    acc = tl.sum(probs[:, None] * v, axis=0)

    # Store out[b, s, h*HEAD_SIZE : (h+1)*HEAD_SIZE] as fp16.
    out_ptr = Out + q_base + d_off
    tl.store(out_ptr, acc.to(tl.float16))


class ModelNew(nn.Module):

    def __init__(self, num_heads: int = 8, head_size: int = 64, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = 1.0 / (head_size ** 0.5)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        # query/key/value: [bsz, seq, hidden] fp16 contiguous, hidden = H * D.
        bsz, seq, hidden = query.shape
        heads = self.num_heads
        head_size = self.head_size

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        out = torch.empty((bsz, seq, hidden), dtype=query.dtype, device=query.device)

        BLOCK_D = 64
        BLOCK_K = 128  # >= seq (=83): single KV block, no loop needed.
        grid = (bsz * heads * seq,)
        _mm_enc_attn_kernel[grid](
            query, key, value, out,
            self.scale, bsz, seq, heads, head_size, hidden,
            BLOCK_D=BLOCK_D, BLOCK_K=BLOCK_K,
            num_warps=1,
        )

        return out


def get_inputs():
    # query/key/value: [bsz, seq, num_heads * head_size], float16.
    bsz, seq_len, num_heads, head_size, dtype = 2, 83, 8, 64, torch.float16
    hidden = num_heads * head_size
    query = torch.randn(bsz, seq_len, hidden, dtype=dtype, device="npu")
    key = torch.randn(bsz, seq_len, hidden, dtype=dtype, device="npu")
    value = torch.randn(bsz, seq_len, hidden, dtype=dtype, device="npu")
    return [query, key, value]


def get_init_inputs():
    # num_heads, head_size, num_kv_heads
    return [8, 64, 8]


if __name__ == "__main__":
    init_inputs = get_init_inputs()
    model = ModelNew(*init_inputs).npu().eval()
    inputs = get_inputs()
    with torch.no_grad():
        out = model(*inputs)
    print(out.shape)
