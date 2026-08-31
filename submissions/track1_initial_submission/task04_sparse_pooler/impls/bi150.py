import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _sparse_pooler_fused_kernel(
    logits_ptr,
    seq_lens_ptr,
    out_ptr,
    vocab_size,
    stride_logits_row,
    stride_out_row,
    BLOCK_V: tl.constexpr,
):
    # Grid: (num_seq, cdiv(vocab_size, BLOCK_V)).
    # Each program computes one BLOCK_V-wide tile of one sequence's pooled
    # output: it reads logits[seq_offset : seq_offset+seq_len, v_tile], applies
    # the SPLADE activation log1p(relu(x)) elementwise, and reduces the max
    # over the sequence axis, all without materializing the intermediate
    # [83, 30522] activation tensor.
    pid_s = tl.program_id(0)
    pid_v = tl.program_id(1)

    # On-device seq_len and seq_offset = sum(seq_lens[0:pid_s]). num_seq is
    # small (4), so a bounded prefix scan avoids the D2H sync that
    # seq_lens.tolist() triggers on the host. At most num_seq-1 (<= 3) extra
    # tl.loads per program.
    seq_len = tl.load(seq_lens_ptr + pid_s).to(tl.int32)
    seq_offset = tl.zeros([], dtype=tl.int32)
    for i in range(pid_s):
        seq_offset = seq_offset + tl.load(seq_lens_ptr + i).to(tl.int32)

    v_start = pid_v * BLOCK_V
    v_offs = v_start + tl.arange(0, BLOCK_V)
    v_mask = v_offs < vocab_size

    # Per-sequence max accumulator over the sequence axis.
    acc = tl.full((BLOCK_V,), -float("inf"), dtype=tl.float32)

    row_base = seq_offset * stride_logits_row
    for _row in range(seq_len):
        x = tl.load(
            logits_ptr + row_base + _row * stride_logits_row + v_offs,
            mask=v_mask,
            other=-float("inf"),
        )
        # SPLADE activation: relu(x) = max(x, 0), then log1p(relu(x)).
        # relu output is non-negative, so log(1 + x) is well-conditioned.
        x = tl.where(x > 0.0, x, 0.0)
        x = tl.log(1.0 + x)
        acc = tl.maximum(acc, x)

    tl.store(out_ptr + pid_s * stride_out_row + v_offs, acc, mask=v_mask)


class ModelNew(nn.Module):
    """SPLADESparsePooler: MLM head logits -> ReLU log(1+x) pooled over sequence (max or sum)."""

    def __init__(self, hidden_size: int = 768, vocab_size: int = 30522, pooling: str = "max"):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)
        self.pooling = pooling

    def forward(self, hidden_states: torch.Tensor, seq_lens: torch.Tensor) -> list:
        # MLM head: dense -> GELU -> LayerNorm -> decoder. Library ops, unchanged
        # (vendor TCU path preserved for both GEMMs).
        logits = self.decoder(self.layer_norm(self.act(self.dense(hidden_states))))

        num_seq = seq_lens.shape[0]
        vocab_size = logits.shape[1]
        device = logits.device

        if self.pooling == "max":
            out = torch.empty((num_seq, vocab_size), dtype=torch.float32, device=device)
            BLOCK_V = 1024
            num_vocab_tiles = triton.cdiv(vocab_size, BLOCK_V)
            grid = (num_seq, num_vocab_tiles)
            _sparse_pooler_fused_kernel[grid](
                logits,
                seq_lens,
                out,
                vocab_size,
                logits.stride(0),
                out.stride(0),
                BLOCK_V=BLOCK_V,
            )
            return [out[i] for i in range(num_seq)]

        # sum pooling fallback preserves the public contract for pooling == "sum".
        x = torch.log1p(F.relu(logits))
        result = []
        offset = 0
        for L in seq_lens.tolist():
            chunk = x[offset:offset + L]
            result.append(chunk.sum(dim=0))
            offset += L
        return result


def get_inputs():
    seq_lens = torch.tensor([20, 25, 18, 20], dtype=torch.int32, device="cuda")
    hidden_states = torch.randn(83, 768, device="cuda")
    return [hidden_states, seq_lens]


def get_init_inputs():
    return [768, 30522, "max"]


if __name__ == "__main__":
    model = ModelNew(*get_init_inputs()).eval()
    with torch.no_grad():
        out = model(*get_inputs())
    for o in out:
        print(o.shape)
