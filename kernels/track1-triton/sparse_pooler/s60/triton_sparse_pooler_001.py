import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import torch_gcu
import triton_gcu


@triton.jit
def _sparse_pooler_max_kernel(
    logits_ptr,
    seq_lens_ptr,
    out_ptr,
    vocab_size,
    stride_logits_row,
    stride_out_row,
    BLOCK_V: tl.constexpr,
):
    # Grid: (num_seq, num_vocab_tiles).
    pid_s = tl.program_id(0)
    pid_v = tl.program_id(1)

    # On-device prefix scan: seq_len for this sequence, and its offset as the
    # sum of all preceding sequence lengths. num_seq is small (4 in this
    # project), so this bounded in-kernel sum (at most 3 extra tl.load calls)
    # removes the D2H synchronization that seq_lens.tolist() triggers in the
    # baseline. All arithmetic stays in int32.
    seq_len = tl.load(seq_lens_ptr + pid_s).to(tl.int32)
    seq_offset = tl.zeros([], dtype=tl.int32)
    for i in range(pid_s):
        seq_offset = seq_offset + tl.load(seq_lens_ptr + i).to(tl.int32)

    v_start = pid_v * BLOCK_V
    v_offs = v_start + tl.arange(0, BLOCK_V)
    v_mask = v_offs < vocab_size

    # Accumulator for the per-segment max over the sequence axis.
    acc = tl.full((BLOCK_V,), float("-inf"), tl.float32)

    row_base = seq_offset * stride_logits_row
    for row in range(seq_len):
        row_offset = row_base + row * stride_logits_row
        x = tl.load(
            logits_ptr + row_offset + v_offs, mask=v_mask, other=float("-inf")
        )
        # relu
        x = tl.where(x > 0.0, x, 0.0)
        # log1p(x) = log(1 + x); numerically stable because relu output >= 0
        # so 1 + x >= 1. No tl.log1p primitive is required.
        x = tl.log(1.0 + x)
        # per-segment max update. tl.where is proven Supported on GCU; it is
        # the documented fallback for the unproven tl.maximum.
        acc = tl.where(acc < x, x, acc)

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
        # MLM head: dense -> GELU -> LayerNorm -> decoder. Library ops, unchanged.
        x = self.decoder(self.layer_norm(self.act(self.dense(hidden_states))))

        num_seq = seq_lens.shape[0]
        vocab_size = x.shape[1]
        device = x.device

        if self.pooling == "max":
            # Output: list of num_seq tensors each [vocab_size] fp32 on the
            # caller-selected device. Allocated per forward (no cross-forward
            # cache in this round).
            out = torch.empty(
                (num_seq, vocab_size),
                dtype=torch.float32,
                device=device,
            )
            BLOCK_V = 256
            num_vocab_tiles = (vocab_size + BLOCK_V - 1) // BLOCK_V
            grid = (num_seq, num_vocab_tiles)
            _sparse_pooler_max_kernel[grid](
                x,
                seq_lens,
                out,
                vocab_size,
                x.stride(0),
                out.stride(0),
                BLOCK_V=BLOCK_V,
                num_warps=1,
            )
            return [out[i] for i in range(num_seq)]

        # sum pooling fallback preserves the public contract for pooling == "sum".
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
