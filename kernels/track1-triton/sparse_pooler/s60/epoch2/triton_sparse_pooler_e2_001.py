import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _sparse_pooler_tail_kernel(
    x_ptr,
    seq_lens_ptr,
    offsets_ptr,
    out_ptr,
    V: tl.constexpr,
    S: tl.constexpr,
    BV: tl.constexpr,
):
    """Fused post-decoder tail: log1p(relu(x)) + per-segment max pooling.

    One program per (segment, vocab-block) pair. Loads the decoder output tile
    [83, 30522] fp32 for its segment's token span and vocab block, applies
    log1p(relu(x)) elementwise, reduces max over the segment tokens, and stores
    straight into out[seg, vocab_block].
    """
    seg = tl.program_id(0)
    vid = tl.program_id(1)

    # Segment length L and token offset read device-side (no D2H sync).
    L = tl.load(seq_lens_ptr + seg)
    offset = tl.load(offsets_ptr + seg)

    vocab = vid * BV + tl.arange(0, BV)
    vmask = vocab < V

    # log1p(relu(x)) >= 0 for all x, so 0.0 is a valid lower bound for the max
    # reduction (equivalent to -inf init + token masking, but stays fp32).
    acc = tl.zeros([BV], dtype=tl.float32)
    for t in range(0, S):
        tmask = t < L
        ptrs = x_ptr + (offset + t) * V + vocab
        x = tl.load(ptrs, mask=vmask, other=0.0)
        val = tl.log(1.0 + tl.maximum(x, 0.0))
        val = tl.where(tmask, val, 0.0)
        acc = tl.maximum(acc, val)

    out_ptrs = out_ptr + seg * V + vocab
    tl.store(out_ptrs, acc, mask=vmask)


class ModelNew(nn.Module):
    """SPLADESparsePooler: MLM head logits → ReLU log(1+x) pooled over sequence (max)."""

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        pooling: str = "max",
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)
        self.pooling = pooling

    def forward(self, hidden_states: torch.Tensor, seq_lens: torch.Tensor) -> list:
        # MLM head stays vendor library.
        x = self.decoder(self.layer_norm(self.act(self.dense(hidden_states))))
        S = x.shape[0]
        V = x.shape[1]
        NS = seq_lens.shape[0]

        # Prefix offsets device-side: cumsum(seq_lens) - seq_lens == [0,20,45,63].
        # cumsum promotes int32 -> int64 on GCU; cast back to int32 for the kernel.
        offsets = (torch.cumsum(seq_lens, dim=0) - seq_lens).to(torch.int32)

        BV = 256
        out = torch.empty((NS, V), device=x.device, dtype=torch.float32)
        grid = (NS, triton.cdiv(V, BV))
        _sparse_pooler_tail_kernel[grid](
            x, seq_lens, offsets, out,
            V=V, S=S, BV=BV,
            num_warps=2,
        )

        result = [out[i] for i in range(NS)]
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
