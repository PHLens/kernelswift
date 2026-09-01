import torch
from dataclasses import dataclass
from torch import nn


@dataclass
class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 2600
    dim: int = 1024
    index_n_heads: int = 16
    index_head_dim: int = 64
    index_topk: int = 128
    q_lora_rank: int = 256
    rope_head_dim: int = 32


class ModelNew(nn.Module):
    def __init__(self, args, freqs_cis: torch.Tensor, kv_cache: torch.Tensor, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.compress_ratio = compress_ratio
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim ** -0.5 * self.n_heads ** -0.5
        self.freqs_cis = freqs_cis
        self.kv_cache = kv_cache
        max_seq_len = freqs_cis.size(0)
        max_candidates = kv_cache.size(1)
        token = torch.arange(max_seq_len, device=freqs_cis.device).view(max_seq_len, 1)
        candidate = torch.arange(max_candidates, device=freqs_cis.device).view(1, max_candidates)
        self.causal_mask = torch.where(candidate >= (token + 1) // compress_ratio, float("-inf"), 0.0).to(torch.bfloat16)
        self.valid_count = ((torch.arange(max_seq_len, device=freqs_cis.device) + 1) // compress_ratio).view(1, max_seq_len, 1)

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        batch_size, seq_len, _ = x.size()
        end_pos = start_pos + seq_len
        query = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        query_rope = query[..., -self.rope_head_dim:]
        query_complex = torch.view_as_complex(query_rope.float().unflatten(-1, (-1, 2)))
        rope = self.freqs_cis[start_pos:end_pos].view(1, seq_len, 1, -1)
        query_rope.copy_(torch.view_as_real(query_complex * rope).flatten(-2))
        weights = self.weights_proj(x) * self.softmax_scale
        scores = torch.einsum("bshd,btd->bsht", query, self.kv_cache[:batch_size, :end_pos // self.compress_ratio])
        scores.relu_().mul_(weights.unsqueeze(-1))
        scores = scores.sum(dim=2)
        if start_pos == 0:
            scores.add_(self.causal_mask[:seq_len, :end_pos // self.compress_ratio])
        indices = scores.topk(min(self.index_topk, end_pos // self.compress_ratio), dim=-1)[1]
        if start_pos == 0:
            valid = self.valid_count[:, :seq_len]
            if offset == 0:
                return torch.where(indices >= valid, -1, indices)
            return torch.where(indices >= valid, -1, indices + offset)
        if offset == 0:
            return indices
        return indices + offset


def get_init_inputs():
    compress_ratio = 4
    args = ModelArgs(
        max_batch_size=8,
        max_seq_len=2600,
        dim=1024,
        index_n_heads=16,
        index_head_dim=64,
        index_topk=128,
        q_lora_rank=256,
        rope_head_dim=32,
    )
    freqs = 1.0 / (10000.0 ** (torch.arange(0, args.rope_head_dim, 2).float() / args.rope_head_dim))
    freqs = torch.outer(torch.arange(args.max_seq_len, dtype=torch.float32), freqs).to(device="cuda")
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs).view(args.max_seq_len, -1)
    kv_cache = torch.randn(args.max_batch_size, args.max_seq_len // compress_ratio, args.index_head_dim, dtype=torch.bfloat16, device="cuda")
    return [args, freqs_cis, kv_cache, compress_ratio]


def get_inputs():
    args = ModelArgs(max_batch_size=8, max_seq_len=2600, dim=1024, index_n_heads=16, index_head_dim=64, index_topk=128, q_lora_rank=256, rope_head_dim=32)
    x = torch.randn(8, 2600, args.dim, dtype=torch.bfloat16, device="cuda")
    qr = torch.randn(8, 2600, args.q_lora_rank, dtype=torch.bfloat16, device="cuda")
    return [x, qr, 0, 0]
