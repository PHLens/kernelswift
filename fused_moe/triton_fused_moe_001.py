"""First Triton fused-MoE kernel: one program per token, two GEMMs via elementwise sum."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_mlu  # noqa: F401 - registers the MLU device


@triton.jit
def _fused_moe_per_token_kernel(
    hidden_ptr,        # [T, H] fp16
    topk_ids_ptr,      # [T, K] int32
    topk_weights_ptr,  # [T, K] fp16
    w1_ptr,            # [E, 2I, H] fp16
    w2_ptr,            # [E, H, I] fp16
    out_ptr,           # [T, H] fp16
    H: tl.constexpr,
    I: tl.constexpr,
    TWO_I: tl.constexpr,
    K: tl.constexpr,
):
    token_id = tl.program_id(0)
    h_idx = tl.arange(0, H)
    i_idx = tl.arange(0, I)
    two_i_idx = tl.arange(0, TWO_I)

    x = tl.load(hidden_ptr + token_id * H + h_idx)  # [H] fp16
    out_acc = tl.zeros((H,), dtype=tl.float32)

    for k in tl.static_range(0, K):
        expert_id = tl.load(topk_ids_ptr + token_id * K + k)  # int32 scalar
        weight = tl.load(topk_weights_ptr + token_id * K + k)  # fp16 scalar

        # w1[expert_id]: [2I, H]
        w1_off = expert_id.to(tl.int64) * TWO_I * H + two_i_idx[:, None] * H + h_idx[None, :]
        w1_block = tl.load(w1_ptr + w1_off)  # [2I, H] fp16

        # gate_up = x @ w1_block.T  -> [2I]
        gate_up = tl.sum(
            x[None, :].to(tl.float32) * w1_block.to(tl.float32), axis=1
        )  # [2I]

        gate = tl.reshape(gate_up[:I], (I,))
        up = tl.reshape(gate_up[I:], (I,))
        act = (gate * (1.0 / (1.0 + tl.exp(-gate)))) * up  # [I]

        # w2[expert_id]: [H, I]
        w2_off = expert_id.to(tl.int64) * H * I + h_idx[:, None] * I + i_idx[None, :]
        w2_block = tl.load(w2_ptr + w2_off)  # [H, I] fp16

        # out_k = act @ w2_block.T  -> [H]
        out_k = tl.sum(
            act[None, :].to(tl.float32) * w2_block.to(tl.float32), axis=1
        )  # [H]

        out_acc += weight.to(tl.float32) * out_k

    tl.store(out_ptr + token_id * H + h_idx, out_acc.to(x.dtype))


def fused_moe_triton(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    renormalize: bool = True,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    dtype = hidden_states.dtype

    scores = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)
    topk_ids = topk_ids.to(torch.int32)

    out = torch.empty_like(hidden_states)
    with torch.mlu.device(hidden_states.device):
        _fused_moe_per_token_kernel[(num_tokens,)](
            hidden_states,
            topk_ids,
            topk_weights,
            w1,
            w2,
            out,
            H=hidden_states.shape[1],
            I=w2.shape[-1],
            TWO_I=w1.shape[1],
            K=top_k,
            num_warps=1,
            num_stages=1,
        )
    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize
        self.w1 = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.w2 = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        return fused_moe_triton(
            hidden_states,
            router_logits,
            self.w1,
            self.w2,
            self.top_k,
            self.renormalize,
        )


def get_inputs():
    num_tokens, hidden_size, num_experts = 83, 128, 8
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device="cuda")
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")
    return [hidden_states, router_logits]


def get_init_inputs():
    return [8, 2, 128, 64]
