"""v6: torch_mlu_ops library pipeline (moe_softmax_topk + moe_gen_idx + group_gemm x2 + combine).

对比 v5 单 Triton kernel。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlu  # noqa: F401
import torch_mlu_ops as tmo


def fused_moe_v6_out(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    T, H = hidden_states.shape
    E = w1.shape[0]
    dtype = hidden_states.dtype

    # 1. routing: tmo fused softmax + topk + renorm + scaling (single op)
    reduce_weight, expert_id = tmo.moe_softmax_topk(
        router_logits,
        topk=top_k,
        normalize=True,
        num_expert_group=-1,
        route_scale=1.0,
    )
    # reduce_weight: [T, K] fp32, expert_id: [T, K] int32

    # 2. gen expand/combine indices (single op, returns 4 tensors)
    expand_idx, combine_idx, token_count, _cusum = tmo.moe_gen_idx(expert_id, E)

    # 3. first group_gemm: hidden [T, H] -expand_idx-> [T*K, 2I]
    w1_h = w1.to(dtype)
    gate_up = tmo.group_gemm(
        hidden_states, w1_h, token_count, expand_idx,
        None, None, None,
        max_in_group_list=T * top_k,
        trans_a=False, trans_b=True,
    )  # [T*K, 2I]

    # 4. SiLU * up (PyTorch eager, 2 kernels)
    gate, up = gate_up.chunk(2, dim=-1)
    act = F.silu(gate) * up  # [T*K, I]

    # 5. second group_gemm: act [T*K, I] -> [T*K, H]
    w2_h = w2.to(dtype)
    out_exp = tmo.group_gemm(
        act, w2_h, token_count, expand_idx,
        None, None, None,
        max_in_group_list=T * top_k,
        trans_a=False, trans_b=True,
    )  # [T*K, H] in expert-sorted order

    # 6. combine: gather back to (T, K) order, weight, sum
    out_token_order = out_exp[combine_idx]  # [T*K, H]
    out_weighted = out_token_order * reduce_weight.view(-1, 1).to(dtype)
    out.copy_(out_weighted.view(T, top_k, H).sum(dim=1))
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
        self._out_cache: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        out = self._out_cache
        if out is None or out.shape != hidden_states.shape or out.device != hidden_states.device:
            out = torch.empty_like(hidden_states)
            self._out_cache = out
        return fused_moe_v6_out(
            hidden_states, router_logits, self.w1, self.w2, out, self.top_k
        )


def get_inputs():
    num_tokens, hidden_size, num_experts = 83, 128, 8
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device="cuda")
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")
    return [hidden_states, router_logits]


def get_init_inputs():
    return [8, 2, 128, 64]
