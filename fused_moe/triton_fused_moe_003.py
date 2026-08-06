"""v3: fast_libentry launcher + cached output buffer to cut host overhead."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_mlu  # noqa: F401
from triton.runtime import fast_libentry


@triton.jit
def _fused_moe_v3_kernel(
    hidden_ptr,         # [T, H] fp16
    router_logits_ptr,  # [T, E] fp32
    w1_ptr,             # [E, 2I, H] fp16
    w2_ptr,             # [E, H, I] fp16
    out_ptr,            # [T, H] fp16
    H: tl.constexpr,
    I: tl.constexpr,
    TWO_I: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
):
    token_id = tl.program_id(0)
    h_idx = tl.arange(0, H)
    i_idx = tl.arange(0, I)
    two_i_idx = tl.arange(0, TWO_I)
    e_idx = tl.arange(0, E)

    # --- routing: softmax over E experts ---
    logits = tl.load(router_logits_ptr + token_id * E + e_idx)
    max_logit = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logit)
    scores = exp_logits / tl.sum(exp_logits, axis=0)

    # --- top-2 by repeated argmax ---
    remaining = scores
    topk_vals = tl.zeros((K,), dtype=tl.float32)
    topk_ids = tl.zeros((K,), dtype=tl.int32)
    for k in tl.static_range(0, K):
        best_val = tl.max(remaining, axis=0)
        is_best = remaining == best_val
        best_id = tl.sum(tl.where(is_best, e_idx, 0), axis=0).to(tl.int32)
        k_mask = tl.arange(0, K) == k
        topk_vals = tl.where(k_mask, best_val, topk_vals)
        topk_ids = tl.where(k_mask, best_id, topk_ids)
        remaining = tl.where(is_best, -1.0, remaining)

    weight_sum = tl.sum(topk_vals, axis=0)
    topk_weights = topk_vals / weight_sum

    # --- expert compute ---
    x = tl.load(hidden_ptr + token_id * H + h_idx)
    out_acc = tl.zeros((H,), dtype=tl.float32)

    for k in tl.static_range(0, K):
        k_mask = tl.arange(0, K) == k
        expert_id = tl.sum(tl.where(k_mask, topk_ids, 0), axis=0)
        weight = tl.sum(tl.where(k_mask, topk_weights, 0.0), axis=0)

        w1_off = expert_id.to(tl.int64) * TWO_I * H + two_i_idx[:, None] * H + h_idx[None, :]
        w1_block = tl.load(w1_ptr + w1_off)
        gate_up = tl.sum(x[None, :].to(tl.float32) * w1_block.to(tl.float32), axis=1)

        gate = tl.reshape(gate_up[:I], (I,))
        up = tl.reshape(gate_up[I:], (I,))
        act = (gate * (1.0 / (1.0 + tl.exp(-gate)))) * up

        w2_off = expert_id.to(tl.int64) * H * I + h_idx[:, None] * I + i_idx[None, :]
        w2_block = tl.load(w2_ptr + w2_off)
        out_k = tl.sum(act[None, :].to(tl.float32) * w2_block.to(tl.float32), axis=1)

        out_acc += weight * out_k

    tl.store(out_ptr + token_id * H + h_idx, out_acc.to(x.dtype))


_fused_moe_v3_fast = fast_libentry()(_fused_moe_v3_kernel)


def fused_moe_v3_out(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    H = hidden_states.shape[1]
    I = w2.shape[-1]
    E = w1.shape[0]
    TWO_I = w1.shape[1]

    with torch.mlu.device(hidden_states.device):
        _fused_moe_v3_fast[(num_tokens,)](
            hidden_states,
            router_logits,
            w1,
            w2,
            out,
            H=H,
            I=I,
            TWO_I=TWO_I,
            K=top_k,
            E=E,
            num_warps=1,
            num_stages=1,
        )
    return out


class ModelNew(nn.Module):
    if "_fused_moe_v3_fast" not in globals():
        globals()["_fused_moe_v3_fast"] = fast_libentry()(_fused_moe_v3_kernel)

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
        return fused_moe_v3_out(
            hidden_states, router_logits, self.w1, self.w2, out, self.top_k
        )


def get_inputs():
    num_tokens, hidden_size, num_experts = 83, 128, 8
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device="cuda")
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")
    return [hidden_states, router_logits]


def get_init_inputs():
    return [8, 2, 128, 64]
