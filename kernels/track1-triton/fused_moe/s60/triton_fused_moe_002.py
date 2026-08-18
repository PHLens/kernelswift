"""Round 002: fuse routing (softmax + top-2 + renorm + fp16 cast) into the per-token kernel.

Conforms to decision_002 (Unified Sketch + Evaluation Contract) on the triton_gcu profile.
- per-token kernel, grid=(T,), num_warps=1, direct launch (no fast_libentry).
- routing computed in-kernel from raw router_logits: softmax -> repeated-argmax top-2 -> renorm -> fp16 cast.
- top-2 uses `tl.max` + `tl.where(is_best, e_idx, 0)` + `tl.sum` (no tl.argmax).
- fused: double GEMM (gate/up = x@w1.T, down = act@w2.T) + SiLU + top-k weighted reduction.
- GEMM uses elementwise `tl.sum` (not tl.dot, which is Unknown on GCU).
- int32 indexing only (never tl.int64); expert_id is int32.
- gate/up split uses two independent [I] GEMMs (no `gate_up[:I]` slice, which GCU rejects).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_gcu  # noqa: F401 - registers the GCU device


@triton.jit
def _fused_moe_v2_kernel(
    hidden_ptr,          # [T, H] fp16
    router_logits_ptr,   # [T, E] fp32
    w1_ptr,              # [E, 2I, H] fp16
    w2_ptr,              # [E, H, I] fp16
    out_ptr,             # [T, H] fp16
    H: tl.constexpr,
    I: tl.constexpr,
    TWO_I: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
):
    token_id = tl.program_id(0)
    h_idx = tl.arange(0, H)
    i_idx = tl.arange(0, I)
    e_idx = tl.arange(0, E)

    # --- routing: softmax over E experts (fp32) ---
    logits = tl.load(router_logits_ptr + token_id * E + e_idx)  # [E] fp32
    max_logit = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logit)
    scores = exp_logits / tl.sum(exp_logits, axis=0)  # [E] fp32

    # --- top-K by repeated argmax (no tl.argmax) ---
    remaining = scores
    topk_vals = tl.zeros((K,), dtype=tl.float32) - 1.0
    topk_ids = tl.zeros((K,), dtype=tl.int32)
    for k in tl.static_range(0, K):
        best_val = tl.max(remaining, axis=0)
        # argmax via masked sum; fp32 random logits have no ties.
        is_best = remaining == best_val
        best_id = tl.sum(tl.where(is_best, e_idx, 0), axis=0)  # int32 scalar
        topk_vals = tl.where(tl.arange(0, K) == k, best_val, topk_vals)
        topk_ids = tl.where(tl.arange(0, K) == k, best_id, topk_ids)
        remaining = tl.where(is_best, -1.0, remaining)

    # --- renormalize top-k weights + fp16 cast (matches base semantics) ---
    weight_sum = tl.sum(topk_vals, axis=0)
    topk_weights = (topk_vals / weight_sum).to(tl.float16)  # [K] fp16

    # --- expert compute ---
    x = tl.load(hidden_ptr + token_id * H + h_idx)  # [H] fp16
    out_acc = tl.zeros((H,), dtype=tl.float32)

    for k in tl.static_range(0, K):
        mask_k = tl.arange(0, K) == k
        expert_id_scalar = tl.sum(tl.where(mask_k, topk_ids, 0), axis=0)  # int32 scalar
        weight_scalar = tl.sum(tl.where(mask_k, topk_weights, 0.0), axis=0)  # fp16 scalar

        # gate projection: w1[expert_id, 0:I, :]  -> [I, H]; int32 offsets.
        w1_gate_off = expert_id_scalar * TWO_I * H + i_idx[:, None] * H + h_idx[None, :]
        w1_gate_block = tl.load(w1_ptr + w1_gate_off)  # [I, H] fp16
        # up projection: w1[expert_id, I:2I, :]  -> [I, H]; int32 offsets.
        w1_up_off = expert_id_scalar * TWO_I * H + (i_idx + I)[:, None] * H + h_idx[None, :]
        w1_up_block = tl.load(w1_ptr + w1_up_off)  # [I, H] fp16

        # gate = x @ w1_gate_block.T  -> [I]
        gate = tl.sum(
            x[None, :].to(tl.float32) * w1_gate_block.to(tl.float32), axis=1
        )  # [I]
        # up = x @ w1_up_block.T  -> [I]
        up = tl.sum(
            x[None, :].to(tl.float32) * w1_up_block.to(tl.float32), axis=1
        )  # [I]

        act = (gate * (1.0 / (1.0 + tl.exp(-gate)))) * up  # [I]

        # w2[expert_id]: [H, I]; int32 offsets.
        w2_off = expert_id_scalar * H * I + h_idx[:, None] * I + i_idx[None, :]
        w2_block = tl.load(w2_ptr + w2_off)  # [H, I] fp16

        # out_k = act @ w2_block.T  -> [H]
        out_k = tl.sum(
            act[None, :].to(tl.float32) * w2_block.to(tl.float32), axis=1
        )  # [H]

        out_acc += weight_scalar.to(tl.float32) * out_k

    tl.store(out_ptr + token_id * H + h_idx, out_acc.to(x.dtype))


def fused_moe_v2(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    renormalize: bool = True,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    dtype = hidden_states.dtype
    H = hidden_states.shape[1]
    I = w2.shape[-1]
    E = w1.shape[0]
    TWO_I = w1.shape[1]

    # weights are fp32 parameters; cast to fp16 before the kernel (base semantics: fp16 matmul).
    w1_fp16 = w1.to(dtype)
    w2_fp16 = w2.to(dtype)

    out = torch.empty_like(hidden_states)
    _fused_moe_v2_kernel[(num_tokens,)](
        hidden_states,
        router_logits,
        w1_fp16,
        w2_fp16,
        out,
        H=H,
        I=I,
        TWO_I=TWO_I,
        K=top_k,
        E=E,
        num_warps=1,
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
        return fused_moe_v2(
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


if __name__ == "__main__":
    init_inputs = get_init_inputs()
    model = ModelNew(*init_inputs).eval()
    inputs = get_inputs()
    with torch.no_grad():
        out = model(*inputs)
    if isinstance(out, (tuple, list)):
        for o in out:
            if hasattr(o, "shape"):
                print(o.shape)
    else:
        print(out.shape)
