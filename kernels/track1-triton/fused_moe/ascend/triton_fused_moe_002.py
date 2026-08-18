"""Round 2: fuse routing (softmax + top-2 argmax + renormalize) into the kernel.

Builds on round 001: the per-token Triton kernel now takes `router_logits [T,E]`
fp32 directly and performs softmax (max-subtract + exp + sum), top-2 selection by
repeated argmax over E=8 experts, and renormalize inside the kernel. The expert
FFN (gate/up GEMM via two independent `tl.sum` rank-1 outer-products, SiLU via
`tl.exp`, down GEMM via `tl.sum`) and weighted reduce are unchanged from round 1.
No `tl.dot`.
"""

import torch
import torch.nn as nn
import torch_npu  # noqa: F401 - registers the Ascend NPU device
import triton
import triton.language as tl


@triton.jit
def _fused_moe_per_token_kernel(
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
    RENORMALIZE: tl.constexpr,
):
    token_id = tl.program_id(0)
    h_idx = tl.arange(0, H)          # [H]
    i_idx = tl.arange(0, I)          # [I]
    e_idx = tl.arange(0, E)          # [E]
    k_idx = tl.arange(0, K)          # [K]
    gate_idx = tl.arange(0, I)       # [I]  gate rows: w1[e, 0:I, :]
    up_idx = tl.arange(0, I) + I     # [I]  up rows:   w1[e, I:2I, :]

    # --- routing: softmax over E experts (fp32, matches base.py) ---
    logits = tl.load(router_logits_ptr + token_id * E + e_idx)  # [E] fp32
    max_logit = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logit)
    scores = exp_logits / tl.sum(exp_logits, axis=0)  # [E] fp32

    # --- top-K by repeated argmax (E=8, no sort/winner-tree/gather) ---
    remaining = scores
    topk_vals = tl.zeros((K,), dtype=tl.float32)
    topk_ids = tl.zeros((K,), dtype=tl.int32)
    for k in tl.static_range(0, K):
        best_val = tl.max(remaining, axis=0)
        best_id = tl.argmax(remaining, axis=0)  # lowest index on tie (matches torch.topk)
        topk_vals = tl.where(k_idx == k, best_val, topk_vals)
        topk_ids = tl.where(k_idx == k, best_id, topk_ids)
        remaining = tl.where(e_idx == best_id, float("-inf"), remaining)

    # --- renormalize top-k weights (fp32), then cast to fp16 (matches base.py) ---
    if RENORMALIZE:
        topk_weights = topk_vals / tl.sum(topk_vals, axis=0)
    else:
        topk_weights = topk_vals
    topk_weights = topk_weights.to(tl.float16)  # base.py casts to dtype (fp16)

    # --- expert FFN + weighted reduce (unchanged from round 001) ---
    x = tl.load(hidden_ptr + token_id * H + h_idx)  # [H] fp16
    x_f32 = x.to(tl.float32)
    out_acc = tl.zeros((H,), dtype=tl.float32)

    for k in tl.static_range(0, K):
        mask_k = k_idx == k
        expert_id = tl.sum(tl.where(mask_k, topk_ids, 0), axis=0)      # int32 scalar
        weight = tl.sum(tl.where(mask_k, topk_weights, 0.0), axis=0)   # fp16 scalar
        base = expert_id.to(tl.int64)

        # gate: w1[e, 0:I, :]  ->  gate[j] = sum_h x[h] * w1[e, j, h]  ->  [I]
        gate_off = base * TWO_I * H + gate_idx[:, None] * H + h_idx[None, :]
        gate_block = tl.load(w1_ptr + gate_off)  # [I, H] fp16
        gate = tl.sum(x_f32[None, :] * gate_block.to(tl.float32), axis=1)  # [I]

        # up: w1[e, I:2I, :]  ->  up[j] = sum_h x[h] * w1[e, I+j, h]  ->  [I]
        up_off = base * TWO_I * H + up_idx[:, None] * H + h_idx[None, :]
        up_block = tl.load(w1_ptr + up_off)  # [I, H] fp16
        up = tl.sum(x_f32[None, :] * up_block.to(tl.float32), axis=1)  # [I]

        # silu(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
        act = (gate * (1.0 / (1.0 + tl.exp(-gate)))) * up  # [I]

        # w2[expert_id]: [H, I]
        w2_off = base * H * I + h_idx[:, None] * I + i_idx[None, :]
        w2_block = tl.load(w2_ptr + w2_off)  # [H, I] fp16

        # out_k[h] = sum_i act[i] * w2[e, h, i]  ->  [H]
        out_k = tl.sum(
            act[None, :].to(tl.float32) * w2_block.to(tl.float32), axis=1
        )  # [H]

        out_acc += weight.to(tl.float32) * out_k

    tl.store(out_ptr + token_id * H + h_idx, out_acc.to(tl.float16))


class ModelNew(nn.Module):

    def __init__(self, num_experts: int, top_k: int, hidden_size: int, intermediate_size: int, renormalize: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize
        self.w1 = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        num_tokens = hidden_states.shape[0]

        w1 = self.w1.to(dtype)   # [E, 2I, H]
        w2 = self.w2.to(dtype)   # [E, H, I]

        out = torch.empty_like(hidden_states)
        _fused_moe_per_token_kernel[(num_tokens,)](
            hidden_states,
            router_logits,
            w1,
            w2,
            out,
            H=self.hidden_size,
            I=self.intermediate_size,
            TWO_I=2 * self.intermediate_size,
            K=self.top_k,
            E=self.num_experts,
            RENORMALIZE=self.renormalize,
            num_warps=1,
        )
        return out


def get_inputs():
    num_tokens, hidden_size, num_experts = 83, 128, 8
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device="npu")
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="npu")
    return [hidden_states, router_logits]


def get_init_inputs():
    return [8, 2, 128, 64]


if __name__ == "__main__":
    init_inputs = get_init_inputs()
    model = ModelNew(*init_inputs).npu().eval()
    inputs = get_inputs()
    with torch.no_grad():
        out = model(*inputs)
    print(out.shape)
