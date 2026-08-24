import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_moe_expert_kernel(
    hidden_ptr,      # [T, H] fp16
    flat_ids_ptr,    # [T*K] int64
    flat_w_ptr,      # [T*K] fp16
    w1_ptr,          # [E, 2*I, H] fp16
    w2_ptr,          # [E, H, I] fp16
    out_ptr,         # [T, H] fp16 (zero-initialized)
    T: tl.constexpr,     # num_tokens = 83
    K: tl.constexpr,     # top_k = 2
    E: tl.constexpr,     # num_experts = 8
    H: tl.constexpr,     # hidden = 128
    I: tl.constexpr,     # intermediate = 64
    BLOCK_M: tl.constexpr,  # power-of-two >= T*K (166 -> 256)
):
    e = tl.program_id(0)  # expert index

    rm = tl.arange(0, BLOCK_M)              # [BLOCK_M]
    valid = rm < T * K                       # [BLOCK_M] bool
    ids = tl.load(flat_ids_ptr + rm, mask=valid, other=-1)   # [BLOCK_M] int64
    is_e = ids == e                          # [BLOCK_M] bool (this expert's rows)
    w = tl.load(flat_w_ptr + rm, mask=valid, other=0.0).to(tl.float32)  # [BLOCK_M]
    w = tl.where(is_e, w, 0.0)               # zero out non-expert rows

    token = rm // K                          # [BLOCK_M] token index

    rk = tl.arange(0, H)                     # hidden dim
    rn = tl.arange(0, I)                     # intermediate dim

    # gather x rows: hidden[token, :] -> [BLOCK_M, H], non-expert rows zeroed
    x = tl.load(
        hidden_ptr + token[:, None] * H + rk[None, :],
        mask=is_e[:, None] & (rk[None, :] < H),
        other=0.0,
    )                                        # [BLOCK_M, H] fp16

    # w1[e] is [2*I, H]; gate weights = first I rows, up weights = last I rows.
    # We want x @ w1[e].T -> [BLOCK_M, 2*I]; do it as two dots with N=I.
    w1_base = w1_ptr + e * (2 * I * H)
    gate_w = tl.load(w1_base + rn[:, None] * H + rk[None, :])          # [I, H] fp16
    up_w = tl.load(w1_base + (I + rn)[:, None] * H + rk[None, :])      # [I, H] fp16

    acc_g = tl.zeros((BLOCK_M, I), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, I), dtype=tl.float32)
    gate = tl.dot(x, tl.trans(gate_w), acc_g)   # [BLOCK_M, I] fp32
    up = tl.dot(x, tl.trans(up_w), acc_u)       # [BLOCK_M, I] fp32

    act = gate * tl.sigmoid(gate) * up   # [BLOCK_M, I]  (SiLU(gate) * up)

    # w2[e] is [H, I]; down GEMM act @ w2[e].T -> [BLOCK_M, H]
    w2_base = w2_ptr + e * (H * I)
    w2e = tl.load(w2_base + rk[:, None] * I + rn[None, :])             # [H, I] fp16
    acc_y = tl.zeros((BLOCK_M, H), dtype=tl.float32)
    y = tl.dot(act.to(tl.float16), tl.trans(w2e), acc_y)   # [BLOCK_M, H] fp32

    contrib = y * w[:, None]             # [BLOCK_M, H] weighted

    # atomic accumulate into out[token, :]
    tl.atomic_add(out_ptr + token[:, None] * H + rk[None, :], contrib, mask=is_e[:, None])


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

    def forward(
        self,
        hidden_states: torch.Tensor,   # [T, H]
        router_logits: torch.Tensor,   # [T, E]  float32
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        dtype = hidden_states.dtype
        H = self.hidden_size
        I = self.intermediate_size
        E = self.num_experts
        K = self.top_k

        # --- routing (preserved exactly) ---
        scores = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(scores, self.top_k, dim=-1)
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        flat_ids = topk_ids.view(-1)          # [T*K] int64
        flat_w = topk_weights.view(-1)        # [T*K] fp16

        w1 = self.w1.to(dtype)                # [E, 2*I, H]
        w2 = self.w2.to(dtype)                # [E, H, I]

        out = torch.zeros((num_tokens, H), dtype=dtype, device=hidden_states.device)

        BLOCK_M = triton.next_power_of_2(num_tokens * K)
        _fused_moe_expert_kernel[(E,)](
            hidden_states, flat_ids, flat_w, w1, w2, out,
            T=num_tokens, K=K, E=E, H=H, I=I, BLOCK_M=BLOCK_M,
        )
        return out


def get_inputs():
    num_tokens, hidden_size, num_experts = 83, 128, 8
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device="cuda")
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")
    return [hidden_states, router_logits]


def get_init_inputs():
    return [8, 2, 128, 64]


if __name__ == "__main__":
    init_inputs = get_init_inputs()
    model = ModelNew(*init_inputs).cuda().eval()
    inputs = get_inputs()
    with torch.no_grad():
        out = model(*inputs)
    if isinstance(out, (tuple, list)):
        for o in out:
            if hasattr(o, "shape"):
                print(o.shape)
    else:
        print(out.shape)
