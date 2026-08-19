import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _weighted_reduce_kernel(
    y_sorted_ptr,     # [166, 128] fp16 row-major, expert outputs ordered by sort_idx
    inv_ptr,          # [166] int64: inv[sort_idx[i]] = i  (flat index -> sorted slot)
    flat_w_ptr,       # [166] fp16 weights
    out_ptr,          # [83, 128] fp16
    H: tl.constexpr,  # hidden size = 128
    BLOCK_H: tl.constexpr,  # power-of-two >= H
):
    # One program per token t in [0, 83).
    t = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    hmask = h < H

    # slot positions of this token's two (k=0, k=1) rows in the sorted buffer
    s0 = tl.load(inv_ptr + 2 * t)
    s1 = tl.load(inv_ptr + 2 * t + 1)

    w0 = tl.load(flat_w_ptr + 2 * t)
    w1 = tl.load(flat_w_ptr + 2 * t + 1)

    y0 = tl.load(y_sorted_ptr + s0 * H + h, mask=hmask, other=0.0)
    y1 = tl.load(y_sorted_ptr + s1 * H + h, mask=hmask, other=0.0)

    acc = y0.to(tl.float32) * w0.to(tl.float32) + y1.to(tl.float32) * w1.to(tl.float32)
    tl.store(out_ptr + t * H + h, acc.to(tl.float16), mask=hmask)


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

        # --- routing (preserved exactly) ---
        scores = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(scores, self.top_k, dim=-1)
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        flat_ids = topk_ids.view(-1)          # [T*top_k]
        flat_w = topk_weights.view(-1)        # [T*top_k]
        x_rep = (
            hidden_states.unsqueeze(1)
            .expand(-1, self.top_k, -1)
            .reshape(-1, H)
        )  # [T*top_k, H]

        w1 = self.w1.to(dtype)   # [E, 2*I, H]
        w2 = self.w2.to(dtype)   # [E, H, I]

        # --- fused dispatch: one sort replaces the per-expert CUB DeviceSelect ---
        # Sort flat_ids so each expert's rows become a contiguous block.
        sort_idx = torch.argsort(flat_ids, stable=False)   # [T*top_k]
        x_sorted = x_rep[sort_idx]                          # [T*top_k, H] (1 gather)
        counts = torch.bincount(flat_ids, minlength=self.num_experts)  # [E]
        offsets = torch.cumsum(counts, dim=0)               # [E] exclusive via slicing

        y_sorted = torch.zeros_like(x_sorted)               # [T*top_k, H]

        for e in range(self.num_experts):
            start = int(offsets[e].item()) - int(counts[e].item())
            end = int(offsets[e].item())
            if start >= end:
                continue
            x_e = x_sorted[start:end]                       # contiguous [n_e, H]
            gate_up = x_e @ w1[e].T                         # GEMM1 [n_e, 2*I]
            gate, up = gate_up.chunk(2, dim=-1)             # [n_e, I] each
            act = F.silu(gate) * up                         # [n_e, I]
            y_sorted[start:end] = act @ w2[e].T             # GEMM2 [n_e, H]

        # --- fused weighted reduction: one Triton kernel replaces scatter+scale+sum ---
        # inv[j] = sorted slot of flat row j; inv[sort_idx[i]] = i
        inv = torch.empty_like(sort_idx)
        inv[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)

        out = torch.empty((num_tokens, H), dtype=dtype, device=hidden_states.device)
        BLOCK_H = triton.next_power_of_2(H)
        _weighted_reduce_kernel[(num_tokens,)](
            y_sorted, inv, flat_w, out,
            H=H, BLOCK_H=BLOCK_H,
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
