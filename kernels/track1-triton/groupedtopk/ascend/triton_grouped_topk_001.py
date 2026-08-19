import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _grouped_topk_kernel(
    gating_ptr,
    weights_ptr,
    ids_ptr,
    T,
    E: tl.constexpr,
    n_group: tl.constexpr,
    epg: tl.constexpr,
    K: tl.constexpr,
    KG: tl.constexpr,
    renorm: tl.constexpr,
    scaling: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= T:
        return

    offs = tl.arange(0, BLOCK_E)
    e_mask = offs < E
    values = tl.load(
        gating_ptr + pid * E + offs,
        mask=e_mask,
        other=-float("inf"),
    ).to(tl.float32)

    row_max = tl.max(values, axis=0)
    exp_values = tl.exp(values - row_max)
    scores = exp_values / tl.sum(exp_values, axis=0)

    group_scores = tl.max(tl.reshape(scores, (n_group, epg)), axis=1)
    group_offsets = tl.arange(0, n_group)
    group_keep = tl.zeros((n_group,), tl.int1)
    remaining_groups = group_scores

    for _ in tl.static_range(KG):
        group_id = tl.argmax(remaining_groups, axis=0)
        group_keep = group_keep | (group_offsets == group_id)
        remaining_groups = tl.where(
            group_offsets == group_id,
            -float("inf"),
            remaining_groups,
        )

    group_keep = tl.reshape(group_keep, (n_group, 1))
    expert_keep = tl.reshape(
        tl.broadcast_to(group_keep, (n_group, epg)),
        (BLOCK_E,),
    )
    remaining = tl.where(expert_keep, scores, -float("inf"))

    expert_offsets = tl.arange(0, BLOCK_E)
    output_offsets = tl.arange(0, K)
    output_weights = tl.zeros((K,), tl.float32)
    output_ids = tl.zeros((K,), tl.int32)

    for k in tl.static_range(K):
        expert_id = tl.argmax(remaining, axis=0)
        expert_value = tl.max(remaining, axis=0)
        output_mask = output_offsets == k
        output_weights = tl.where(
            output_mask,
            tl.full((K,), expert_value, tl.float32),
            output_weights,
        )
        output_ids = tl.where(
            output_mask,
            tl.full((K,), expert_id, tl.int32),
            output_ids,
        )
        remaining = tl.where(
            expert_offsets == expert_id,
            -float("inf"),
            remaining,
        )

    if renorm:
        output_weights = output_weights / tl.sum(output_weights, axis=0)
    if scaling != 1.0:
        output_weights = output_weights * scaling

    tl.store(weights_ptr + pid * K + output_offsets, output_weights)
    tl.store(ids_ptr + pid * K + output_offsets, output_ids)


class ModelNew(nn.Module):
    def __init__(
        self,
        topk: int,
        renormalize: bool,
        num_expert_group: int,
        topk_group: int,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.topk = topk
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor, gating_output: torch.Tensor):
        del hidden_states
        if self.scoring_func != "softmax":
            raise ValueError("triton_ascend candidate supports softmax only")

        tokens, experts = gating_output.shape
        block_e = triton.next_power_of_2(experts)
        weights = torch.empty(
            (tokens, self.topk), dtype=torch.float32, device=gating_output.device
        )
        ids = torch.empty(
            (tokens, self.topk), dtype=torch.int32, device=gating_output.device
        )
        _grouped_topk_kernel[(tokens,)](
            gating_output,
            weights,
            ids,
            tokens,
            E=experts,
            n_group=self.num_expert_group,
            epg=experts // self.num_expert_group,
            K=self.topk,
            KG=self.topk_group,
            renorm=self.renormalize,
            scaling=self.routed_scaling_factor,
            BLOCK_E=block_e,
            num_warps=1,
        )
        return weights, ids


def get_inputs():
    num_tokens, hidden_size, num_experts = 83, 7168, 256
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.float16, device="npu"
    )
    gating_output = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device="npu"
    )
    return [hidden_states, gating_output]


def get_init_inputs():
    return [8, True, 8, 4]
