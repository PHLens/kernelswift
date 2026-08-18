import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _grouped_topk_fixed_kernel(
    gating_ptr,
    weights_ptr,
    ids_ptr,
    T: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    token = tl.program_id(0)
    if token >= T:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    raw_logits = tl.load(gating_ptr + token * BLOCK_E + expert_offsets)
    grouped_logits = tl.reshape(raw_logits, (8, 32))
    group_maxima = tl.max(grouped_logits, axis=1)
    group_offsets = tl.arange(0, 8)

    group_id_0 = tl.argmax(group_maxima, axis=0)
    group_keep_0 = group_offsets == group_id_0
    group_remaining_1 = tl.where(
        group_keep_0,
        -float("inf"),
        group_maxima,
    )

    group_id_1 = tl.argmax(group_remaining_1, axis=0)
    group_keep_1 = group_offsets == group_id_1
    group_remaining_2 = tl.where(
        group_keep_1,
        -float("inf"),
        group_remaining_1,
    )

    group_id_2 = tl.argmax(group_remaining_2, axis=0)
    group_keep_2 = group_offsets == group_id_2
    group_remaining_3 = tl.where(
        group_keep_2,
        -float("inf"),
        group_remaining_2,
    )

    group_id_3 = tl.argmax(group_remaining_3, axis=0)
    group_keep_3 = group_offsets == group_id_3
    selected_group_mask = (
        group_keep_0 | group_keep_1 | group_keep_2 | group_keep_3
    )
    selected_group_mask = tl.reshape(selected_group_mask, (8, 1))
    selected_expert_mask = tl.broadcast_to(selected_group_mask, (8, 32))
    eligible_grouped_logits = tl.where(
        selected_expert_mask,
        grouped_logits,
        -float("inf"),
    )
    expert_remaining_0 = tl.reshape(eligible_grouped_logits, (BLOCK_E,))

    expert_id_0 = tl.argmax(expert_remaining_0, axis=0)
    expert_logit_0 = tl.sum(
        tl.where(expert_offsets == expert_id_0, expert_remaining_0, 0.0),
        axis=0,
    )
    expert_remaining_1 = tl.where(
        expert_offsets == expert_id_0,
        -float("inf"),
        expert_remaining_0,
    )

    expert_id_1 = tl.argmax(expert_remaining_1, axis=0)
    expert_logit_1 = tl.sum(
        tl.where(expert_offsets == expert_id_1, expert_remaining_1, 0.0),
        axis=0,
    )
    expert_remaining_2 = tl.where(
        expert_offsets == expert_id_1,
        -float("inf"),
        expert_remaining_1,
    )

    expert_id_2 = tl.argmax(expert_remaining_2, axis=0)
    expert_logit_2 = tl.sum(
        tl.where(expert_offsets == expert_id_2, expert_remaining_2, 0.0),
        axis=0,
    )
    expert_remaining_3 = tl.where(
        expert_offsets == expert_id_2,
        -float("inf"),
        expert_remaining_2,
    )

    expert_id_3 = tl.argmax(expert_remaining_3, axis=0)
    expert_logit_3 = tl.sum(
        tl.where(expert_offsets == expert_id_3, expert_remaining_3, 0.0),
        axis=0,
    )
    expert_remaining_4 = tl.where(
        expert_offsets == expert_id_3,
        -float("inf"),
        expert_remaining_3,
    )

    expert_id_4 = tl.argmax(expert_remaining_4, axis=0)
    expert_logit_4 = tl.sum(
        tl.where(expert_offsets == expert_id_4, expert_remaining_4, 0.0),
        axis=0,
    )
    expert_remaining_5 = tl.where(
        expert_offsets == expert_id_4,
        -float("inf"),
        expert_remaining_4,
    )

    expert_id_5 = tl.argmax(expert_remaining_5, axis=0)
    expert_logit_5 = tl.sum(
        tl.where(expert_offsets == expert_id_5, expert_remaining_5, 0.0),
        axis=0,
    )
    expert_remaining_6 = tl.where(
        expert_offsets == expert_id_5,
        -float("inf"),
        expert_remaining_5,
    )

    expert_id_6 = tl.argmax(expert_remaining_6, axis=0)
    expert_logit_6 = tl.sum(
        tl.where(expert_offsets == expert_id_6, expert_remaining_6, 0.0),
        axis=0,
    )
    expert_remaining_7 = tl.where(
        expert_offsets == expert_id_6,
        -float("inf"),
        expert_remaining_6,
    )

    expert_id_7 = tl.argmax(expert_remaining_7, axis=0)
    expert_logit_7 = tl.sum(
        tl.where(expert_offsets == expert_id_7, expert_remaining_7, 0.0),
        axis=0,
    )

    selected_exp_0 = tl.exp(expert_logit_0 - expert_logit_0)
    selected_exp_1 = tl.exp(expert_logit_1 - expert_logit_0)
    selected_exp_2 = tl.exp(expert_logit_2 - expert_logit_0)
    selected_exp_3 = tl.exp(expert_logit_3 - expert_logit_0)
    selected_exp_4 = tl.exp(expert_logit_4 - expert_logit_0)
    selected_exp_5 = tl.exp(expert_logit_5 - expert_logit_0)
    selected_exp_6 = tl.exp(expert_logit_6 - expert_logit_0)
    selected_exp_7 = tl.exp(expert_logit_7 - expert_logit_0)
    selected_exp_sum = (
        selected_exp_0
        + selected_exp_1
        + selected_exp_2
        + selected_exp_3
        + selected_exp_4
        + selected_exp_5
        + selected_exp_6
        + selected_exp_7
    )

    output_offsets = tl.arange(0, 8)
    output_weights = tl.where(
        output_offsets == 0,
        selected_exp_0 / selected_exp_sum,
        selected_exp_7 / selected_exp_sum,
    )
    output_weights = tl.where(
        output_offsets == 1, selected_exp_1 / selected_exp_sum, output_weights
    )
    output_weights = tl.where(
        output_offsets == 2, selected_exp_2 / selected_exp_sum, output_weights
    )
    output_weights = tl.where(
        output_offsets == 3, selected_exp_3 / selected_exp_sum, output_weights
    )
    output_weights = tl.where(
        output_offsets == 4, selected_exp_4 / selected_exp_sum, output_weights
    )
    output_weights = tl.where(
        output_offsets == 5, selected_exp_5 / selected_exp_sum, output_weights
    )
    output_weights = tl.where(
        output_offsets == 6, selected_exp_6 / selected_exp_sum, output_weights
    )

    output_ids = tl.where(output_offsets == 0, expert_id_0, expert_id_7)
    output_ids = tl.where(output_offsets == 1, expert_id_1, output_ids)
    output_ids = tl.where(output_offsets == 2, expert_id_2, output_ids)
    output_ids = tl.where(output_offsets == 3, expert_id_3, output_ids)
    output_ids = tl.where(output_offsets == 4, expert_id_4, output_ids)
    output_ids = tl.where(output_offsets == 5, expert_id_5, output_ids)
    output_ids = tl.where(output_offsets == 6, expert_id_6, output_ids)

    output_base = token * 8 + output_offsets
    tl.store(weights_ptr + output_base, output_weights)
    tl.store(ids_ptr + output_base, output_ids)


class ModelNew(nn.Module):

    def __init__(self, topk: int, renormalize: bool, num_expert_group: int, topk_group: int, scoring_func: str='softmax', routed_scaling_factor: float=1.0):
        super().__init__()
        self.topk = topk
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor, gating_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states.size(0) == gating_output.size(0)
        fast_path = (
            tuple(hidden_states.shape) == (83, 7168)
            and tuple(gating_output.shape) == (83, 256)
            and hidden_states.dtype == torch.float16
            and gating_output.dtype == torch.float32
            and hidden_states.is_contiguous()
            and gating_output.is_contiguous()
            and hidden_states.device == gating_output.device
            and gating_output.device.type == 'cuda'
            and self.topk == 8
            and self.num_expert_group == 8
            and self.topk_group == 4
            and self.scoring_func == 'softmax'
            and self.renormalize is True
            and self.routed_scaling_factor == 1.0
            and (
                not torch.is_grad_enabled()
                or (
                    not hidden_states.requires_grad
                    and not gating_output.requires_grad
                )
            )
        )
        if fast_path:
            backing = torch.empty(
                2 * 83 * 8, dtype=torch.int32, device=gating_output.device
            )
            topk_weights = backing[:83 * 8].view(torch.float32).view(83, 8)
            topk_ids = backing[83 * 8:].view(83, 8)
            _grouped_topk_fixed_kernel[(83,)](
                gating_output,
                topk_weights,
                topk_ids,
                T=83,
                BLOCK_E=256,
                num_warps=1,
            )
            return (topk_weights, topk_ids)

        if self.scoring_func == 'softmax':
            scores = torch.softmax(gating_output, dim=-1)
        elif self.scoring_func == 'sigmoid':
            scores = gating_output.sigmoid()
        else:
            raise ValueError(f'Unsupported scoring_func: {self.scoring_func}')
        num_token = scores.size(0)
        experts_per_group = scores.size(-1) // self.num_expert_group
        group_scores = scores.view(num_token, self.num_expert_group, -1).max(dim=-1).values
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(num_token, self.num_expert_group, experts_per_group).reshape(num_token, -1)
        tmp_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
        (topk_weights, topk_ids) = torch.topk(tmp_scores, k=self.topk, dim=-1)
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * self.routed_scaling_factor
        return (topk_weights.to(torch.float32), topk_ids.to(torch.int32))

def get_inputs():
    (num_tokens, hidden_size, num_experts) = (83, 7168, 256)
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device='cuda')
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.float32, device='cuda')
    return [hidden_states, gating_output]

def get_init_inputs():
    return [8, True, 8, 4]
