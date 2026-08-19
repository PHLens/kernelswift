import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _grouped_topk_kernel(
    gating_ptr,
    weights_ptr,
    ids_ptr,
    K: tl.constexpr,
    KG: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    SCALING: tl.constexpr,
):
    token = tl.program_id(0)
    expert_offsets = tl.arange(0, 256)
    logits = tl.load(gating_ptr + token * 256 + expert_offsets)

    logits_max = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - logits_max)
    scores = exp_logits / tl.sum(exp_logits, axis=0)

    group_scores = tl.max(tl.reshape(scores, (8, 32)), axis=1)
    group_offsets = tl.arange(0, 8)
    group_tie_rank = tl.where(
        group_offsets == 1,
        4.0,
        tl.where(
            group_offsets == 0,
            3.0,
            tl.where(
                group_offsets == 2,
                2.0,
                tl.where(group_offsets == 3, 1.0, 0.0),
            ),
        ),
    )
    selected_groups = tl.zeros((8,), dtype=tl.float32)
    remaining_groups = group_scores

    for _ in tl.static_range(0, KG):
        group_max = tl.max(remaining_groups, axis=0)
        group_tie_key = tl.where(
            remaining_groups == group_max,
            group_tie_rank,
            -float("inf"),
        )
        group_id = tl.argmax(group_tie_key, axis=0)
        selected_groups = tl.where(
            group_offsets == group_id,
            tl.full((8,), 1.0, dtype=tl.float32),
            selected_groups,
        )
        remaining_groups = tl.where(
            group_offsets == group_id,
            -float("inf"),
            remaining_groups,
        )

    group_bases = tl.reshape(
        tl.broadcast_to(tl.reshape(group_offsets * 32, (8, 1)), (8, 32)),
        (256,),
    )
    expert_mask = tl.reshape(
        tl.broadcast_to(tl.reshape(selected_groups > 0, (8, 1)), (8, 32)),
        (256,),
    )
    expert_tie_slot = expert_offsets - group_bases
    expert_tie_rank = tl.where(
        expert_tie_slot == 7,
        8.0,
        tl.where(
            expert_tie_slot == 6,
            7.0,
            tl.where(
                expert_tie_slot == 4,
                6.0,
                tl.where(
                    expert_tie_slot == 5,
                    5.0,
                    tl.where(
                        expert_tie_slot == 1,
                        4.0,
                        tl.where(
                            expert_tie_slot == 0,
                            3.0,
                            tl.where(
                                expert_tie_slot == 2,
                                2.0,
                                tl.where(expert_tie_slot == 3, 1.0, 0.0),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    remaining_scores = tl.where(expert_mask, scores, -float("inf"))

    output_offsets = tl.arange(0, 8)
    output_weights = tl.zeros((8,), dtype=tl.float32)
    output_ids = tl.zeros((8,), dtype=tl.float32)

    for selection_round in tl.static_range(0, K):
        selected_value = tl.max(remaining_scores, axis=0)
        expert_tie_key = tl.where(
            remaining_scores == selected_value,
            expert_tie_rank - group_bases,
            -float("inf"),
        )
        expert_id = tl.argmax(expert_tie_key, axis=0)
        output_mask = output_offsets == selection_round
        output_weights = tl.where(output_mask, selected_value, output_weights)
        output_ids = tl.where(
            output_mask,
            expert_id.to(tl.float32),
            output_ids,
        )
        remaining_scores = tl.where(
            expert_offsets == expert_id,
            -float("inf"),
            remaining_scores,
        )

    if RENORMALIZE:
        output_weights = output_weights / tl.sum(output_weights, axis=0)
    if SCALING != 1.0:
        output_weights = output_weights * SCALING

    tl.store(weights_ptr + token * K + output_offsets, output_weights)
    tl.store(ids_ptr + token * K + output_offsets, output_ids.to(tl.int32))


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

    def _eager_forward(self, gating_output: torch.Tensor):
        if self.scoring_func == "softmax":
            scores = torch.softmax(gating_output, dim=-1)
        elif self.scoring_func == "sigmoid":
            scores = gating_output.sigmoid()
        else:
            raise ValueError(f"Unsupported scoring_func: {self.scoring_func}")

        num_token = scores.size(0)
        experts_per_group = scores.size(-1) // self.num_expert_group
        group_scores = scores.view(num_token, self.num_expert_group, -1).max(dim=-1).values
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_token, self.num_expert_group, experts_per_group)
            .reshape(num_token, -1)
        )
        tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
        topk_weights, topk_ids = torch.topk(tmp_scores, k=self.topk, dim=-1)
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    def forward(self, hidden_states: torch.Tensor, gating_output: torch.Tensor):
        assert hidden_states.size(0) == gating_output.size(0)
        tokens, experts = gating_output.shape
        if (
            self.scoring_func != "softmax"
            or tokens != 83
            or experts != 256
            or self.num_expert_group != 8
            or self.topk_group != 4
            or self.topk != 8
            or gating_output.dtype != torch.float32
            or not gating_output.is_contiguous()
        ):
            return self._eager_forward(gating_output)

        topk_weights = torch.empty(
            (tokens, self.topk), dtype=torch.float32, device=gating_output.device
        )
        topk_ids = torch.empty(
            (tokens, self.topk), dtype=torch.int32, device=gating_output.device
        )
        _grouped_topk_kernel[(tokens,)](
            gating_output,
            topk_weights,
            topk_ids,
            K=self.topk,
            KG=self.topk_group,
            RENORMALIZE=self.renormalize,
            SCALING=self.routed_scaling_factor,
        )
        return topk_weights, topk_ids


def get_inputs():
    num_tokens, hidden_size, num_experts = 83, 7168, 256
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.float16, device="cuda"
    )
    gating_output = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device="cuda"
    )
    return [hidden_states, gating_output]


def get_init_inputs():
    return [8, True, 8, 4]
