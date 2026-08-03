from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch_mlu  # noqa: F401 - registers the MLU device with PyTorch
import triton
import triton.language as tl
from triton.runtime import fast_libentry


NUM_EXPERTS = 256
NUM_GROUPS = 8
EXPERTS_PER_GROUP = 32
TOPK_GROUP = 4
TOPK = 8


@triton.jit
def _grouped_topk_group_rank_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    num_tokens,
    logits_row_stride,
    routed_scaling_factor,
    NUM_EXPERTS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EXPERTS_PER_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    TOPK: tl.constexpr,
    STATIC_NUM_TOKENS: tl.constexpr,
    STATIC_ROW_STRIDE: tl.constexpr,
    SCALE_IS_ONE: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    expert_offsets = tl.arange(0, NUM_EXPERTS)
    group_offsets = tl.arange(0, NUM_GROUPS)

    if STATIC_NUM_TOKENS > 0:
        loop_num_tokens = STATIC_NUM_TOKENS
        row_stride = STATIC_ROW_STRIDE
    else:
        loop_num_tokens = num_tokens
        row_stride = logits_row_stride

    for row in range(program_id, loop_num_tokens, num_programs):
        logits = tl.load(
            logits_ptr + row * row_stride + expert_offsets
        ).to(tl.float32)
        grouped_logits = tl.reshape(logits, (NUM_GROUPS, EXPERTS_PER_GROUP))
        group_scores = tl.max(grouped_logits, axis=1)

        # Ranking eight groups in parallel avoids four serial indexed argmax
        # reductions. The group id provides deterministic tie-breaking.
        this_score = group_scores[:, None]
        other_score = group_scores[None, :]
        this_group = group_offsets[:, None]
        other_group = group_offsets[None, :]
        outranks = (other_score > this_score) | (
            (other_score == this_score) & (other_group < this_group)
        )
        group_rank = tl.sum(outranks.to(tl.int32), axis=1)
        selected_groups = group_rank < TOPK_GROUP
        remaining = tl.reshape(
            tl.where(
                selected_groups[:, None], grouped_logits, -float("inf")
            ),
            (NUM_EXPERTS,),
        )

        selected_rank = tl.full((NUM_EXPERTS,), -1, tl.int32)
        for rank in tl.static_range(0, TOPK):
            _, best_id = tl.max(remaining, axis=0, return_indices=True)
            is_selected = expert_offsets == best_id
            selected_rank = tl.where(is_selected, rank, selected_rank)
            remaining = tl.where(is_selected, -float("inf"), remaining)

        selected = selected_rank >= 0
        selected_logits = tl.where(selected, logits, -float("inf"))
        selected_max = tl.max(selected_logits, axis=0)
        numerators = tl.where(
            selected, tl.exp(selected_logits - selected_max), 0.0
        )
        denominator = tl.sum(numerators, axis=0)
        if SCALE_IS_ONE:
            normalized = numerators / denominator
        else:
            normalized = numerators / denominator * routed_scaling_factor

        output_offsets = row * TOPK + selected_rank
        tl.store(weights_ptr + output_offsets, normalized, mask=selected)
        tl.store(ids_ptr + output_offsets, expert_offsets, mask=selected)


_group_rank_fast = fast_libentry()(_grouped_topk_group_rank_kernel)


@triton.jit
def _grouped_topk_group_rank_t83_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    expert_offsets = tl.arange(0, 256)
    group_offsets = tl.arange(0, 8)

    for row in range(program_id, 83, num_programs):
        logits = tl.load(logits_ptr + row * 256 + expert_offsets).to(tl.float32)
        grouped_logits = tl.reshape(logits, (8, 32))
        group_scores = tl.max(grouped_logits, axis=1)

        this_score = group_scores[:, None]
        other_score = group_scores[None, :]
        this_group = group_offsets[:, None]
        other_group = group_offsets[None, :]
        outranks = (other_score > this_score) | (
            (other_score == this_score) & (other_group < this_group)
        )
        group_rank = tl.sum(outranks.to(tl.int32), axis=1)
        selected_groups = group_rank < 4
        remaining = tl.reshape(
            tl.where(selected_groups[:, None], grouped_logits, -float("inf")),
            (256,),
        )

        selected_rank = tl.full((256,), -1, tl.int32)
        for rank in tl.static_range(0, 8):
            _, best_id = tl.max(remaining, axis=0, return_indices=True)
            is_selected = expert_offsets == best_id
            selected_rank = tl.where(is_selected, rank, selected_rank)
            remaining = tl.where(is_selected, -float("inf"), remaining)

        selected = selected_rank >= 0
        selected_logits = tl.where(selected, logits, -float("inf"))
        selected_max = tl.max(selected_logits, axis=0)
        numerators = tl.where(
            selected, tl.exp(selected_logits - selected_max), 0.0
        )
        denominator = tl.sum(numerators, axis=0)
        output_offsets = row * 8 + selected_rank
        tl.store(
            weights_ptr + output_offsets,
            numerators / denominator,
            mask=selected,
        )
        tl.store(ids_ptr + output_offsets, expert_offsets, mask=selected)


_group_rank_t83_fast = fast_libentry()(_grouped_topk_group_rank_t83_kernel)


@functools.lru_cache(maxsize=None)
def _mlu_core_count(device_index: int) -> int:
    return torch.mlu.get_device_properties(device_index).multi_processor_count


def _validate(
    gating_output: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
) -> None:
    if gating_output.device.type != "mlu":
        raise ValueError("gating_output must be an MLU tensor")
    if gating_output.dtype != torch.float32:
        raise ValueError("gating_output must have dtype torch.float32")
    if gating_output.ndim != 2 or gating_output.shape[1] != NUM_EXPERTS:
        raise ValueError(f"gating_output must have shape [T, {NUM_EXPERTS}]")
    if gating_output.stride(1) != 1:
        raise ValueError("gating_output must be contiguous in its last dimension")

    expected_shape = (gating_output.shape[0], TOPK)
    if weights.shape != expected_shape or weights.dtype != torch.float32:
        raise ValueError(f"weights must be float32 with shape {expected_shape}")
    if ids.shape != expected_shape or ids.dtype != torch.int32:
        raise ValueError(f"ids must be int32 with shape {expected_shape}")
    if weights.device != gating_output.device or ids.device != gating_output.device:
        raise ValueError("inputs and outputs must be on the same MLU device")


def grouped_topk_triton_optimized_out(
    gating_output: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    *,
    routed_scaling_factor: float = 1.0,
    specialize_t83: bool = True,
    grid_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run grouped top-k into preallocated outputs.

    The base.py configuration is fixed at 256 experts, 8 groups, top 4
    groups, and top 8 experts. For contiguous [83, 256] logits with scaling
    factor 1.0, compile-time token/stride/scaling specialization is selected
    automatically. Other token counts use the same optimized general kernel.
    """
    _validate(gating_output, weights, ids)
    num_tokens = gating_output.shape[0]
    if num_tokens == 0:
        return weights, ids

    device_index = gating_output.device.index
    core_count = _mlu_core_count(0 if device_index is None else device_index)
    grid = min(num_tokens, core_count) if grid_size is None else min(
        num_tokens, grid_size
    )
    use_t83_specialization = (
        specialize_t83
        and num_tokens == 83
        and gating_output.stride(0) == NUM_EXPERTS
        and routed_scaling_factor == 1.0
    )

    with torch.mlu.device(gating_output.device):
        if use_t83_specialization:
            _group_rank_t83_fast[(grid,)](
                gating_output,
                weights,
                ids,
                num_warps=1,
                num_stages=1,
            )
        else:
            _group_rank_fast[(grid,)](
                gating_output,
                weights,
                ids,
                num_tokens,
                gating_output.stride(0),
                routed_scaling_factor,
                NUM_EXPERTS=NUM_EXPERTS,
                NUM_GROUPS=NUM_GROUPS,
                EXPERTS_PER_GROUP=EXPERTS_PER_GROUP,
                TOPK_GROUP=TOPK_GROUP,
                TOPK=TOPK,
                STATIC_NUM_TOKENS=0,
                STATIC_ROW_STRIDE=0,
                SCALE_IS_ONE=False,
                num_warps=1,
                num_stages=1,
            )
    return weights, ids


def grouped_topk_triton_optimized(
    gating_output: torch.Tensor,
    *,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = gating_output.shape[0]
    weights = torch.empty(
        (num_tokens, TOPK), device=gating_output.device, dtype=torch.float32
    )
    ids = torch.empty(
        (num_tokens, TOPK), device=gating_output.device, dtype=torch.int32
    )
    return grouped_topk_triton_optimized_out(
        gating_output,
        weights,
        ids,
        routed_scaling_factor=routed_scaling_factor,
    )


class OptimizedTritonGroupedTopK(nn.Module):
    def __init__(self, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.routed_scaling_factor = routed_scaling_factor

    def forward(
        self, hidden_states: torch.Tensor, gating_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[0] != gating_output.shape[0]:
            raise ValueError("Number of tokens mismatch")
        return grouped_topk_triton_optimized(
            gating_output,
            routed_scaling_factor=self.routed_scaling_factor,
        )
