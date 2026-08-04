from __future__ import annotations

import functools

import torch
import torch_mlu  # noqa: F401
import triton
import triton.language as tl
from triton.language.extra.mlu import gather as mlu_gather
from triton.runtime import fast_libentry

from triton_grouped_topk_hierarchical import (
    NUM_EXPERTS,
    NUM_GROUPS,
    EXPERTS_PER_GROUP,
    TOPK_GROUP,
    TOPK,
    NUM_TOKENS,
    _mlu_core_count,
)


@triton.jit
def _grouped_topk_compact128_prefix_t83_kernel(logits_ptr, weights_ptr, ids_ptr):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    group_offsets = tl.arange(0, 8)
    compact_offsets = tl.arange(0, 4)
    local_offsets = tl.arange(0, 32)
    candidate_offsets = tl.arange(0, 128)

    for row in range(program_id, 83, num_programs):
        logits = tl.load(
            logits_ptr + row * 256 + tl.arange(0, 256)
        ).to(tl.float32)
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

        # Form the four compact slots with fixed-width compares. This keeps
        # the original ascending group-ID order without masked_select.
        prior_selected = tl.sum(
            (
                selected_groups[None, :]
                & (other_group < this_group)
            ).to(tl.int32),
            axis=1,
        )
        selected_group_ids = tl.sum(
            tl.where(
                selected_groups[:, None]
                & (prior_selected[:, None] == compact_offsets[None, :]),
                group_offsets[:, None],
                0,
            ),
            axis=0,
        )

        group_base_offsets = selected_group_ids * 32
        candidates_2d = mlu_gather(logits, group_base_offsets, None, 32)
        candidate_ids_2d = (
            selected_group_ids[:, None] * 32
            + local_offsets[None, :]
        )
        remaining = tl.reshape(candidates_2d, (128,))
        candidate_ids = tl.reshape(
            candidate_ids_2d, (128,)
        )

        selected_rank = tl.full(
            (128,), -1, tl.int32
        )
        for rank in tl.static_range(0, 8):
            _, best_position = tl.max(remaining, axis=0, return_indices=True)
            selected = candidate_offsets == best_position
            selected_rank = tl.where(selected, rank, selected_rank)
            remaining = tl.where(selected, -float("inf"), remaining)

        selected = selected_rank >= 0
        selected_logits = tl.where(
            selected, tl.reshape(candidates_2d, (128,)), -float("inf")
        )
        selected_max = tl.max(selected_logits, axis=0)
        numerators = tl.where(
            selected, tl.exp(selected_logits - selected_max), 0.0
        )
        denominator = tl.sum(numerators, axis=0)
        output_offsets = row * 8 + selected_rank
        tl.store(weights_ptr + output_offsets, numerators / denominator, mask=selected)
        tl.store(ids_ptr + output_offsets, candidate_ids, mask=selected)


_prefix_t83_fast = fast_libentry()(_grouped_topk_compact128_prefix_t83_kernel)


# This runner uses the unchanged Entry 018 kernel but requests the MLU
# compiler's shared-memory promotion. It is kept separate because fast_libentry
# caches the first compiled metadata for a kernel object.
from triton_grouped_topk_hierarchical import _grouped_topk_compact128_t83_kernel

_compact128_shared_fast = fast_libentry()(_grouped_topk_compact128_t83_kernel)


def _validate(gating_output: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> int:
    if gating_output.device.type != "mlu" or gating_output.dtype != torch.float32:
        raise ValueError("gating_output must be a contiguous MLU float32 tensor")
    if gating_output.shape != (NUM_TOKENS, NUM_EXPERTS) or gating_output.stride() != (NUM_EXPERTS, 1):
        raise ValueError("gating_output must have shape [83, 256] and be contiguous")
    if weights.shape != (NUM_TOKENS, TOPK) or weights.dtype != torch.float32:
        raise ValueError("weights must be float32 with shape [83, 8]")
    if ids.shape != (NUM_TOKENS, TOPK) or ids.dtype != torch.int32:
        raise ValueError("ids must be int32 with shape [83, 8]")
    if weights.device != gating_output.device or ids.device != gating_output.device:
        raise ValueError("inputs and outputs must be on the same MLU device")
    device_index = gating_output.device.index
    return _mlu_core_count(0 if device_index is None else device_index)


def grouped_topk_triton_prefix_out(gating_output, weights, ids, *, grid_size=None):
    grid = _validate(gating_output, weights, ids)
    if grid_size is not None:
        grid = min(grid_size, grid)
    with torch.mlu.device(gating_output.device):
        _prefix_t83_fast[(grid,)](
            gating_output, weights, ids, num_warps=1, num_stages=1
        )
    return weights, ids


def grouped_topk_triton_compact128_shared_out(gating_output, weights, ids, *, grid_size=None):
    grid = _validate(gating_output, weights, ids)
    if grid_size is not None:
        grid = min(grid_size, grid)
    with torch.mlu.device(gating_output.device):
        _compact128_shared_fast[(grid,)](
            gating_output,
            weights,
            ids,
            num_warps=1,
            num_stages=1,
            force_use_shared_memory=True,
        )
    return weights, ids
