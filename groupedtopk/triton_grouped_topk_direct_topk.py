from __future__ import annotations

import torch
import torch_mlu  # noqa: F401
import triton
import triton.language as tl
from triton.language.extra.mlu import gather as mlu_gather
from triton.runtime import fast_libentry

from triton_grouped_topk_hierarchical import _mlu_core_count


@triton.jit
def _grouped_topk_direct_dense_t83_kernel(logits_ptr, weights_ptr, ids_ptr):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    group_offsets = tl.arange(0, 8)
    local_offsets = tl.arange(0, 32)
    compact_block_offsets = tl.arange(0, 2)
    candidate_offsets = tl.arange(0, 128)
    top_offsets = tl.arange(0, 8)

    for row in range(program_id, 83, num_programs):
        logits = tl.load(logits_ptr + row * 256 + tl.arange(0, 256)).to(tl.float32)
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

        compacted_group_ids, _ = tl.masked_select(group_offsets, selected_groups)
        compacted_group_blocks = tl.reshape(compacted_group_ids, (2, 4))
        selected_group_ids = tl.sum(
            tl.where(compact_block_offsets[:, None] == 0, compacted_group_blocks, 0),
            axis=0,
        )
        group_base_offsets = selected_group_ids * 32
        candidates_2d = mlu_gather(logits, group_base_offsets, None, 32)
        candidate_ids_2d = selected_group_ids[:, None] * 32 + local_offsets[None, :]
        remaining = tl.reshape(candidates_2d, (128,))
        candidate_ids = tl.reshape(candidate_ids_2d, (128,))

        # Keep the top-k result dense throughout selection. The previous
        # kernel stored a 128-lane selected-rank vector and scattered it at
        # the end; this path writes the rank slot on each fixed iteration.
        top_values = tl.full((8,), -float("inf"), tl.float32)
        top_ids = tl.zeros((8,), tl.int32)
        for rank in tl.static_range(0, 8):
            best_value, best_position = tl.max(
                remaining, axis=0, return_indices=True
            )
            best_id = tl.sum(
                tl.where(candidate_offsets == best_position, candidate_ids, 0),
                axis=0,
            )
            top_values = tl.where(top_offsets == rank, best_value, top_values)
            top_ids = tl.where(top_offsets == rank, best_id, top_ids)
            remaining = tl.where(
                candidate_offsets == best_position, -float("inf"), remaining
            )

        selected_max = tl.max(top_values, axis=0)
        numerators = tl.exp(top_values - selected_max)
        denominator = tl.sum(numerators, axis=0)
        output_offsets = row * 8 + top_offsets
        tl.store(weights_ptr + output_offsets, numerators / denominator)
        tl.store(ids_ptr + output_offsets, top_ids)


_direct_dense_fast = fast_libentry()(_grouped_topk_direct_dense_t83_kernel)


def grouped_topk_triton_direct_dense_out(gating_output, weights, ids, *, grid_size=None):
    if gating_output.device.type != "mlu" or gating_output.dtype != torch.float32:
        raise ValueError("gating_output must be a contiguous MLU float32 tensor")
    if gating_output.shape != (83, 256) or gating_output.stride() != (256, 1):
        raise ValueError("gating_output must have shape [83, 256] and be contiguous")
    if weights.shape != (83, 8) or weights.dtype != torch.float32:
        raise ValueError("weights must be float32 with shape [83, 8]")
    if ids.shape != (83, 8) or ids.dtype != torch.int32:
        raise ValueError("ids must be int32 with shape [83, 8]")
    grid = _mlu_core_count(0 if gating_output.device.index is None else gating_output.device.index)
    if grid_size is not None:
        grid = min(grid_size, grid)
    with torch.mlu.device(gating_output.device):
        _direct_dense_fast[(grid,)](
            gating_output, weights, ids, num_warps=1, num_stages=1
        )
    return weights, ids
