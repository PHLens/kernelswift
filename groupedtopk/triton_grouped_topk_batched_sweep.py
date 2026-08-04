from __future__ import annotations

import torch
import torch_mlu  # noqa: F401 - registers the MLU device with PyTorch
import triton
import triton.language as tl
from triton.language.extra.mlu import gather as mlu_gather
from triton.runtime import fast_libentry


@triton.jit
def _grouped_topk_batched_t83_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    BLOCK_ROWS: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
):
    program_id = tl.program_id(0)
    row_slots = tl.arange(0, BLOCK_ROWS)
    group_offsets = tl.arange(0, 8)
    local_offsets = tl.arange(0, 32)
    candidate_offsets = tl.arange(0, 128)
    rank_offsets = tl.arange(0, 8)

    rows = program_id + row_slots * ROW_STRIDE
    row_mask = rows < 83
    expert_offsets = tl.arange(0, 256)
    logits_offsets = rows[:, None] * 256 + expert_offsets[None, :]
    logits = tl.load(
        logits_ptr + logits_offsets,
        mask=row_mask[:, None],
        other=-float("inf"),
    ).to(tl.float32)

    grouped_logits = tl.reshape(logits, (BLOCK_ROWS, 8, 32))
    group_scores = tl.max(grouped_logits, axis=2)
    this_score = group_scores[:, :, None]
    other_score = group_scores[:, None, :]
    this_group = group_offsets[None, :, None]
    other_group = group_offsets[None, None, :]
    outranks = (other_score > this_score) | (
        (other_score == this_score) & (other_group < this_group)
    )
    group_rank = tl.sum(outranks.to(tl.int32), axis=2)
    selected_groups = group_rank < 4

    selected_before = selected_groups[:, None, :] & (
        group_offsets[None, None, :] < group_offsets[None, :, None]
    )
    compact_slots = tl.sum(selected_before.to(tl.int32), axis=2)
    compact_slot_offsets = tl.arange(0, 4)
    selected_group_ids = tl.sum(
        tl.where(
            selected_groups[:, :, None]
            & (compact_slots[:, :, None] == compact_slot_offsets[None, None, :]),
            group_offsets[None, :, None],
            0,
        ),
        axis=1,
    )

    flat_logits = tl.reshape(logits, (BLOCK_ROWS * 256,))
    linear_group_bases = tl.reshape(
        row_slots[:, None] * 256 + selected_group_ids * 32,
        (BLOCK_ROWS * 4,),
    )
    candidates = mlu_gather(flat_logits, linear_group_bases, None, 32)
    candidate_logits = tl.reshape(candidates, (BLOCK_ROWS, 128))
    candidate_ids = tl.reshape(
        selected_group_ids[:, :, None] * 32 + local_offsets[None, None, :],
        (BLOCK_ROWS, 128),
    )

    remaining = candidate_logits
    selected_rank = tl.full((BLOCK_ROWS, 128), -1, tl.int32)
    for rank in tl.static_range(0, 8):
        _, best_position = tl.max(
            remaining,
            axis=1,
            return_indices=True,
        )
        selected = candidate_offsets[None, :] == best_position[:, None]
        selected_rank = tl.where(selected, rank, selected_rank)
        remaining = tl.where(selected, -float("inf"), remaining)

    selected = selected_rank >= 0
    selected_logits = tl.where(selected, candidate_logits, -float("inf"))
    selected_max = tl.max(selected_logits, axis=1)
    numerators = tl.where(
        selected,
        tl.exp(selected_logits - selected_max[:, None]),
        0.0,
    )
    denominator = tl.sum(numerators, axis=1)
    normalized = numerators / denominator[:, None]
    winner_mask = selected[:, :, None] & (
        selected_rank[:, :, None] == rank_offsets[None, None, :]
    )
    dense_weights = tl.sum(
        tl.where(winner_mask, normalized[:, :, None], 0.0),
        axis=1,
    )
    dense_ids = tl.sum(
        tl.where(winner_mask, candidate_ids[:, :, None], 0),
        axis=1,
    )
    output_offsets = rows[:, None] * 8 + rank_offsets[None, :]
    tl.store(weights_ptr + output_offsets, dense_weights, mask=row_mask[:, None])
    tl.store(ids_ptr + output_offsets, dense_ids, mask=row_mask[:, None])


_grouped_topk_batched_w2_fast = fast_libentry()(_grouped_topk_batched_t83_kernel)
_grouped_topk_batched_w4_fast = fast_libentry()(_grouped_topk_batched_t83_kernel)


def _run(
    gating_output: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    *,
    block_rows: int,
    row_stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    runner = _grouped_topk_batched_w2_fast if block_rows == 2 else _grouped_topk_batched_w4_fast
    grid = 48 if block_rows == 2 else 24
    runner[(grid,)](
        gating_output,
        weights,
        ids,
        BLOCK_ROWS=block_rows,
        ROW_STRIDE=row_stride,
        num_warps=1,
        num_stages=1,
    )
    return weights, ids


def grouped_topk_batched2_out(gating_output, weights, ids):
    return _run(gating_output, weights, ids, block_rows=2, row_stride=48)


def grouped_topk_batched4_out(gating_output, weights, ids):
    return _run(gating_output, weights, ids, block_rows=4, row_stride=24)
