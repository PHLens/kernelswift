from __future__ import annotations

import torch
import torch_mlu  # noqa: F401 - registers the MLU device with PyTorch
import triton
import triton.language as tl
from triton.language.extra.mlu import gather as mlu_gather
from triton.runtime import fast_libentry


NUM_TOKENS = 83
NUM_EXPERTS = 256
NUM_CLUSTERS = 12
ROWS_PER_CLUSTER = 8
TOPK = 8


@triton.jit
def _grouped_topk_batched_u1_t83_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
):
    cluster_id = tl.program_id(0)
    row_slots = tl.arange(0, 8)
    group_offsets = tl.arange(0, 8)
    local_offsets = tl.arange(0, 32)
    candidate_offsets = tl.arange(0, 128)
    rank_offsets = tl.arange(0, 8)

    rows = cluster_id + row_slots * 12
    row_mask = rows < 83
    expert_offsets = tl.arange(0, 256)
    logits_offsets = rows[:, None] * 256 + expert_offsets[None, :]
    logits = tl.load(
        logits_ptr + logits_offsets,
        mask=row_mask[:, None],
        other=-float("inf"),
    ).to(tl.float32)

    grouped_logits = tl.reshape(logits, (8, 8, 32))
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

    # Encode each selected group into one of four compact slots per row.
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

    group_base_offsets = selected_group_ids * 32
    flat_logits = tl.reshape(logits, (2048,))
    linear_group_bases = tl.reshape(
        row_slots[:, None] * 256 + group_base_offsets,
        (32,),
    )
    candidates_2d = mlu_gather(flat_logits, linear_group_bases, None, 32)
    candidate_logits = tl.reshape(candidates_2d, (8, 128))
    candidate_ids = tl.reshape(
        selected_group_ids[:, :, None] * 32 + local_offsets[None, None, :],
        (8, 128),
    )

    remaining = candidate_logits
    selected_rank = tl.full((8, 128), -1, tl.int32)
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
    tl.store(
        weights_ptr + output_offsets,
        dense_weights,
        mask=row_mask[:, None],
    )
    tl.store(
        ids_ptr + output_offsets,
        dense_ids,
        mask=row_mask[:, None],
    )


_grouped_topk_batched_u1_w1_t83_fast = fast_libentry()(
    _grouped_topk_batched_u1_t83_kernel
)
_grouped_topk_batched_u1_w4_t83_fast = fast_libentry()(
    _grouped_topk_batched_u1_t83_kernel
)


def grouped_topk_triton_batched_u1_out(
    gating_output: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    *,
    num_warps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gating_output.device.type != "mlu":
        raise ValueError("gating_output must be an MLU tensor")
    if gating_output.dtype != torch.float32 or gating_output.shape != (83, 256):
        raise ValueError("gating_output must be contiguous float32 [83, 256]")
    if gating_output.stride() != (256, 1):
        raise ValueError("gating_output must be contiguous")
    if weights.shape != (83, 8) or weights.dtype != torch.float32:
        raise ValueError("weights must be float32 [83, 8]")
    if ids.shape != (83, 8) or ids.dtype != torch.int32:
        raise ValueError("ids must be int32 [83, 8]")
    if weights.device != gating_output.device or ids.device != gating_output.device:
        raise ValueError("inputs and outputs must be on the same MLU device")

    with torch.mlu.device(gating_output.device):
        runner = (
            _grouped_topk_batched_u1_w1_t83_fast
            if num_warps == 1
            else _grouped_topk_batched_u1_w4_t83_fast
        )
        runner[(12,)](
            gating_output,
            weights,
            ids,
            num_warps=num_warps,
            num_stages=1,
        )
    return weights, ids
