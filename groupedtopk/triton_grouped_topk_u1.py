from __future__ import annotations

import torch
import torch_mlu  # noqa: F401 - registers the MLU device with PyTorch
import triton
import triton.language as tl
from triton.runtime import fast_libentry


NUM_TOKENS = 83
NUM_EXPERTS = 256
NUM_CLUSTERS = 12
ROWS_PER_CLUSTER = 8


@triton.jit
def _row_max_single_t83_kernel(logits_ptr, output_ptr):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    expert_offsets = tl.arange(0, 256)

    for row in range(program_id, 83, num_programs):
        logits = tl.load(logits_ptr + row * 256 + expert_offsets)
        tl.store(output_ptr + row, tl.max(logits, axis=0))


@triton.jit
def _row_max_batched_u1_t83_kernel(logits_ptr, output_ptr):
    cluster_id = tl.program_id(0)
    row_slots = tl.arange(0, 8)
    expert_offsets = tl.arange(0, 256)
    rows = cluster_id + row_slots * 12
    row_mask = rows < 83
    offsets = rows[:, None] * 256 + expert_offsets[None, :]
    logits = tl.load(logits_ptr + offsets, mask=row_mask[:, None], other=-float("inf"))
    row_max = tl.max(logits, axis=1)
    tl.store(output_ptr + rows, row_max, mask=row_mask)


_row_max_single_t83_fast = fast_libentry()(_row_max_single_t83_kernel)
_row_max_batched_u1_w1_t83_fast = fast_libentry()(_row_max_batched_u1_t83_kernel)
_row_max_batched_u1_w4_t83_fast = fast_libentry()(_row_max_batched_u1_t83_kernel)
_row_max_batched_u1_w4_shared_t83_fast = fast_libentry()(
    _row_max_batched_u1_t83_kernel
)


def row_max_single_out(logits: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    _row_max_single_t83_fast[(48,)](
        logits,
        output,
        num_warps=1,
        num_stages=1,
    )
    return output


def row_max_batched_u1_out(
    logits: torch.Tensor,
    output: torch.Tensor,
    *,
    num_warps: int = 4,
    force_use_shared_memory: bool = False,
) -> torch.Tensor:
    if num_warps == 1:
        runner = _row_max_batched_u1_w1_t83_fast
    elif force_use_shared_memory:
        runner = _row_max_batched_u1_w4_shared_t83_fast
    else:
        runner = _row_max_batched_u1_w4_t83_fast
    runner[(NUM_CLUSTERS,)](
        logits,
        output,
        num_warps=num_warps,
        num_stages=1,
        force_use_shared_memory=force_use_shared_memory,
    )
    return output
