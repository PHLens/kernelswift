"""Grouped Top-K expert routing - bi150-round2 round 001 candidate.

Decision-001 change family ``preprocess-fusion-triton-stages`` (sketch
rounds/sketch_001.json, normative): the eager framework preprocessing chain is
replaced by three direct per-token Triton stages around the two RETAINED exact
library ``torch.topk`` selection calls:

- stage A ``_softmax_group_scores_kernel``: fused fp32 softmax over the expert
  row plus per-expert-group max reduce, producing ``scores_out[T,E]`` and
  ``group_scores_out[T,G]``;
- stage B ``_group_mask_kernel``: arithmetic group-membership masking via
  ``tl.where`` (replaces the zeros/scatter_/expand/bitwise_not/masked_fill
  framework chain), selecting score lanes whose ``floor(expert/experts_per_group)``
  belongs to the library-selected group ids;
- stage C ``_renorm_scale_narrow_kernel``: renormalize divide, routed-scaling
  multiply, and the int64-to-int32 id narrowing folded into one small kernel,
  producing ``out_weights[T,K]`` fp32 and ``out_ids[T,K]`` int32.

Both ``torch.topk`` call sites keep identical argument values, shapes, dtypes,
ordering, and tie behavior versus base.py semantics. ``tl.argmax``, num_warps,
and num_stages are intentionally absent: launch hints remain Unknown on the
frozen triton_cuda profile snapshot and stay unset (direct default launches).
The int64-to-int32 narrowing is performed kernel-side per the Decision-scoped
bounded probe ``log/probes/probe_cast_narrow_int64_int32.py``
(cast.narrow.int64-to-int32-kernel-side: observed, numerically checked).

Lifecycle follows the Host Plan: the instance owns compiled kernel handles and
immutable configuration constants only; every tensor, including the output
buffers supplied to ``run_out``, remains owned by the calling invocation; fresh
temporaries are allocated on ``gating_output.device`` at every call with no
cross-instance or cross-call caching. Caller device selection and the current
stream are preserved (direct-launch syntax on the current stream, no device
context mutation, no synchronization beyond base.py behavior).
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

_BLOCK_EXPERTS = 256
_NUM_GROUPS = 8
_KG = 4
_TOPK = 8


@triton.jit
def _softmax_group_scores_kernel(
    gating_ptr,
    scores_ptr,
    group_scores_ptr,
    EXPERTS: tl.constexpr,
    GROUPS: tl.constexpr,
    EXPERTS_PER_GROUP: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.arange(0, EXPERTS)
    logits = tl.load(gating_ptr + token * EXPERTS + offsets)
    logits_max = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - logits_max)
    probs = exp_logits / tl.sum(exp_logits, axis=0)
    group_tile = tl.max(tl.reshape(probs, (GROUPS, EXPERTS_PER_GROUP)), axis=1)
    tl.store(scores_ptr + token * EXPERTS + offsets, probs)
    tl.store(group_scores_ptr + token * GROUPS + tl.arange(0, GROUPS), group_tile)


@triton.jit
def _group_mask_kernel(
    scores_ptr,
    sel_groups_ptr,
    masked_scores_ptr,
    EXPERTS: tl.constexpr,
    GROUPS: tl.constexpr,
    EXPERTS_PER_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.arange(0, EXPERTS)
    scores_row = tl.load(scores_ptr + token * EXPERTS + offsets)
    expert_groups = offsets // EXPERTS_PER_GROUP
    sel_first = tl.load(sel_groups_ptr + token * TOPK_GROUP + 0)
    membership = expert_groups == sel_first
    for group_slot in tl.static_range(1, TOPK_GROUP):
        sel_slot = tl.load(sel_groups_ptr + token * TOPK_GROUP + group_slot)
        membership = membership | (expert_groups == sel_slot)
    masked_tile = tl.where(membership, scores_row, -float("inf"))
    tl.store(masked_scores_ptr + token * EXPERTS + offsets, masked_tile)


@triton.jit
def _renorm_scale_narrow_kernel(
    topk_vals_ptr,
    topk_ids_ptr,
    out_weights_ptr,
    out_ids_ptr,
    routed_scaling_factor,
    TOPK: tl.constexpr,
    RENORMALIZE: tl.constexpr,
):
    token = tl.program_id(0)
    k_offsets = tl.arange(0, TOPK)
    vals_tile = tl.load(topk_vals_ptr + token * TOPK + k_offsets)
    if RENORMALIZE:
        vals_sum = tl.sum(vals_tile, axis=0)
        wnorm_tile = vals_tile / vals_sum
    else:
        wnorm_tile = vals_tile
    wnorm_tile = wnorm_tile * routed_scaling_factor
    ids_tile = tl.load(topk_ids_ptr + token * TOPK + k_offsets)
    ids32_tile = ids_tile.to(tl.int32)
    tl.store(out_weights_ptr + token * TOPK + k_offsets, wnorm_tile)
    tl.store(out_ids_ptr + token * TOPK + k_offsets, ids32_tile)


class ModelNew(nn.Module):
    """Grouped Top-K expert routing: score experts, select the highest-scoring
    expert groups, then select experts only from those groups.

    Shared cross-backend reference (device-neutral): tensors are created with
    the 'cuda' device string and auto_bench.py rewrites/moves them to the
    active accelerator (mlu / gcu / cuda).
    """

    def __init__(self, topk: int, renormalize: bool, num_expert_group: int, topk_group: int, scoring_func: str='softmax', routed_scaling_factor: float=1.0):
        super().__init__()
        self.topk = topk
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor

    def _eager_forward(self, gating_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Byte-equivalent base.py decomposition for regimes outside the
        proven Triton fast path (semantic preservation, not a substitution)."""
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

    def _fast_path_applies(self, gating_output: torch.Tensor) -> bool:
        return (
            self.scoring_func == 'softmax'
            and gating_output.dtype == torch.float32
            and gating_output.is_cuda
            and gating_output.dim() == 2
            and gating_output.is_contiguous()
            and gating_output.size(-1) == _BLOCK_EXPERTS
            and gating_output.size(0) > 0
            and self.num_expert_group == _NUM_GROUPS
            and self.topk_group == _KG
            and self.topk == _TOPK
            and _BLOCK_EXPERTS % self.num_expert_group == 0
        )

    def _triton_forward(
        self,
        gating_output: torch.Tensor,
        out_weights: torch.Tensor | None = None,
        out_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_token = gating_output.size(0)
        # Per-call temporaries; owned by this invocation only (Host Plan).
        scores_out = torch.empty_like(gating_output)
        group_scores_out = torch.empty((num_token, _NUM_GROUPS), dtype=torch.float32, device=gating_output.device)
        masked_scores = torch.empty_like(gating_output)

        # Stage A: fused softmax + per-group max.
        _softmax_group_scores_kernel[(num_token,)](
            gating_output,
            scores_out,
            group_scores_out,
            EXPERTS=_BLOCK_EXPERTS,
            GROUPS=_NUM_GROUPS,
            EXPERTS_PER_GROUP=_BLOCK_EXPERTS // _NUM_GROUPS,
        )
        # Retained exact library selection #1: top-k expert groups.
        sel_groups = torch.topk(group_scores_out, k=self.topk_group, dim=-1)[1]
        # Stage B: arithmetic lane-membership masking replacing the scatter/
        # expand/not/masked_fill chain.
        _group_mask_kernel[(num_token,)](
            scores_out,
            sel_groups,
            masked_scores,
            EXPERTS=_BLOCK_EXPERTS,
            GROUPS=_NUM_GROUPS,
            EXPERTS_PER_GROUP=_BLOCK_EXPERTS // _NUM_GROUPS,
            TOPK_GROUP=self.topk_group,
        )
        # Retained exact library selection #2: top-k experts inside chosen groups.
        (topk_vals, topk_ids) = torch.topk(masked_scores, k=self.topk, dim=-1)

        if out_weights is None:
            out_weights = torch.empty((num_token, self.topk), dtype=torch.float32, device=gating_output.device)
        if out_ids is None:
            out_ids = torch.empty((num_token, self.topk), dtype=torch.int32, device=gating_output.device)
        # Stage C: renormalize divide + routed-scaling multiply + int64->int32
        # id narrowing folded into one small kernel (kernel-side narrowing
        # evidenced by the Decision-scoped bounded probe).
        _renorm_scale_narrow_kernel[(num_token,)](
            topk_vals,
            topk_ids,
            out_weights,
            out_ids,
            float(self.routed_scaling_factor),
            TOPK=self.topk,
            RENORMALIZE=bool(self.renormalize),
        )
        return (out_weights, out_ids)

    def forward(self, hidden_states: torch.Tensor, gating_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states.size(0) == gating_output.size(0)
        if self._fast_path_applies(gating_output):
            return self._triton_forward(gating_output)
        return self._eager_forward(gating_output)

    def run_out(self, gating_output: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> None:
        """Preallocated-output execution surface required by kernel-mode
        profiling (auto_bench.make_profile_call: called as
        run_out(gating_output, *reference_outputs, **model.run_kwargs)); the
        return value is ignored and results are completed in-place into the
        provided buffers before returning. Produces byte-identical results to
        forward() for identical inputs."""
        if self._fast_path_applies(gating_output):
            self._triton_forward(gating_output, out_weights=topk_weights, out_ids=topk_ids)
            return
        (weights, ids) = self._eager_forward(gating_output)
        topk_weights.copy_(weights)
        topk_ids.copy_(ids)


def get_inputs():
    (num_tokens, hidden_size, num_experts) = (83, 7168, 256)
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device='cuda')
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.float32, device='cuda')
    return [hidden_states, gating_output]

def get_init_inputs():
    return [8, True, 8, 4]
