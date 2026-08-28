"""Grouped Top-K expert routing - bi150-round2 round 002 candidate.

Decision-002 change family ``compile-graph-default`` (sketch
rounds/sketch_002.json, normative): the ACCEPTED round-001 staged Triton
pipeline is wrapped behind ONE shared torch.compile(mode='default',
dynamic=False) callable consumed by BOTH forward and run_out on the exact
target regime (contiguous fp32 [83,256] gating on the current device plus the
fixed constructor config topk=8 / renormalize=True / num_expert_group=8 /
topk_group=4 / scoring_func='softmax' / routed_scaling_factor=1.0). All three
@triton.jit stage kernels and BOTH retained torch.topk call sites are
byte-for-byte unchanged versus triton_grouped_topk_r2_001.py
@4ae64cad913267f2198fec735e08f1b9490cafa1139d3a48ee11400aacb80de3.

Execution strategy, not dataflow: dynamo traces the fixed-shape pipeline once;
inductor partitions ATen work while torch.topk routes to the same ATen vendor
selection kernels; partial-graph outcomes (graph breaks around the opaque
Triton launches) are acceptable and every produced value must stay
mathematically and orderingly identical.

Fallback chain is NORMATIVE because torch.compile is Constrained on the frozen
triton_cuda profile snapshot:
- a strict per-call regime guard routes non-target regimes to the unmodified
  round-001 staged execution (and its framework-eager fallback), exactly as
  accepted in round 001; the compiled wrapper is never constructed for them;
- ANY exception while constructing or invoking the compiled callable
  permanently binds this instance to the unmodified staged execution (the flag
  transitions at most once, compiled -> eager, and never back);
- compilation configuration is restricted to mode='default' and dynamic=False:
  no precision settings, no backend selection overrides, and no cache-size
  environment handling exist anywhere in this file.

Host Plan lifecycle: the instance owns the compiled-callable handle, the
constructor guard constants, and the immutable-at-runtime fallback flag; those
are code objects/constants only (framework build caches contain no tensor
data). Every tensor, including run_out-provided output buffers, remains owned
by the calling invocation; temporaries are allocated fresh inside each executed
implementation exactly as round 001 does; the caller-selected device and the
current stream are preserved (dynamo guards evaluate host-side; compiled
partitions and eager Triton launches execute on the current stream).
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
        self._compiled_staged = None
        self._compile_failed = False

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

    def _compile_regime_applies(self, gating_output: torch.Tensor) -> bool:
        """Strict target-regime guard for the single compiled callable: fixed
        shape [83,256] contiguous fp32 cuda gating plus the fixed constructor
        configuration. Anything else uses the unmodified staged execution."""
        return (
            gating_output.size(0) == 83
            and self.renormalize
            and float(self.routed_scaling_factor) == 1.0
        )

    def _get_compiled_staged(self):
        """One-time lazy construction of the shared compiled callable for the
        target regime; any construction failure permanently binds the instance
        to the unmodified staged execution."""
        if self._compile_failed:
            return None
        if self._compiled_staged is None:
            try:
                self._compiled_staged = torch.compile(self._triton_forward, mode='default', dynamic=False)
            except Exception:
                self._compile_failed = True
                self._compiled_staged = None
        return self._compiled_staged

    def _invoke_compiled_or_staged(
        self,
        gating_output: torch.Tensor,
        out_weights: torch.Tensor | None = None,
        out_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared invocation route for forward AND run_out on the target
        regime: one compiled callable invoked positionally with the optional
        preallocated-output arguments so run_out keeps zero-copy in-place
        writes. Any invocation failure permanently binds the instance to the
        unmodified staged execution with identical argument mapping."""
        compiled = self._get_compiled_staged()
        if compiled is not None:
            try:
                return compiled(gating_output, out_weights, out_ids)
            except Exception:
                self._compile_failed = True
                self._compiled_staged = None
        return self._triton_forward(gating_output, out_weights, out_ids)

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
            if self._compile_regime_applies(gating_output):
                return self._invoke_compiled_or_staged(gating_output)
            # Non-target regime: unmodified staged execution (round 001).
            return self._triton_forward(gating_output)
        return self._eager_forward(gating_output)

    def run_out(self, gating_output: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> None:
        """Preallocated-output execution surface required by kernel-mode
        profiling (auto_bench.make_profile_call: called as
        run_out(gating_output, *reference_outputs, **model.run_kwargs)); the
        return value is ignored and results are completed in-place into the
        provided buffers before returning. Byte-identical to forward() for
        identical inputs on every routing branch."""
        if self._fast_path_applies(gating_output):
            if self._compile_regime_applies(gating_output):
                # Same shared compiled callable; the staged pipeline writes the
                # stage-C results directly into the provided buffers.
                self._invoke_compiled_or_staged(gating_output, topk_weights, topk_ids)
                return
            self._triton_forward(gating_output, topk_weights, topk_ids)
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
