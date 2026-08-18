# Coder Result 009

Result: candidate-ready

## Identity

- Round: `009`
- Decision: `rounds/decision_009.md`
- Decision SHA-256: `066045e737fa1aedcc283c4058d2eceb28b8630013c7b93342abdb516af908b8`
- Canonical source: `triton_grouped_topk_008.py`
- Canonical source SHA-256: `d1fb6b03d3be92cdd6423f1f44f33ea81d13f0e4df18227fe2d5f7dceb582535`
- Candidate: `triton_grouped_topk_009.py`
- Candidate SHA-256: `9b58f861ef6c3de86577dfe819327895311298cc4edf4b3f514f7fe9f4bff194`
- Language/backend/profile: `triton` / `cuda` / `triton_cuda`
- Runtime fingerprint: `project.md#runtime-fingerprint`

## Implementation

The candidate applies exactly one host-only change over the accepted
`triton_grouped_topk_008.py`: the constructor-owned compiled callable is created
with `torch.compile(self._target_forward, mode="reduce-overhead")` instead of the
default mode. This is the sole difference (confirmed by `diff`), localized to
`ModelNew.__init__`.

Everything else is byte-for-byte identical to the accepted canonical: the two
direct Triton stages (`_softmax_group_scores_kernel`, `_group_mask_kernel`), the
exact library `torch.topk(group_scores, 4)` and `torch.topk(masked_scores, 8)`
boundaries, the eager `_eager_forward` fallback, the target-shape dispatch
guard, `_compile_failed` lifecycle handling, per-forward temporary buffer
allocation (`scores`/`group_scores`/`masked_scores`), and the public constructor
and `forward` contract are all unchanged.

## Gate Evidence

| Gate | Observation | Verdict |
|---|---|---|
| Decision validation | `validate_decision.py decision_009.md --expected-profile triton_cuda` returned `valid: true`, exit code 0 | pass |
| AST | `python3 -m py_compile kernels/track1-triton/groupedtopk/bi150/triton_grouped_topk_009.py` succeeded | pass |
| Harness compile smoke | `auto_bench.py --v0_file base.py --v1_file triton_grouped_topk_009.py --warmup 5 --repeat 10 --full-traceback`: `PASS accuracy`; v0 `0.487708 ms`, v1 `0.287010 ms`, speedup `1.699x` | pass |
| All-equal ties | IDs `[7,6,4,5,1,0,2,3]` both sides, `torch.equal` true, weights allclose | pass |
| Two-expert tie | base `[1,0,2,3,4,5,7,6]` == candidate `[1,0,2,3,4,5,7,6]`, `torch.equal` true, weights allclose | pass |
| Structured group tie | IDs `[32,0,64,96,4,3,1,2]` both sides, `torch.equal` true, weights allclose | pass |

Tie checks loaded both `base.py` (`Model`) and the candidate (`ModelNew`) through
the real harness AST loader (`auto_bench.load_ks_module`) and fed constructed
`[83,256]` fp32 contiguous `gating_output` tensors with an `[83,7168]` fp16
`hidden_states` for the batch assertion. The candidate compiled target path was
warmed up before comparison. `torch.equal` on `topk_ids` and
`torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` on `topk_weights` held for
all three cases. The all-equal and structured-group-tie IDs reproduce the Round
008 recorded expectations exactly; the two-expert-tie case's tail ordering is
construction-specific but base and candidate agree element-for-element, which is
the required correctness criterion.

## Conformance

- Host-only change; no new selector, kernel, dataflow, or lifecycle is introduced.
- Uses the constrained `torch.compile` evidence from
  `scripts/bi150_torch_compile_reduce_overhead_probe.py`; the candidate itself
  validates the `reduce-overhead` compiled target-shape graph path on BI150.
- Retains caller device/current-stream behavior and accepted per-forward buffer
  ownership; no output or temporary cache, no stream/context mutation.
- Fallback to the accepted eager two-stage forward remains for non-target inputs,
  compile failure, and unsupported graph/lifecycle cases.
- Does not use `tl.dot`, block pointers, `fast_libentry`, `num_warps`,
  `num_stages`, or mixed precision.

## Handoff

Remote smoke is a Coder gate only. Route `triton_grouped_topk_009.py` to
Verifier for independent correctness, paired timing, and targeted profiling
before adoption. The 5% wall threshold is a Verifier measurement, not a Coder
claim.
