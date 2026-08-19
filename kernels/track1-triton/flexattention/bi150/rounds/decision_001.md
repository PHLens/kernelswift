# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"measurement-bound","intervention":"no stable intervention clears the adoption threshold; device time is a single vendor-tuned fused Ixmma flash-attention kernel (CausalM_t=2) and the remaining ~91% of wall time is harness-fixed host overhead outside the candidate boundary","allowed_changes":[],"invariants":["ModelNew public contract","output dtype and shape","benchmark semantics"],"expected_wall_improvement_pct":0.0}
```

## Unified Sketch

N/A: aborted

## Host Plan

```json
{"applicability":"not-applicable","reason":"aborted"}
```

## Evaluation Contract

```json
{"applicability":"not-applicable","reason":"aborted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. No single entry is a direct precondition
  match (the catalog records MLU590-H8 grouped-topk selection-network failures),
  but the same structural lesson governs this round: a hand-written Triton kernel
  that re-implements a vendor library kernel's dataflow cannot be justified to
  beat that library kernel without a matched primitive-probe win. Here the device
  work is already a single vendor-tuned `FlashAttnFwdF16Ixmma` tensor-core
  flash-attention kernel; a Triton rewrite (naive `tl.dot` QK^T + softmax + PV, or
  tiled online-softmax) would replace one fused kernel with multiple kernels plus
  materialized intermediates and has no evidenced path to beat the vendor's Ixmma
  path. This mirrors the groupedtopk `torch.topk` library-kernel dead end.
- The `tl.dot` primitive is now recorded Supported in
  `prompts/coder_targets/triton_cuda.md` for `(32,32) @ (32,32)`, but that is a
  correctness claim at one square tile shape, not a performance claim for a
  `(83,83,64)` attention tile, and `num_warps`/`num_stages`/block pointers remain
  Unknown. No resource-bounded primitive win has been demonstrated on BI150, so
  the anti-patterns requirement to prove a primitive win before committing to a
  rewrite remains unmet.

## Rationale and Evidence

This task is the causal twin of task 6 (`mm_encoder_attention`), and `report_000.md`
establishes the same terminal structure. The sole meaningful difference is the
SDPA backend dispatch: causal (`is_causal=True`) SDPA lowers to
`FlashAttnFwdF16Ixmma<128,128,16,64,64,(CausalM_t)2,(AlibiMode_t)0,...>`, versus
task 6's non-causal `(CausalM_t)0`. Both are the same fused Iluvatar Ixmma
flash-attention tensor-core kernel; the causal flag only changes the
`CausalM_t` mask enumeration. There is no `bmm`/`softmax` decomposition, no
routing or wrapper kernel inside the candidate boundary, and no materialized
intermediate to remove — the attention is already maximally fused into one
vendor-tuned kernel.

The workload is host-bound with harness-fixed overhead. From `report_000.md`, the
reference wall median is `0.150070 ms` (= `150.070 us`) while device time is
`12.8803125 us/call` at `0.84 kernels/call`, giving `device_ratio ≈ 0.086` (~91%
of wall is host/launch overhead). The ~137 us outside the kernel is
harness-fixed: `auto_bench.py`'s `time_forward` runs `set_seed(seed)` (iterating
`manual_seed_all` over every accelerator) plus `sync_devices()` (a full
`torch.cuda.synchronize()`) on every timed call, and clones inputs — all outside
`ModelNew.forward` and outside any candidate change boundary. The `forward` body
contributes only zero-copy `view`/`transpose`/`reshape` plus a single SDPA
dispatch, so there is no candidate-owned host work to compress. A
`torch.compile`-driven dispatch-reduction hypothesis is not falsifiable at 5% in
this regime: the dominant host cost is fixed seed plus synchronization, not
per-op dispatch, and the triton_cuda profile records `torch.compile` only for
trivial file-backed CUDA add-one functions, with attention-graph coverage under
this harness explicitly unproven.

The upper bound on any device-only optimization is decisive: even a hypothetical
zero-cost Triton attention could remove only the `12.880 us` device time, and a
Triton kernel necessarily introduces its own launch/JIT overhead on first call.
Clearing the 5% threshold requires `150.070 us → ≤ 142.57 us`, i.e. eliminating
the entire device time with zero added host cost — not defensible against a
vendor-tuned Ixmma kernel with unproven `tl.dot` performance.

Per `references/bottleneck-judgment.md`, a stop is measurement-bound when
normalized evidence shows remaining device work below the stated bound and the
remaining host time is harness-fixed. Both hold: device work is one
non-decomposable vendor kernel (12.880 us, ratio 0.086), and the remaining host
time is harness seed/synchronization that the candidate must not alter. No
falsifiable intervention is expected to improve unrounded median wall time by at
least 5% against `baseline_adapter.py`, so this round aborts rather than
manufacturing an unprovable proceeding decision.

**Follow-up deliverable (advisory, for Orchestrator):** Following the task 6
precedent (`triton_mm_encoder_attention_001.py`), even though optimization has no
space, a correct naive Triton *causal* attention (a `tl.dot` QK^T + lower-
triangular causal-mask softmax + PV kernel, reproducing the `CausalM_t=2` mask
within `atol=1e-2, rtol=1e-2`) is a valuable submission artifact. This is a
deliverable, not an optimization: it is expected to be correct but slower than
the vendor kernel, and it does not enter the adoption criterion. Because the
Designer contract owns only the `abort` decision and does not dispatch coding,
the decision to produce this deliverable (and whether it routes through a
separate Coder dispatch outside the optimization loop, as task 6 did) belongs to
Orchestrator. The causal-mask semantics to reproduce are documented in
`report_000.md` (SDPA dispatch section) and `project.md#semantics`.
