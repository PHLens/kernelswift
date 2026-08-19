# Decision 003

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"003","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"gcu","target_profile":"triton_gcu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"none","intervention":"no stable intervention closes the residual device gap: the fused single-warp Triton elementwise kernel's math-dialect cos/sin and per-element index-decomposition overhead cannot be tuned to beat the vendor-optimized eager elementwise ops","allowed_changes":[],"invariants":["ModelNew public contract","output tuple structure and shape [4,32,128] fp32","state_dict keys {inv_freq, position_angles} unchanged","numerical semantics (atol=1e-2, rtol=1e-2, equal_nan=True)"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md`. The catalog records that BLOCK/launch-config tuning on MLU produced no-improvement, and that reduction/selection-network rewrites regressed. None of those entries gives a positive, evidence-backed lever here.
- The Round 2 result already exhausted the one falsifiable, mechanism-backed intervention this workload admits (grid parallelism): device execution collapsed ~10x (5.15 ms → ~0.517 ms) but still lands at wall 0.525050 ms vs the 0.464657 ms eager baseline (-13.00%). The residual is not a parallelism defect — 128 programs already far oversubscribe the device's `multi_processor_count=2`.
- Every remaining candidate lever is either unproven on GCU or known-marginal: `num_warps > 1` is Unsupported-in-evidence and adds no independent parallel lanes on a 2-MP device; `vectorize` is Unknown and the kernel's loads are gather-style with index decomposition; BLOCK retuning is marginal with no evidence it moves device time 16%; constant folding space is negligible.
- The binding anti-pattern: the launch-overhead win (~97 us) is already realized, and the remaining gap is the intrinsic cost of a hand-written, math-dialect (`tl.cos`/`tl.sin`) single-warp elementwise kernel versus the vendor's highly-tuned eager `cos`/`sin` kernels. No remaining change family has a falsifiable >=5% mechanism.

## Rationale and Evidence

Across two proceeding rounds the fusion hypothesis has been fully explored and is now measured at its best achievable point:

- Round 1 (`report_001.md`) proved correctness and the launch-collapse mechanism (13 → 1 launch, launch 139.38 → 9.70 us/call) but a `grid=(1,)` defect made the kernel serially process 16384 elements (wall 5.162427 ms).
- Round 2 (`report_002.md`) fixed the grid to `BLOCK=128, grid=(128,)`, collapsing device time ~10x to ~0.517 ms, and retained the full launch saving (launch 8.51 us/call vs baseline 105.44 us/call). Yet wall time is 0.525050 ms vs 0.464657 ms, an improvement of -13.00%.

The arithmetic is decisive. The eager baseline's wall (0.464657 ms) is ~0.36 ms device plus ~105 us launch; the fused candidate's wall (0.525050 ms) is ~0.517 ms device plus ~8.5 us launch. The fusion saved ~97 us of launch but the hand-written kernel costs ~157 us more device time than the eager elementwise chain (~44% slower on device). The two mechanisms nearly cancel, leaving a net -60 us wall gap. Reaching the >=5% threshold would require another ~84 us (16%) device reduction, which no remaining lever has evidence to deliver.

The residual device cost is the intrinsic price of a single-warp Triton elementwise kernel whose `cos`/`sin` lower through the MLIR math dialect and whose per-element `b/t/d` index decomposition (int div/mod) and branch select add overhead, versus the vendor's dedicated elementwise transcendentals. This is not a defect a third kernel rewrite can remove. With `performance_miss_streak` at 2 and the adoption threshold unmet after the mechanism was correctly fixed, no stable >=5% intervention remains; the correct terminal decision is abort.
