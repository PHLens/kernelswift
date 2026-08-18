# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"002","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"gcu","target_profile":"triton_gcu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"measurement-bound","intervention":"no stable intervention clears the 5% adoption threshold: Round 001 proved hand-written Triton fusion is slower than the GCU library elementwise/reduction ops, and the remaining compressible host budget (D2H sync alone) is below 5% while the MLM head library GEMM launches must be retained","allowed_changes":[],"invariants":["ModelNew public contract","output dtype and shape","numerical semantics","benchmark semantics"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md`. No single recorded failure invalidates fusion in isolation, but the measured Round 001 result is decisive: the fused kernel's added device-side compute dominates the launch/sync savings on GCU, a result consistent with the general principle that hand-written Triton reductions are not competitive with tuned library ops when `num_warps=1` and the workload is small (num_seq=4, seq_len <= 25, vocab=30522).
- Consulted `references/bottleneck-judgment.md`. Device ratio is unavailable on the GCU exporter (no `cat=kernel` events), so attribution rests on runtime-launch evidence and wall time. Round 001 showed the launch/sync budget is ~111 us/call (12.9% wall), of which only the D2H sync portion is independently removable without the (slower) fused kernel; the MLM head library GEMM launches are not removable. The removable host component is below the 5% threshold, matching the measurement-bound stop condition.
- Consulted MLU v2/v4 history: BLOCK_V tuning was no-improvement on MLU (2048 vs 1024), and the only additional MLU win (v4 +5.79%) came from `fast_libentry`, which is unavailable on GCU. Neither path transfers to GCU.
- `num_warps=1` is the only proven value on GCU; `num_warps>1` and `tl.dot` remain Unknown, so no alternative kernel dataflow has a verifier-backed 5% hypothesis.
- The D2H-sync-only path (copying `seq_lens` to host once without fusion) is estimated below 5%: it removes at most a small fraction of the ~111 us launch budget while leaving all 11 library launches in place.

## Rationale and Evidence

Round 001 (`rounds/report_001.md`) is the controlling evidence. The candidate was correct (correctness PASS, all guardrails pass) but 26.79% slower (wall 1.092186 ms vs reference 0.861388 ms). Both mechanism observables were confirmed — `runtime_launch_count_per_call` fell 11→6 and the `seq_lens.tolist()` D2H sync was fully eliminated — yet the fused kernel's device-side compute (~270 us/call added, from serializing `range(seq_len)` over the sequence axis with `num_warps=1` and `BLOCK_V=256`) dwarfed the ~49 us/call of launch/sync savings.

This falsifies the kernel-fusion change family on GCU: the library relu/log1p/4x `chunk.max(dim=0)` ops are already faster than a hand-written Triton elementwise+segment-max reduction for this small workload. The remaining optimization directions do not clear 5%:

1. Tuning the fused kernel attacks the wrong root cause — GCU Triton device-compute efficiency, not launch overhead — and is contradicted by MLU v2 and the GCU profile.
2. Reverting to library ops forfeits the only confirmed mechanism (D2H sync elimination).
3. Eliminating only the D2H sync while keeping library ops removes a fraction of the ~111 us launch budget (well under 5% of 861 us wall) while the MLM head library GEMM launches are fixed.

The residual host time is library-fixed launch overhead plus harness-fixed synchronization, and the residual device time is already efficiently served by library ops. There is no falsifiable intervention expected to improve benchmark wall time by at least 5%, so the campaign stops as measurement-bound.
