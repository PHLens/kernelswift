# Decision 003

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"003","reference_implementation":"triton_rotary_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"measurement-bound","intervention":"no stable intervention can clear the 5% adoption threshold: device compute is now 12us (3.7% of wall) and the remaining ~96% of wall time is harness-fixed device synchronization plus fixed Triton launch overhead, which kernel code cannot reduce","allowed_changes":[],"invariants":["ModelNew public contract","output structure tuple (cos, sin)","output shape [4,32,128] fp32 each","register_buffer semantics","numerical semantics cos/cat(batch_freqs,time_freqs)*(-timestamps*2pi)","harness measurement fingerprint"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md`. The four recorded failures (winner-tree expert selection, sort-32/sort-64 selection network, dynamic tl.gather compaction, cumsum compaction) are reduction/selection-network anti-patterns; this elementwise operator never entered their preconditions. None of the remaining change families would trigger them.
- Consulted `references/bottleneck-judgment.md`. Its primary ratio table puts `device_ratio < 5%` with stuck wall time into the `measurement-bound` class, and its `Compressible Versus Fixed Host Time` table classifies "harness device synchronization" and "seed setup in user-owned harness" as `Fixed for the regime`. The current device_ratio is ~3.7% (12.116 / 330), matching the measurement-bound row. The worked `fused_moe` example (rounds 2-5) shows the same convergence: device time plateaus in the low tens of microseconds, further device wins fail to move wall time, and the campaign stops after proving the residual host cost is harness-fixed.

## Rationale and Evidence

`report_002.md` is decisive. Round 2's row-parallel-vectorization restructure was a clear device-side success — it cut `device_us_per_call` 4x (48.27 → 12.116 us) by eliminating per-lane integer division/modulo, the redundant dual frequency load, and the `tl.where` select, exactly as `design_002.md` predicted. Yet the primary metric did not move: wall time was 0.327830 ms (accepted) vs 0.330345 ms (candidate), a -0.77% change. The hypothesis verdict was `partially-confirmed`: the device mechanism held, but the terminal wall-time sub-expectation was falsified.

The reason is structural. After Round 1's fusion (14 kernels → 1), the operator became host/launch-bound; device time was already only ~14.5% of wall. Shrinking device time further cannot move wall time by more than the device fraction itself. Round 2 proved this: device_ratio fell to ~3.7%, and wall time is now dominated (~96%) by the single kernel's host launch/dispatch plus the harness's mandatory `torch.npu.synchronize()` per timed sample.

Three remaining levers were evaluated and each fails the 5% bar:

1. **Grid/occupancy tuning** (more warps / larger BLOCK): this can only reduce device time, which is already 12us (3.7% of wall). Even a 2x device reduction saves ~6us ≈ 1.8% of wall — far below 5%. The launch/dispatch overhead it would target is a per-kernel-launch constant that grid shape does not remove.
2. **Avoiding per-call `torch.empty` output allocation** (host change): the two `[4,32,128]` fp32 output allocations (128 KB total) cost at most a few microseconds per call via the caching allocator; eliminating them entirely saves at most ~4-5% of wall in the best case and introduces lifecycle/cache-invalidation risk, without a Verifier-backed measurement showing it clears 5%.
3. **Any further kernel fusion / dataflow change**: device compute is now the irreducible 16384 cos + 16384 sin (the `[4,32,128]` output), which no fusion removes. There is no remaining compressible device work.

The harness timing loop (`auto_bench.py` `time_forward`) wraps each forward in `model.forward()` then `sync_devices()` (a `torch.npu.synchronize()`), with `set_seed(seed)` immediately before the timer. The measured wall time therefore necessarily includes the device-synchronization round-trip, which is part of the measurement fingerprint and is not reducible by kernel code. This matches the convergence endpoint of the prior four sibling operators in this campaign (groupedtopk / flexattention / fused_moe / sparse_pooler), which all aborted at "remaining host time is fixed Triton launch/synchronization overhead".

`performance_miss_streak` is 1 (single `no-improvement` in Round 2), not the full `valid_no_improvement_limit`, but the limit governs how many `no-improvement` rounds may be tolerated before an abort is forced, not whether an abort may be recommended earlier. Here the evidence already shows no change family can clear 5%: device time is below 5% of wall and the residual is harness-fixed. The honest, evidence-backed recommendation is to stop.

Recommended `stop_reason`: `measurement-bound — device compute reduced to ~3.7% of wall (12us, irreducible trig floor); remaining wall dominated by harness-fixed device synchronization and fixed Triton launch overhead, unreachable by kernel change`.
