# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"002","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_001.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"measurement-bound"}
```

## Optimization Intent

```json
{"bottleneck_class":"measurement-bound","intervention":"no falsifiable kernel-only intervention clears the 5% adoption threshold; remaining wall time is harness-fixed host overhead with device_ratio at 0.039","allowed_changes":[],"invariants":["ModelNew public contract (forward signature and output tuple)","output shapes and dtypes (grad_input_mix[2,1024,4], grad_mhc_scale[1], grad_mhc_base[4], all fp32)","numerical semantics (sigmoid chain plus both reductions)","base.py and harness remain immutable (seed setup and sync_devices are fixed for the regime)"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md` and `references/bottleneck-judgment.md`. The bottleneck-judgment guidance is decisive here: "Stop as measurement-bound only when normalized evidence shows remaining device work is below the stated bound and targeted Level 2 evidence shows the remaining host time is harness-fixed. Otherwise return the unresolved observation to the next Designer without inventing a cause."
- Round 001 already falsified the single attributable device-side mechanism. The two remaining device-side levers are both within (or adjacent to) the `kernel-fusion` change family and are arithmetically incapable of clearing 5%:
  - Removing the two `aclnnInplaceZero`/`ZerosLike` kernels (from `torch.zeros` accumulators, 1.44 us/call total) takes kernel count 3 → 1 but saves at most ~1.44 us device + 2 host launches — under 0.4% of the ~430 us wall.
  - Even eliminating 100% of remaining device time (16.85 us → 0) saves <17 us, which is <4% of ~430 us wall — still below the 5% threshold.
- A host-side allocation-reuse or dispatch-reduction change (different `change_family`) would target per-call `torch.empty_like`/`torch.zeros` and triton launch dispatch, but those are not the dominant remaining cost: the harness's own per-iteration seed setup and `sync_devices()` (both immutable, outside any allowed change boundary) dominate the ~96% host-side wall. Optimizing base.py or the harness to manufacture a speedup is explicitly forbidden.
- This matches the recorded cross-campaign signal: the prior nine operators in this campaign all converged at the same host-bound floor. This operator is at wall ≈ 0.43 ms, device ≈ 17 us, device_ratio ≈ 0.039, consistent with the harness-fixed floor.

## Rationale and Evidence

Round 001 (authoritative, `rounds/report_001.md`) fused the sigmoid elementwise chain plus both reductions into a single Triton kernel. The mechanism observables confirm the fusion worked as intended on device: `device_us_per_call` fell 41.06 → 16.85 us (2.4x) and the two baseline `aclnnReduceSum` kernels (22.4 us) plus eight elementwise kernels collapsed into one `_mhc_mix_bwd_fused_kernel` at 15.41 us. Kernel count fell 10 → 3 (the two remaining kernels are `aclnnInplaceZero`/`ZerosLike` from per-call `torch.zeros` accumulator allocation, 1.44 us/call total).

The hypothesis is falsified at the wall-time level: wall moved only 0.445723 → 0.431210 ms (+3.26%), below the 5% adoption threshold and within the reference's ~20% run-to-run noise (reference medians 0.380–0.475 ms across six runs). `device_ratio` fell from 0.092 to 0.039, meaning ~96% of candidate wall time is now host-side. This proves the operator is host-bound: reducing device compute does not move wall time.

Round 002 has no falsifiable kernel-only intervention that can clear 5%:
1. The 2 remaining zero-init kernels are worth ~1.44 us device total; removing them (kernel-fusion family, same as Round 001) cannot produce even 1% wall gain.
2. Eliminating all remaining device work (16.85 us → 0) is still <4% of the ~430 us wall — below threshold.
3. The ~96% host-side cost is dominated by harness-fixed per-iteration seed setup and `sync_devices()`, plus triton launch dispatch and per-call allocation, none of which is reachable by a kernel-only change and the immutable harness/base.py cannot be touched.

Per `bottleneck-judgment.md`, remaining device work is below the stated bound (device_ratio 0.039 < 5%) and the remaining host time is harness-fixed; the correct action is a measurement-bound stop, not another kernel-fusion attempt or an invented host change. The campaign-wide pattern (nine prior operators converging at the host-bound floor) corroborates that this operator has reached the harness-fixed overhead floor.

stop_reason: `measurement-bound floor — device_ratio 0.039, wall dominated by harness-fixed seed+sync; no kernel-only intervention clears 5%.`
