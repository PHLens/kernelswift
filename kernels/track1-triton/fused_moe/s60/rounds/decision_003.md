# Decision 003

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"003","reference_implementation":"triton_fused_moe_002.py","reference_report":"rounds/report_002.md","language":"triton","backend":"gcu","target_profile":"triton_gcu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"measurement-bound","intervention":"no stable intervention clears the 5% adoption threshold; remaining wall is launch-light (30 us of 390 us) and unattributable without GCU device-duration evidence","allowed_changes":[],"invariants":["ModelNew public contract","output dtype and shape","routing numerical semantics","int32 expert indexing","immutable harness and measurement fingerprint"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/bottleneck-judgment.md`; runtime-launch evidence (30.24 us/call) is 7.7% of wall (0.390289 ms), and the GCU exporter exposes no device-duration events, so the remaining ~92% of wall cannot be attributed to any compressible kernel, launcher, allocation, or host source. A measurement-bound stop requires proving the residual host time is harness-fixed, which the current exporter cannot establish.
- Consulted `references/anti-patterns.md`; the weight-cast-budget path repeats the same host/launcher family whose best observed GCU result (groupedtopk s60) was only +2.06%, below the 5% threshold.
- Consulted `prompts/coder_targets/triton_gcu.md`; `tl.dot`, `fast_libentry`, `num_stages`, and stream/context semantics are all Unknown or unproven, so no device-side or launcher-side intervention is falsifiable on this runtime.
- Consulted `references/invariants.md`; any weight-cast cache would require a Host Plan with load_state_dict invalidation that risks the immutable state_dict contract for an edge-of-threshold gain.

## Rationale and Evidence

Round 1 and Round 2 exhausted the two high-yield change families: per-token kernel fusion (147 to 8 launches, 10.55x) and routing fusion (8 to 3 launches, +26.57%). report_002's `evidence_for_next_round` names only the two remaining host-side fp16 weight casts (19.91 us/call launch) as further surface. Wall is now 0.390289 ms with only 30.24 us of launch overhead (7.7%); the residual ~92% is either device execution (unobservable on the GCU exporter, which reports no cat=kernel durations) or harness-fixed cost (seed, sync, build_case/load_state_dict). Eliminating the two weight-cast launches would recover at most ~19.91 us of launch overhead — marginally at, but not verifiably above, the 5% threshold (19.5 us) — and carries a load_state_dict invalidation correctness risk. No remaining intervention has a falsifiable, Verifier-backed expectation of at least 5% wall improvement against `triton_fused_moe_002.py`. The campaign should stop as measurement-bound.
