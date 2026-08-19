# Decision 004

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"004","reference_implementation":"triton_flexattention_002.py","reference_report":"rounds/report_003.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"no falsifiable intervention is expected to clear the 5% adoption threshold; the remaining wall time is fixed backend launch/dispatch plus harness synchronization, and the one device-side speedup (tl.dot Cube routing) provably regresses wall through a net-negative host penalty","allowed_changes":[],"invariants":["ModelNew public contract","output shape [83,512] and fp16 dtype","causal numerical semantics","benchmark wall-time semantics"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/bottleneck-judgment.md`. The candidate is host-bound (device_ratio 0.184 for the accepted vector kernel; 0.075 for the rejected Cube kernel). The fused_moe worked example rounds 3-5 and the groupedtopk-ascend `decision_003.md` abort both document the same terminal state: once wall is host-dominated and the host cost is fixed backend launch/dispatch plus harness synchronization, no device-side or host-side falsifiable intervention clears 5%. This campaign reproduces that state.
- Consulted `references/invariants.md`. The remaining host cost is backend/harness-fixed (Triton launch/dispatch + `sync_devices()` in `time_forward`), not candidate-compressible. Altering `base.py` or the harness to manufacture a speedup is forbidden. No candidate-side mechanism preserves the public contract while compressing fixed launch overhead.
- Consulted `references/anti-patterns.md`. No entry names a host-side launcher reduction or a device-side vector-path improvement that avoids the Cube host penalty; the catalog's device-selection failures are unrelated to this causal-SDPA shape.
- Consulted `prompts/coder_targets/triton_ascend.md`. The fast-launcher path (`fast_libentry`) and stream/context semantics are `Unknown` on Ascend; direct launch is the proven launcher path and is already in use. `tl.dot` is Supported but its use provably incurred a +55 us/call host penalty that outweighed the device saving.

## Rationale and Evidence

Round 003 conclusively mapped the runtime's device/host tradeoff. The `tl.dot` Cube intervention achieved its device mechanism exactly — device time halved from 54.43 to 24.05 us/call, reaching the ~25 us `aclnnFlashAttentionScore` structural floor — but wall time regressed -8.34% (0.296535 → 0.321280 ms) because host-side launch/dispatch/synchronization cost rose ~55 us/call, more than offsetting the ~30 us device saving. Report 003's Level0 trace shows the increase is not op enqueue (~0.03 us/call) nor HostToDevice transfer (zero duration): it is the Triton launch / Cube-dispatch / stream-sync path, which the candidate cannot control.

This closes both remaining lever categories with Verifier-backed evidence:

1. **Device-side**: the accepted vector `tl.sum` kernel sits at 54 us (19% of 282 us wall), with a 24 us floor. The *only* known way to reach that floor is Cube routing (`tl.dot` or an equivalent matrix-unit path), which provably costs +55 us host — a net-negative tradeoff already measured at -8.34%. A pure-vector micro-optimization (`num_warps` tuning or reduction re-layout) has no evidence of the ~2x device reduction needed, and even a full 54→24 us saving (30 us, ~10.6% wall) is unattainable without incurring the Cube host penalty. No falsifiable device intervention with expected ≥5% wall gain remains.

2. **Host-side**: `fast_libentry`/stream/context are `Unknown` on Ascend (no probe; probing is Verifier/Coder territory, not Designer, and an unproven normative launcher requirement would be a capability-miss). The groupedtopk-ascend campaign on this identical runtime already measured ~107 us of fixed Triton launch/dispatch and aborted. The residual host time is backend launch/dispatch plus the harness's per-iteration `sync_devices()`, both outside the candidate's change boundary.

The remaining wall time is fixed backend/harness overhead. Optimizing `base.py` or altering the harness to manufacture a speedup is forbidden by the invariants. Accordingly this round recommends halting: the decision is `abort` with no further candidate dispatched. The accepted canonical `triton_flexattention_002.py` (wall 0.281900 ms, cumulative +31.1% vs baseline) stands as the campaign result.
