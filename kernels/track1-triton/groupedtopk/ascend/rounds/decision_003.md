# Decision 003

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"003","reference_implementation":"triton_grouped_topk_002.py","reference_report":"rounds/report_002.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"no falsifiable intervention is expected to clear the 5% adoption threshold; the remaining host cost is fixed backend launch/dispatch overhead","allowed_changes":[],"invariants":["ModelNew public contract","output shapes and dtypes","grouped top-k numerical semantics","caller-selected NPU device","current stream preservation","benchmark wall-time semantics"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/bottleneck-judgment.md`. The candidate is host-bound
  (device_ratio 0.131, i.e. 35.134 us device of 267.220 us wall). The remaining
  ~232 us/call host cost is dominated by the Triton launch/dispatch path
  (~107 us measured fixed launch overhead independent of kernel size). The
  launcher path is "potentially compressible" only when the target profile and
  same-regime wall evidence name an implementable reduction; here the profile
  lists `fast_libentry`, `async_copy`, and stream/context semantics as
  `Unknown` (no Ascend probe), and direct launch is already the proven path, so
  no falsifiable launcher intervention with an expected >=5% gain exists.
- Consulted `references/invariants.md`. The remaining host cost is
  backend-fixed, not harness-fixed; however, removing or altering `base.py` or
  the harness to manufacture a speedup is forbidden, and no candidate-side
  mechanism remains that preserves the public contract while compressing the
  fixed launch overhead.
- Consulted `references/anti-patterns.md`. Device-side selection optimization
  (winner-tree, sort-32+sort-64 selection network, dynamic tl.gather compaction,
  cumsum compaction) regressed on the sibling runtime; on Ascend the device is
  only 13.1% of wall, so even eliminating all device work could not clear 5%
  after the fixed host overhead is accounted for. No entry names a host-side
  launcher reduction.
- Consulted `prompts/coder_targets/triton_ascend.md`. The fast-launcher path
  (`fast_libentry`) and stream/context semantics are `Unknown`; direct Triton
  launch is the proven launcher path and is already in use.

## Rationale and Evidence

Round 002 accepted a host-only allocation-reuse change that drove wall time to
0.267220 ms (+18.21% vs round 001), with device time statistically flat at
35.134 us/call and the single fused `_grouped_topk_kernel` at 1.0 kernel/call.
Report 002's `evidence_for_next_round` states the causal chain is confirmed and
the allocation-reuse mechanism is now exhausted: steady-state forward performs
0 fresh `torch.empty` allocations, and host time dropped from 291.743 to
232.086 us/call.

The candidate is host-bound (device_ratio 0.131). The residual ~232 us/call of
host time is the Triton launch/dispatch path, of which ~107 us is measured fixed
launch overhead independent of kernel size. This mirrors the fused_moe worked
example in `references/bottleneck-judgment.md`, rounds 3-5: at device_ratio
0.13-0.15 the source project tested whether device work could still move wall
time and found it could not, then stopped after proving the remaining host cost
was fixed backend overhead.

No remaining falsifiable intervention is expected to clear 5%:

- Launcher reduction: the fast-launcher and stream/context primitives are
  `Unknown` on this runtime, and direct launch is already the proven path; there
  is no implementable host change with a named, observable mechanism.
- Device-side work: at 13.1% of wall, even total elimination of device time is
  bounded below the 5% threshold once the fixed ~107 us launch overhead is
  excluded, and the anti-patterns catalog shows device selection optimizations
  regress on this primitive set.
- Allocation reuse: already exhausted (0 steady-state allocations).

The remaining host cost is fixed backend launch/dispatch overhead. Optimizing
`base.py` or altering the harness to manufacture a speedup is forbidden by the
invariants. Accordingly this round recommends stopping: the decision is
`abort` (halt) with no further candidate dispatched.
