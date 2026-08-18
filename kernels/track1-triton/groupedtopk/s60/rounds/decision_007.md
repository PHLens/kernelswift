# Decision 007

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"007","reference_implementation":"triton_grouped_topk_003.py","reference_report":"rounds/report_003.md","language":"triton","backend":"gcu","target_profile":"triton_gcu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"measurement-bound","intervention":"no stable intervention clears the adoption threshold under the currently recorded GCU evidence","allowed_changes":[],"invariants":["ModelNew public contract","benchmark semantics","accepted kernel and report remain canonical","one direct Triton-GCU launch per forward","output-pool and metadata-cache lifecycle guarantees","caller-selected device and current stream"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md`. The recorded winner-tree, sorting-network, gather, and cumsum results are MLU-specific and cannot serve as GCU performance or lowering evidence. Their reconsideration conditions require a matched target/compiler probe, which has not yet run.
- Consulted `prompts/coder_targets/triton_gcu.md`. SSH access makes the recorded S60 harness executable by a future Verifier, but the target profile still records no GCU device-duration events, leaves alternative reduction/dataflow primitives unproven, constrains `tl.argmax` tie behavior, and establishes direct launch with `num_warps=1` as the observed path.
- Round 004 remains the only Verifier-backed test of `launcher-context-specialization`: all correctness and lifecycle/device/stream guardrails passed, but the authoritative paired wall improvement was only `2.058982586436897%`. Rounds 005 and 006 aborted without candidates. No new Verifier report, profiler trace, runtime fingerprint, or matched GCU microbenchmark has been supplied in Round 007.

## Rationale and Evidence

The canonical implementation remains `triton_grouped_topk_003.py`, with accepted evidence in `rounds/report_003.md`. It already executes one direct Triton-GCU launch per forward and retains the validated output-pool and exact-key metadata-cache behavior. The new SSH fact removes an environment-execution obstacle and lowers the cost of obtaining evidence; it does not itself identify a compressible component or support an expected wall-time improvement.

A distinct `kernel-selection-dataflow` decision is not defensible yet. The current GCU profile has no attributable device-kernel duration, and no same-runtime microbenchmark has shown that a changed reduction, compaction, precision, or launch configuration is supported, preserves exact top-k ordering and numerical semantics, and reduces wall time. Runtime-launch duration remains diagnostic only and cannot be substituted for device time. Repeating `launcher-context-specialization` would violate the post-no-improvement rule because no new Verifier-backed observation explains how it could clear five percent.

Therefore no falsifiable intervention with a defensible expected wall improvement of at least five percent exists at design time. The next eligible proceeding decision requires a future Verifier artifact from the now-available S60 execution path, such as a matched GCU microbenchmark or authoritative targeted observation naming a candidate-owned mechanism; this Designer does not implement or measure that probe in Round 007.
