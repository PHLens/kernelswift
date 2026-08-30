# Decision 005

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"005","reference_implementation":"triton_grouped_topk_003.py","reference_report":"rounds/report_003.md","language":"triton","backend":"gcu","target_profile":"triton_gcu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"measurement-bound","intervention":"no stable intervention clears the adoption threshold under the recorded GCU evidence","allowed_changes":[],"invariants":["ModelNew public contract","benchmark semantics","accepted kernel and report remain canonical","one direct Triton-GCU launch per forward","output-pool and metadata-cache lifecycle guarantees","caller-selected device and current stream"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md`. Its expert-selection and compaction failures are MLU-specific and cannot prove a GCU result, but their reconsideration conditions require matched runtime lowering or microbenchmark evidence that is absent here.
- Consulted `prompts/coder_targets/triton_gcu.md`. The current GCU profile supplies no device-duration events, leaves selection-dataflow alternatives unproven, and establishes only the existing direct-launch path with `num_warps=1`.
- Round 004 is a valid no-improvement in `launcher-context-specialization`: removing one candidate-owned stream lookup produced `2.058982586436897%`, below the `5%` threshold, while the remaining backend-internal stream lookup is outside candidate scope.

## Rationale and Evidence

The canonical implementation is `triton_grouped_topk_003.py`, with `rounds/report_003.md` as the accepted evidence. It already has one direct launch per forward, output-pool reuse, and exact-key metadata caching. Round 004 tested the remaining candidate-owned context lookup while retaining all guardrails; its authoritative paired wall median improved from `0.277370 ms` to `0.271659 ms`, only `2.058982586436897%`, so that change family cannot be repeated without new Verifier-backed evidence explaining a five-percent path.

No remaining host intervention has an attributable, candidate-owned compressible mechanism. The only visible extra stream lookup is in Triton-GCU backend launch internals, outside the candidate boundary. Kernel selection and dataflow changes are not justified because the selected profile records GCU runtime-launch events only, not device kernel duration, and no matched GCU microbenchmark or exporter establishes a selection bottleneck or a supported lower-cost lowering. Proceeding would either repeat the measured host family or introduce an unproven kernel change without the required justified five-percent path.
