# Decision 006

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"006","reference_implementation":"triton_grouped_topk_003.py","reference_report":"rounds/report_003.md","language":"triton","backend":"gcu","target_profile":"triton_gcu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"measurement-bound","intervention":"no stable intervention clears the adoption threshold under the unchanged GCU evidence","allowed_changes":[],"invariants":["ModelNew public contract","benchmark semantics","accepted kernel and report remain canonical","one direct Triton-GCU launch per forward","output-pool and metadata-cache lifecycle guarantees","caller-selected device and current stream"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md`. The recorded selection-network, winner-tree, gather, and cumsum failures are MLU-specific evidence and do not establish a GCU result; their reconsideration conditions require a matched target/compiler probe that is absent.
- Consulted `prompts/coder_targets/triton_gcu.md`. The current profile still records no GCU device-duration events, leaves `tl.dot`, block pointers, stages, vectorization, and stream/context semantics unproven, and establishes direct launch with `num_warps=1` as the observed path.
- Round 004 remains the only Verifier-backed test of the launcher-context family: it passed correctness and lifecycle/device/stream guardrails but achieved only `2.058982586436897%`, below the `5%` adoption threshold. Round 005 then aborted because no distinct family had a justified path. No new Verifier report, profiler evidence, runtime fingerprint, or accepted candidate was added before Round 006.

## Rationale and Evidence

The canonical implementation remains `triton_grouped_topk_003.py`, supported by accepted `rounds/report_003.md`. It already provides one direct launch per forward, output-pool reuse, and exact-key metadata caching. The only distinct plausible family in the persisted backlog is kernel selection/dataflow, but the recorded GCU profile has no attributable device-kernel duration and no same-runtime microbenchmark proving that a changed reduction, compaction, precision, or launch configuration is both supported and cheaper.

Repeating `launcher-context-specialization` would violate the post-no-improvement rule because no new Verifier-backed observation explains how it could clear five percent. Selecting `kernel-selection-dataflow` would be an unproven intervention rather than a defensible hypothesis: the available MLU anti-patterns are not transferable, `tl.argmax` tie behavior is constrained, and changing the established reduction structure would risk exact top-k ordering and numerical semantics without causal device evidence. Therefore no distinct change family currently supports a falsifiable expected wall improvement of at least five percent, and this round aborts without Coder or Verifier work.
