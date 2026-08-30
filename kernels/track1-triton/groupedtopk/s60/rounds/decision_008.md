# Decision 008

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"008","reference_implementation":"triton_grouped_topk_003.py","reference_report":"rounds/report_003.md","language":"triton","backend":"gcu","target_profile":"triton_gcu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"measurement-bound","intervention":"no stable intervention clears the adoption threshold under the matched GCU probe evidence","allowed_changes":[],"invariants":["ModelNew public contract","benchmark semantics","accepted kernel and report remain canonical","one direct Triton-GCU launch per forward","output-pool and metadata-cache lifecycle guarantees","caller-selected device and current stream"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md`. The recorded winner-tree, sorting-network, gather, and cumsum results are MLU-specific and cannot serve as GCU performance or lowering evidence. Their reconsideration conditions require a matched target/compiler probe that has not established a winning alternative.
- Consulted `prompts/coder_targets/triton_gcu.md`. The Round 008 matched probe confirms direct launch and `num_warps=1`, but still reports `device_time_available=false`; no `cat=kernel` events exist, `tl.argmax` tie behavior remains constrained, and alternate reduction/dataflow primitives remain unproven.
- Round 004 remains the only Verifier-backed launcher-context candidate test and achieved only `2.058982586436897%`, below the adoption threshold. The Round 008 probe is not an optimization result and does not justify repeating that family.

## Rationale and Evidence

The canonical implementation remains `triton_grouped_topk_003.py`, with accepted evidence in `rounds/report_003.md`. The named Round 008 probe completed unchanged remote correctness and recorded one raw pair: reference adapter `0.282114 ms` versus canonical `0.282032 ms`. Both profile scopes emitted `1.0` `topsModuleLaunchKernel` per call. The trace is `1c04a827a50cbb065c1c9943e7c0f5ddf961aeca7f27c06aa2e912f5d2b1a7ec`, but it contains no `cat=kernel` device-duration events; runtime-launch values are diagnostic only and cannot be used as device time or a device ratio.

This matched evidence confirms execution and the unchanged one-launch path but identifies neither a candidate-owned compressible host component nor an attributable GCU device bottleneck. The single pair is measurement-only and cannot establish a five-percent optimization path. A kernel-selection/dataflow intervention would still lack target-supported lowering and semantic proof for exact top-k ordering/ties, while repeating launcher-context-specialization would violate the post-no-improvement rule without a new causal observation. Therefore no falsifiable intervention with a defensible expected wall improvement of at least five percent exists, and Round 008 aborts without Coder or further Verifier candidate work.
