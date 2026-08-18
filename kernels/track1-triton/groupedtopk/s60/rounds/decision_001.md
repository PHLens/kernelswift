# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"gcu","target_profile":"triton_gcu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"fuse grouped softmax routing and masked top-k selection into one direct Triton-GCU kernel","allowed_changes":["kernel dataflow","Triton-GCU direct launch"],"invariants":["ModelNew public contract","output shapes and dtypes","grouped top-k numerical semantics","caller-selected GCU device"],"expected_wall_improvement_pct":25.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor gating_output shape=[T,E] dtype=fp32 layout=row_major memory=global
tensor output_weights shape=[T,K] dtype=fp32 layout=row_major memory=global
tensor output_ids shape=[T,K] dtype=int32 layout=row_major memory=global
tile token_scores shape=[BLOCK_E] dtype=fp32 memory=register

# O Operations
load token_scores <- gating_output[token,0:E]
compute scores = softmax(token_scores)
compute group_scores = max(reshape(scores,[num_group,experts_per_group]))
compute selected_scores,selected_ids = masked_topk(scores,group_scores,topk_group,K)
compute normalized_scores = renormalize(selected_scores)
store output_weights[token,0:K] <- normalized_scores
store output_ids[token,0:K] <- selected_ids

# C Control
parallel token over T
guard token < T
for group_rank over topk_group
for expert_rank over K

# H Target Hints
target=triton_gcu
num_warps=1
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only fusion; output tensors remain per-forward allocations so allocator reuse is not part of this decision"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse grouped softmax routing and masked top-k selection into one direct Triton-GCU kernel","expected_causal_chain":["separate GCU runtime launches for softmax, group selection, masking, and top-k disappear","runtime_launch_count_per_call decreases","benchmark wall time decreases","correctness remains within the base contract"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"runtime_launch_count_per_call","expectation":"decrease from the baseline scoped runtime trace"},{"name":"runtime_launch_us_per_call","expectation":"decrease in the GCU runtime-launch diagnostic"}],"guardrails":["correctness:pass","output dtype and shape unchanged","selected GCU device and current stream are preserved","device duration remains explicitly unavailable rather than inferred from runtime launch duration"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `prompts/coder_targets/triton_gcu.md`; the current runtime has no
  proven `fast_libentry` import path, so the decision uses direct Triton launch.
- Consulted `references/invariants.md`; no output cache, global state, device
  context replacement, or stream change is allowed in this kernel-only round.
- Consulted `references/bottleneck-judgment.md`; GCU runtime-launch evidence is
  diagnostic and cannot be relabeled as device kernel duration.

## Rationale and Evidence

The Phase 0 baseline trace contains 12 GCU runtime launches per forward call in
the `baseline_base` scope. A single direct-launch Triton kernel is expected to
replace those separate routing operations while preserving the exact grouped
top-k semantics. The GCU profile has a matched direct-launch smoke probe, and
the selected target hint is the only recorded `num_warps` value for this S60
architecture.
