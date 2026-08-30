# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"fuse grouped softmax routing and masked top-k selection into one direct Triton-Ascend kernel","allowed_changes":["kernel dataflow","Triton-Ascend direct launch"],"invariants":["ModelNew public contract","output shapes and dtypes","grouped top-k numerical semantics","caller-selected NPU device"],"expected_wall_improvement_pct":25.0}
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
target=triton_ascend
num_warps=1
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only fusion; output tensors remain per-forward allocations so allocator reuse is not part of this decision"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse grouped softmax routing and masked top-k selection into one direct Triton-Ascend kernel","expected_causal_chain":["separate NPU kernels for softmax, group selection, masking, and top-k disappear","kernel_count_per_call decreases","device_us_per_call decreases","benchmark wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease from 38.0 toward 1.0"},{"name":"device_us_per_call","expectation":"decrease from 329.034"}],"guardrails":["correctness:pass","output dtype and shape unchanged","selected NPU device and current stream are preserved"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `prompts/coder_targets/triton_ascend.md`; the profile records
  `tl.load`, `tl.store`, `tl.arange`, `tl.program_id`, `tl.zeros`, `tl.reshape`,
  `tl.max`, `tl.argmax`, `tl.sum`, `tl.exp`, `tl.where`, `tl.broadcast_to`,
  `tl.full`, and `tl.static_range` as Supported on Ascend910B4, and `num_warps`
  1/2/4 all compiled and ran. Direct Triton launch is the proven launcher path.
- Consulted `references/invariants.md`; no output cache, global state, device
  context replacement, or stream change is allowed in this kernel-only round.
- Consulted `references/anti-patterns.md`; the MLU "winner tree",
  "sort-32+sort-64 selection network", and "dynamic tl.gather compaction"
  failures are MLU590 evidence whose preconditions do not match this Ascend
  runtime, so they are not applied. This round uses the simple per-group
  argmax loop proven on the S60 GCU grouped-topk candidate, not those failed
  hierarchical/sort/gather approaches.
- Consulted `references/bottleneck-judgment.md`; the baseline device ratio is
  0.433 (mixed), and the 38-kernel fragmentation is the named device mechanism.

## Rationale and Evidence

The Phase 0 baseline decomposes grouped top-k routing into 38 separate NPU
kernels per forward call (aclnnTopk, aclnnInplaceScatterValue, aclnnSoftmax,
aclnnReduceSum, aclnnDiv, aclnnInplaceCopy_Cast, and others), with device time
329.034 us/call (43% of the 0.760135 ms wall) and the remaining ~57% in host-side
dispatch and synchronization. A single direct-launch Triton kernel replacing the
softmax, group-max selection, masking, and top-k selection collapses kernel
count toward 1 and removes the redundant scatter/mask/cast work. The equivalent
fusion on the S60 GCU backend reduced runtime launches from 12.0 to 1.0 and
improved wall time by 39.1%; the Ascend profile has a matched direct-launch
smoke probe and records all required primitives as Supported.
