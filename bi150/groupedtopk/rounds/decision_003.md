# Decision 003

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"003","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"partial-routing-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"fuse softmax, group-score reduction, and group masking into one direct Triton kernel while retaining library torch.topk for both group selection and final expert ordering","allowed_changes":["one direct Triton kernel over the fixed [83,256] fp32 gating regime","kernel-local softmax and (8,32) group-score reduction","kernel-local group-mask materialization","torch.topk final group and expert selection unchanged"],"invariants":["ModelNew public constructor and forward contract","hidden_states batch-size assertion","gating_output [83,256] fp32 contiguous regime","topk_weights [83,8] fp32 on cuda:0","topk_ids [83,8] int32 on cuda:0","softmax grouped top-k mathematical semantics","exact torch.topk ordering and active-set-dependent tie behavior for group and final expert selections","renormalize and routed_scaling_factor behavior","immutable base.py and unchanged harness","caller-selected device and current stream behavior"],"expected_wall_improvement_pct":6.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor gating_output shape=[83,256] dtype=fp32 layout=row_major memory=global
tensor scores shape=[83,256] dtype=fp32 layout=row_major memory=global
tensor group_scores shape=[83,8] dtype=fp32 layout=row_major memory=global
tensor masked_scores shape=[83,256] dtype=fp32 layout=row_major memory=global

# O Operations
load row <- gating_output[token,0:256]
compute scores = exp(row - logsumexp(row))
compute group_scores = max(reshape(scores,8,32),axis=1)
compute group_idx = torch_topk(group_scores,4)
compute masked_scores = where(group_membership(group_idx),scores,-inf)
compute expert_idx,expert_weights = torch_topk(masked_scores,8)
compute expert_weights = expert_weights / sum(expert_weights)
compute expert_weights = expert_weights * routed_scaling_factor
store scores[token,0:256] <- scores
store group_scores[token,0:8] <- group_scores
store masked_scores[token,0:256] <- masked_scores

# C Control
parallel token over 83
guard token < 83
if selected_group_member
end

# H Target Hints
target=triton_cuda
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only partial fusion; torch.topk remains the exact library selection boundary and no host allocation, cache, device, stream, concurrency, or lifecycle behavior is changed"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-003","intervention":"fuse softmax, group-score reduction, and group masking into one direct Triton kernel while retaining library torch.topk for both group selection and final expert ordering","expected_causal_chain":["softmax, group-score, and group-mask intermediate launches are fused into one direct kernel","the exact torch.topk calls remain for active-set-dependent group and expert ordering","intermediate tensor materialization and launch count decrease without changing integer selection results","scoped device_us_per_call and kernel_count_per_call decrease","unrounded interleaved wall_time median improves by at least five percent"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease versus baseline_adapter.py"},{"name":"device_us_per_call","expectation":"decrease versus baseline_adapter.py"},{"name":"fused_softmax_group_mask_kernels","expectation":"softmax, group-score, and mask work is represented by the direct kernel while torch.topk selection kernels remain"}],"guardrails":["correctness:pass","exact topk_ids equality for seeded, all-equal, two-expert-tie, and structured group-tie inputs","exact active-set-dependent torch.topk ordering preserved for group and final selections","topk_weights allclose atol=1e-2 rtol=1e-2","output shapes dtypes and cuda:0 placement unchanged","constructor and forward signatures unchanged","hidden_states batch-size assertion preserved","renormalization and routed scaling preserved","current caller stream and device behavior preserved"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Round 002 demonstrated that a custom final top-k replacement cannot reproduce BI150/PyTorch active-set-dependent ordering: the structured group-tie reference was `[32,0,64,96,4,3,1,2]`, while the repaired candidate returned `[0,32,64,96,7,6,4,5]`. This decision leaves both exact `torch.topk` selection boundaries intact.
- The profile supports the direct fixed-shape reductions and masking used only for fused preprocessing, while this decision does not require `tl.dot`, `num_warps`, `num_stages`, `fast_libentry`, block pointers, mixed precision, or a custom repeated argmax tie rule.
- The fused kernel must not replace or reorder the library `torch.topk` operations. A candidate that changes either selection boundary is a major deviation or correctness failure.
- Historical MLU compaction and selection-network failures are not direct CUDA evidence, but they support keeping the intervention narrow and avoiding dynamic compaction or custom sort networks.
- No output cache, device-context removal, stream change, or launcher replacement is permitted.

## Rationale and Evidence

The canonical baseline report identifies top-k gather at `48.7290625 us/call`, bitonic sort at `36.879697265625 us/call`, and `179.0703515625 us/call` total device time for `baseline_adapter.py`. This leaves a device-side opportunity, but Round 002 proves that replacing final selection is not semantically safe under the current target.

The partial-fusion boundary removes only preprocessing work whose semantics can be validated independently: row softmax, eight group maxima over the `(8,32)` view, and group-mask construction. Exact library `torch.topk` remains responsible for selecting the four groups and eight experts, including the observed active-set-dependent tie ordering. The evaluation is falsifiable through exact adversarial tie cases plus scoped kernel/device evidence and the unchanged five-percent wall threshold.

If the direct kernel cannot feed the unchanged library selections without changing values, active sets, ordering, stream behavior, or output semantics, classify the candidate as design-rejected or candidate-failed and retain `baseline_adapter.py` as canonical. A profiler-only improvement is insufficient for adoption.
