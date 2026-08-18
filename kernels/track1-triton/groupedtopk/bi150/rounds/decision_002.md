# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"002","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"grouped-topk-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"fuse softmax, grouped top-k selection, masking, renormalization, and routed scaling into one direct Triton kernel for the fixed [83,256] fp32 gating regime","allowed_changes":["one direct Triton kernel over gating_output","kernel-local grouped reduction and selection dataflow","kernel-local output materialization"],"invariants":["ModelNew public constructor and forward contract","hidden_states batch-size assertion","gating_output [83,256] fp32 contiguous regime","topk_weights [83,8] fp32 on cuda:0","topk_ids [83,8] int32 on cuda:0","softmax grouped top-k mathematical semantics","PyTorch top-k ordering and tie behavior","renormalize and routed_scaling_factor behavior","immutable base.py and unchanged harness"],"expected_wall_improvement_pct":8.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor gating_output shape=[83,256] dtype=fp32 layout=row_major memory=global
tensor topk_weights shape=[83,8] dtype=fp32 layout=row_major memory=global
tensor topk_ids shape=[83,8] dtype=int32 layout=row_major memory=global
tile scores shape=[8,32] dtype=fp32 memory=register
tile group_scores shape=[8] dtype=fp32 memory=register

# O Operations
load scores <- gating_output[token,0:256]
compute scores = exp(scores - sum(scores))
compute group_scores = max(reshape(scores,8,32),axis=1)
compute selected_groups = stable_topk(group_scores,4)
compute masked_scores = where(group_membership(selected_groups),scores,-inf)
compute selected_weights,selected_ids = stable_topk(masked_scores,8)
compute selected_weights = selected_weights / sum(selected_weights)
compute selected_weights = selected_weights * routed_scaling_factor
store topk_weights[token,0:8] <- selected_weights
store topk_ids[token,0:8] <- selected_ids

# C Control
parallel token over 83
guard token < 83
for selection_round over 8
if equal_values_then_lower_expert_id_first
end

# H Target Hints
target=triton_cuda
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; direct launch and caller stream/device behavior remain unchanged, with no output cache or model-global state"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-002","intervention":"fuse softmax, grouped top-k selection, masking, renormalization, and routed scaling into one direct Triton kernel for the fixed [83,256] fp32 gating regime","expected_causal_chain":["separate softmax, group-selection, mask, top-k, renormalization, and scaling launches are replaced by one direct kernel","kernel launches and intermediate tensor materializations decrease","scoped device_us_per_call and kernel_count_per_call decrease","unrounded interleaved wall_time median decreases by at least five percent"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease versus baseline_adapter.py"},{"name":"device_us_per_call","expectation":"decrease versus baseline_adapter.py"},{"name":"topk_selection_and_mask_kernels","expectation":"baseline library selection and mask intermediates are absent or reduced"}],"guardrails":["correctness:pass","exact topk_ids equality including PyTorch ordering and ties","topk_weights allclose atol=1e-2 rtol=1e-2","output shapes dtypes and cuda:0 placement unchanged","constructor and forward signatures unchanged","hidden_states batch-size assertion preserved","renormalization and routed scaling preserved","current caller stream and device behavior preserved"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- The updated `triton_cuda` profile supports the required `(256,)` to `(8,32)` reshape, axis-1 and axis-0 float32 reductions, exp, zeros, full, where, broadcast, and four-iteration static range in the matched grouped-topk probe.
- Repeated argmax tie behavior is constrained, not proven. The candidate must implement and correctness-test stable PyTorch-compatible ordering, with lower expert ID used as the deterministic equal-value rule where the reference requires it. A tie mismatch is a correctness failure, not an acceptable approximation.
- This decision does not require `tl.dot`, `num_warps`, `num_stages`, `fast_libentry`, block pointers, mixed precision, arbitrary layouts, or non-contiguous inputs.
- Historical MLU winner-tree, full sort-network, dynamic gather, and cumsum failures are not direct CUDA evidence, but they warn against expanding the design beyond the single fixed-shape fused path. The sketch therefore uses only the matched grouped-topk primitive envelope and one direct launch.
- No host cache, device-context removal, stream change, or launcher replacement is permitted; those lifecycle semantics remain unproven and outside this change family.

## Rationale and Evidence

The canonical baseline report identifies top-k gather at `48.7290625 us/call`, bitonic sort at `36.879697265625 us/call`, and a `179.0703515625 us/call` total device time for `baseline_adapter.py`. The baseline device ratio is mixed at `0.3769941822`, so a device-side intervention with a separately observable launch and device-work mechanism is justified.

The updated target profile provides matched evidence for the fixed grouped shape: contiguous fp32 length 256, reshape to `(8,32)`, group reductions, exp, sum, argmax over an eight-element vector, fill/where/broadcast, and static four-iteration control. The intervention fuses the baseline's softmax, group filtering, final selection, renormalization, and scaling into one direct Triton kernel while retaining the public host contract.

The central falsifier is exact integer-ID comparison against PyTorch, including ties and output ordering. If stable tie behavior cannot be implemented or correctness fails, the candidate is rejected. If correctness passes but the paired wall median does not improve by at least 5%, the candidate is not adopted regardless of profiler improvements. The expected gain is from removing intermediate allocations and separate library launches, and will be tested through the named kernel-count, device-time, and wall-time observables.
