# Decision 004

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"004","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"mixed","change_family":"two-stage-routing-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"replace the baseline preprocessing and post-group-mask tensor path with two direct Triton kernels separated by exact torch.topk group selection, while retaining exact torch.topk for final expert selection","allowed_changes":["stage-one direct Triton kernel for fp32 softmax and (8,32) group-score reduction","exact torch.topk(group_scores,4) group selection unchanged","stage-two direct Triton kernel for group_idx-dependent masked_scores construction","exact torch.topk(masked_scores,8) final expert selection unchanged","per-forward temporary tensors governed by the Host Plan"],"invariants":["ModelNew public constructor and forward contract","hidden_states batch-size assertion","gating_output [83,256] fp32 contiguous regime","topk_weights [83,8] fp32 on cuda:0","topk_ids [83,8] int32 on cuda:0","softmax grouped top-k mathematical semantics","exact torch.topk ordering and active-set-dependent tie behavior for group and final expert selections","seeded, all-equal, two-expert-tie, and structured group-tie semantics","renormalize and routed_scaling_factor behavior","immutable base.py and unchanged harness","caller-selected device and current stream behavior","per-forward allocation ownership and no cross-call aliasing"],"expected_wall_improvement_pct":6.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor gating_output shape=[83,256] dtype=fp32 layout=row_major memory=global
tensor scores shape=[83,256] dtype=fp32 layout=row_major memory=global
tensor group_scores shape=[83,8] dtype=fp32 layout=row_major memory=global
tensor group_idx shape=[83,4] dtype=int32 layout=row_major memory=global
tensor masked_scores shape=[83,256] dtype=fp32 layout=row_major memory=global

# O Operations
load row <- gating_output[token,0:256]
compute scores = exp(row - logsumexp(row))
compute group_scores = max(reshape(scores,8,32),axis=1)
store scores[token,0:256] <- scores
store group_scores[token,0:8] <- group_scores
compute group_idx = torch_topk(group_scores,4)
load scores <- scores[token,0:256]
compute masked_scores = where(group_membership(group_idx),scores,-inf)
store masked_scores[token,0:256] <- masked_scores
compute expert_weights,expert_ids = torch_topk(masked_scores,8)
compute expert_weights = expert_weights / sum(expert_weights)
compute expert_weights = expert_weights * routed_scaling_factor

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
{"applicability":"required","affected_scope":["ModelNew.forward","stage-one Triton preprocessing kernel","stage-two Triton post-group mask kernel","scores","group_scores","masked_scores","torch.topk group and expert calls"],"state_owner":"the current ModelNew.forward invocation; no module-global or cross-invocation state","lifetime":"allocate temporary tensors for one forward and release them after outputs are produced","allocation_reuse":"no cross-call cache or reuse; each invocation allocates distinct scores, group_scores, and masked_scores tensors","cache_key":["tokens","experts","dtype","device","stride","num_expert_group","topk_group","topk"],"invalidation":"there is no persistent cache; every invocation creates fresh buffers, and any unsupported shape, dtype, device, stride, scoring mode, or routing parameter follows the unchanged baseline path","concurrency":"each forward owns distinct temporary buffers; concurrent forwards and model instances must not alias or share state","device_stream_behavior":"allocate on gating_output.device and launch both direct Triton kernels on the caller's current stream; torch.topk consumes same-device tensors on that same stream; do not mutate device context or stream","unchanged_behavior":["non-target shapes and scoring modes use baseline semantics","exact torch.topk group and final expert ordering including active-set-dependent ties","returned topk_weights and topk_ids shapes dtypes and device","renormalization and routed scaling","hidden_states batch-size assertion","public constructor and forward signatures"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-004","intervention":"replace the baseline preprocessing and post-group-mask tensor path with two direct Triton kernels separated by exact torch.topk group selection, while retaining exact torch.topk for final expert selection","expected_causal_chain":["stage-one Triton kernel replaces separate softmax and group-score preprocessing launches","exact torch.topk group selection remains unchanged","stage-two Triton kernel replaces host masked_fill and mask expansion after group selection","exact torch.topk final expert selection remains unchanged","temporary materialization and preprocessing/masking launch overhead decrease","scoped kernel_count_per_call and device_us_per_call decrease","unrounded interleaved wall_time median improves by at least five percent"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease versus baseline_adapter.py"},{"name":"device_us_per_call","expectation":"decrease versus baseline_adapter.py"},{"name":"stage_one_softmax_group_kernel","expectation":"one direct kernel replaces separate softmax/group-score preprocessing work"},{"name":"stage_two_group_mask_kernel","expectation":"one direct kernel replaces post-group mask expansion/fill work while exact torch.topk kernels remain"}],"guardrails":["correctness:pass","exact topk_ids equality for seeded, all-equal, two-expert-tie, and structured group-tie inputs","exact active-set-dependent torch.topk ordering preserved for group and final selections","topk_weights allclose atol=1e-2 rtol=1e-2","output shapes dtypes and cuda:0 placement unchanged","constructor and forward signatures unchanged","hidden_states batch-size assertion preserved","renormalization and routed scaling preserved","per-forward buffers do not alias across concurrent invocations","current caller stream and device behavior preserved","non-target shapes and scoring modes preserve baseline behavior"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Round 003 was rejected because one preprocessing kernel could not consume `group_idx` produced by the exact library `torch.topk`. This decision explicitly authorizes the required second mask kernel after group selection and records that dependency in both the Sketch and Host Plan.
- Round 002 demonstrated that custom final top-k ordering is unsafe: the structured group-tie reference was `[32,0,64,96,4,3,1,2]`, while the repaired candidate returned `[0,32,64,96,7,6,4,5]`. Both exact library `torch.topk` boundaries remain unchanged here.
- The direct kernels use only the matched fixed-shape fp32 reductions, reshape, exp, where, broadcast, load, store, and direct one-dimensional launch envelope. No `tl.dot`, `num_warps`, `num_stages`, `fast_libentry`, block pointers, mixed precision, arbitrary layouts, or custom repeated argmax ordering is required.
- No cross-call cache, device-context removal, stream replacement, or global state is permitted. The Host Plan makes allocation ownership, invalidation, concurrency, and same-stream ordering explicit.
- Historical MLU dynamic compaction and selection-network failures are not direct CUDA evidence, but they support keeping the two stages narrow and avoiding custom selection networks.

## Rationale and Evidence

The canonical baseline report identifies top-k gather at `48.7290625 us/call`, bitonic sort at `36.879697265625 us/call`, and `179.0703515625 us/call` total device time for `baseline_adapter.py`. The baseline device ratio is mixed at `0.3769941822`, leaving a measurable device-side preprocessing and masking opportunity while exact selection remains a library responsibility.

The two-stage arrangement is the smallest dependency-correct extension of the rejected Round 003 plan. Stage one computes only values independent of group selection. The host/library boundary performs exact `torch.topk(group_scores,4)`. Stage two then consumes those indices to construct masked scores, after which exact `torch.topk(masked_scores,8)` preserves BI150/PyTorch active-set-dependent ordering. The Host Plan permits per-forward temporary allocations on the input device and current stream, with no aliasing or lifecycle ambiguity.

The >=5% claim is falsifiable: correctness must pass all specified tie cases, and targeted evidence must show reduced launch/device work plus an unrounded paired wall median improvement of at least five percent against canonical `baseline_adapter.py`. If the two-stage launches and temporary tensors erase the expected gain, or any tie, shape, scoring, stream, or concurrency guardrail fails, the candidate is rejected and the canonical baseline remains unchanged.
