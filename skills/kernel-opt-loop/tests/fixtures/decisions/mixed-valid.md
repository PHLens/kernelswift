# Decision 003

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"003","reference_implementation":"triton_example_002.py","reference_report":"rounds/report_002.md","language":"triton","backend":"mlu","target_profile":"triton_mlu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"mixed","change_family":"mixed-routing-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"fuse the routing reduction and reuse its shape-compatible output buffer","allowed_changes":["kernel dataflow","ModelNew.forward","output cache"],"invariants":["ModelNew public contract","output dtype and shape","numerical semantics"],"expected_wall_improvement_pct":9.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor scores shape=[T,E] dtype=fp32 layout=row_major memory=global
tile row shape=[BLOCK_E] dtype=fp32 memory=register

# O Operations
load row <- scores[token,0:E]
compute probs = softmax(row)
store output[token,0:K] <- topk(probs,K)

# C Control
parallel token over T
guard token < T

# H Target Hints
target=triton_mlu
num_warps=1
num_stages=2
```

## Host Plan

```json
{"applicability":"required","affected_scope":["ModelNew.forward","output cache"],"state_owner":"ModelNew instance","lifetime":"model lifetime","allocation_reuse":"reuse when shape, dtype, and device match","cache_key":["shape","dtype","device"],"invalidation":"replace on cache-key change","concurrency":"one model instance is not shared across concurrent forwards","device_stream_behavior":"caller-selected device and current stream are preserved","unchanged_behavior":["returned shape","returned dtype","numerical semantics"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-003","intervention":"fuse the routing reduction and reuse its shape-compatible output buffer","expected_causal_chain":["external routing kernels disappear","output allocations per call decrease","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"external_kernel_count_per_call","expectation":"decrease"},{"name":"output_allocations_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output dtype and shape unchanged","current stream preserved"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`; kernel fusion and buffer reuse are inseparable but separately observable.

## Rationale and Evidence

The wrapper owns the fused kernel output buffer, so the kernel and host changes must be realized together.
