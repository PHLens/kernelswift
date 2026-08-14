# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"mlu","target_profile":"triton_mlu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel"}
```

## Optimization Intent

```json
{"bottleneck_class":"device-bound","intervention":"fuse the routing reduction into the target kernel","allowed_changes":["kernel dataflow"],"invariants":["ModelNew public contract","output dtype and shape"],"expected_wall_improvement_pct":8.0}
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
{"applicability":"not-applicable","reason":"kernel-only change"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the routing reduction into the target kernel","expected_causal_chain":["external routing kernels disappear","device time decreases","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"external_kernel_count_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output dtype and shape unchanged"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`; no matching failure invalidates this path.

## Rationale and Evidence

The accepted trace contains separate routing kernels that are inside the candidate's change boundary.
