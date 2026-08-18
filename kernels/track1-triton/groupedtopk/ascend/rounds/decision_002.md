# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"002","reference_implementation":"triton_grouped_topk_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"host","change_family":"allocation-reuse"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"reuse the two output tensors across compatible forwards on the ModelNew instance instead of allocating two fresh torch.empty tensors per call","allowed_changes":["ModelNew.forward output allocation path","output buffer cache on ModelNew instance"],"invariants":["ModelNew public contract","output shapes and dtypes","grouped top-k numerical semantics","caller-selected NPU device","current stream preservation","no silent cross-instance or concurrent sharing"],"expected_wall_improvement_pct":8.0}
```

## Unified Sketch

N/A: host-only change

## Host Plan

```json
{"applicability":"required","affected_scope":["ModelNew.forward","output buffer cache on ModelNew instance"],"state_owner":"ModelNew instance","lifetime":"model lifetime (persists across forwards)","allocation_reuse":"reuse cached weights/ids buffers only when tokens, topk, dtype, and device all match the cache key; otherwise allocate fresh buffers and replace the cache","cache_key":["tokens","topk","weights dtype","ids dtype","device"],"invalidation":"replace cached buffers when any cache-key component changes; buffers are never mutated in place across a returned reference that outlives the next forward","concurrency":"one ModelNew instance is not shared across concurrent forwards; no global or class-level cache","device_stream_behavior":"buffers live on the caller-selected NPU device; the current stream and caller-selected device are preserved; no explicit stream or device context is created","unchanged_behavior":["returned shape","returned dtype","numerical semantics","kernel launch grid and dataflow","public constructor and forward signature"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-002","intervention":"reuse the two output tensors across compatible forwards on the ModelNew instance instead of allocating two fresh torch.empty tensors per call","expected_causal_chain":["two per-call torch.empty output allocations disappear","host-side allocation overhead per call decreases","benchmark wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"output_allocations_per_call","expectation":"decrease from 2.0 toward 0.0"},{"name":"host_us_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output dtype and shape unchanged","numerical semantics unchanged","caller-selected device and current stream preserved"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/invariants.md`; the Host Plan names state owner, lifetime,
  cache key, invalidation, concurrency, and device/stream behavior, and forbids
  implicit global or class-level caches. Reuse is gated on a full cache-key match
  so correctness is preserved across shape/dtype/device changes.
- Consulted `references/anti-patterns.md`; the catalog (winner-tree, sort-32+sort-64
  selection network, dynamic tl.gather compaction, cumsum compaction) is MLU590
  device-side selection evidence. None of its preconditions match a host-side
  allocation change on Ascend910B4, so none is applied here.
- Consulted `prompts/coder_targets/triton_ascend.md`; the profile notes "Reusing
  output buffers requires an explicit Host Plan with cache keys, invalidation,
  device/stream behavior, and concurrency assumptions", which this decision
  supplies. The change does not touch the Triton kernel, so no new primitive
  support is required; `num_warps=1` and direct launch remain unchanged.
- Consulted `references/bottleneck-judgment.md`; device_ratio 0.108 (< 20%)
  classifies the candidate as host-bound, so the next intervention targets a
  compressible host component (per-forward allocation) rather than device work.

## Rationale and Evidence

Round 001 accepted a single fused Triton-Ascend kernel whose device time is
34.634 us/call, only 10.8% of the 0.321620 ms benchmark wall (device_ratio
0.108). The candidate is therefore host-bound: the remaining ~287 us/call is host
dispatch. Report 001's `evidence_for_next_round` names two compressible host
costs explicitly: (1) the candidate still allocates two output tensors
(`torch.empty` for `weights[83,8] fp32` and `ids[83,8] int32`) per forward call,
and (2) allocator lifecycle was intentionally deferred. The measured host
breakdown records output tensor allocation around 18.8 us per call, with two
allocations per forward (approximately 37 us combined, ~11.7% of wall).

Caching the two output buffers on the ModelNew instance removes those per-call
allocations. On the fixed `T=83,E=256,topk=8` regime the cache key
(tokens, topk, dtypes, device) never changes across the 100 benchmark repeats,
so every forward after the first reuses the same buffers. Removing ~37 us of
allocation from a 321.6 us wall is expected to yield on the order of 8-11%
improvement, clearing the 5% adoption threshold with a single attributable host
change. The kernel, its grid, dataflow, and numerical semantics are untouched, so
correctness risk is confined to buffer lifecycle, which the Host Plan fully
specifies.
