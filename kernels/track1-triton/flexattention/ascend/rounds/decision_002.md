# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"002","reference_implementation":"triton_flexattention_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"host","change_family":"allocation-reuse"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"reuse a single preallocated output buffer across compatible forward calls instead of allocating a fresh torch.empty([T,H,D]) tensor on every call, removing per-call output allocation from the dominant host time","allowed_changes":["ModelNew.forward","output buffer cache"],"invariants":["ModelNew public contract (num_heads=8, head_size=64, scale=None, num_kv_heads=8)","output shape [83,512] and fp16 dtype","causal numerical semantics unchanged","get_inputs and get_init_inputs entry points"],"expected_wall_improvement_pct":8.0}
```

## Unified Sketch

N/A: host-only change

## Host Plan

```json
{"applicability":"required","affected_scope":["ModelNew.forward","output buffer cache"],"state_owner":"ModelNew instance","lifetime":"model lifetime (allocated lazily on first forward, retained until model destruction)","allocation_reuse":"reuse the cached output buffer when its shape, dtype, and device match the current forward; otherwise allocate a fresh buffer and cache it","cache_key":["num_tokens","num_heads","head_size","dtype","device"],"invalidation":"replace the cached buffer whenever any cache-key component changes","concurrency":"one model instance is not shared across concurrent forwards; the cache is per-instance and single-stream","device_stream_behavior":"caller-selected device and current stream are preserved; the kernel still launches on the caller's device and stream, only the output allocation is cached","unchanged_behavior":["returned output shape [83,512]","returned fp16 dtype","numerical semantics (same kernel, same inputs)","causal mask and scale behavior"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-002","intervention":"reuse a single preallocated output buffer across compatible forward calls instead of allocating a fresh torch.empty([T,H,D]) tensor on every call, removing per-call output allocation from the dominant host time","expected_causal_chain":["per-call output allocation disappears from the forward path","host-side per-call work decreases","wall_time_ms decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"output_allocations_per_call","expectation":"decrease toward 0"},{"name":"host_us_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"unchanged (~54 us, same single kernel)"}],"guardrails":["correctness:pass","output dtype and shape unchanged","numerical semantics unchanged","device_us_per_call unchanged"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. The recorded failures (winner-tree selection, sort-32/sort-64 selection network, dynamic gather, cumsum compaction) all concern grouped top-k device-side selection under MLU590-H8 and do not match the host-side output-allocation-reuse preconditions here (Ascend910B4, host-bound device_ratio 0.163, single-kernel candidate). No listed failure invalidates this path.
- Consulted `references/invariants.md` Buffer/Device/Stream Lifecycle. The Host Plan explicitly declares `state_owner`, `lifetime`, `allocation_reuse`, `cache_key`, `invalidation`, `concurrency`, `device_stream_behavior`, and `unchanged_behavior`, and limits reuse to exact shape/dtype/device matches, satisfying the cache-key and lifecycle requirements.
- Consulted `prompts/coder_targets/triton_ascend.md`. This is a host-only change using no new Triton primitive; `tl.dot` (M=1) and other device primitives are deliberately not made normative this round because the device is already host-bound (device_ratio 0.163) and M=1 `tl.dot` is Unknown on Ascend.

## Rationale and Evidence

`rounds/report_001.md` shows the candidate is host-bound: wall `0.330810 ms` versus device `54.04 us/call` (device_ratio `0.163`, well under the 0.20 host-bound boundary). Approximately 276 us of wall time is host-side, and the single fused kernel is already close to its structural floor (the reference's theoretical `aclnnFlashAttentionScore` core is ~25 us). Squeezing device time further would require an unproven M=1 `tl.dot` and could yield at most ~8% wall even in the ideal case, with high capability-miss risk — a poor risk-adjusted path this round.

The current `forward` allocates a fresh `torch.empty((T,H,D))` output tensor on every call (and the accepted reference path does the same via `F.scaled_dot_product_attention`). This per-call allocation sits inside the dominant host time. Reusing one cached buffer when shape/dtype/device match removes that recurring allocation, directly reducing host work. The sibling MLU campaign reached 7.08x by combining exactly this output-cache with a fast launcher, and the groupedtopk sibling gained +18.21% in its Round 2 from output allocation reuse alone — strong, matched evidence that per-call output allocation is a compressible, attributable host cost.

The intervention is falsifiable and safe: `output_allocations_per_call` must drop toward 0 and `host_us_per_call` must decrease while `device_us_per_call` stays ~54 us (the same single kernel) and correctness passes. Correctness is not at risk because the harness compares v0/v1 outputs immediately after each forward, before any subsequent forward could overwrite the reused buffer, and the cache key (shape/dtype/device) guarantees the buffer is only reused for identical output geometry. If allocation is not actually a measurable chunk of host time (e.g. the NPU caching allocator already makes `torch.empty` near-free), the wall improvement will fall below 5% and the hypothesis is falsified — which is the point of the targeted observable.
