# Decision 003

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"003","reference_implementation":"triton_fused_moe_002.py","reference_report":"rounds/report_002.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"host","change_family":"allocation-reuse"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"reuse a single preallocated output buffer across compatible forward calls on the ModelNew instance instead of allocating a fresh torch.empty_like(hidden_states) tensor every call, removing the per-call output allocation from the dominant host time","allowed_changes":["ModelNew.forward output allocation path","output buffer cache on ModelNew instance"],"invariants":["ModelNew public constructor and forward contract","output shape [83,128] and fp16 dtype","softmax+topk+renormalize routing and weighted top-k reduce semantics unchanged","caller-selected NPU device and current stream preserved","no silent cross-instance or concurrent sharing"],"expected_wall_improvement_pct":10.0}
```

## Unified Sketch

N/A: host-only change

## Host Plan

```json
{"applicability":"required","affected_scope":["ModelNew.forward","output buffer cache on ModelNew instance"],"state_owner":"ModelNew instance","lifetime":"model lifetime (allocated lazily on first forward, retained until model destruction)","allocation_reuse":"reuse the cached output buffer only when num_tokens, hidden_size, dtype, and device all match the cache key; otherwise allocate a fresh buffer on the caller device and replace the cache","cache_key":["num_tokens","hidden_size","dtype","device"],"invalidation":"replace the cached buffer when any cache-key component changes; the buffer is overwritten by the kernel each forward and never returned by reference beyond the next forward","concurrency":"one ModelNew instance is not shared across concurrent forwards; the cache is per-instance and single-stream with no global or class-level cache","device_stream_behavior":"buffer lives on the caller-selected NPU device; the current stream and caller-selected device are preserved; no explicit stream or device context is created","unchanged_behavior":["returned output shape [83,128]","returned fp16 dtype","numerical semantics (same kernel, same inputs)","kernel launch grid and dataflow","public constructor and forward signature"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-003","intervention":"reuse a single preallocated output buffer across compatible forward calls on the ModelNew instance instead of allocating a fresh torch.empty_like(hidden_states) tensor every call, removing the per-call output allocation from the dominant host time","expected_causal_chain":["the per-call torch.empty_like output allocation disappears from the forward path","host-side allocation overhead per call decreases","benchmark wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"output_allocations_per_call","expectation":"decrease toward 0"},{"name":"host_us_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"unchanged (~26.7 us, same single kernel plus 2 casts)"}],"guardrails":["correctness:pass","output dtype and shape unchanged","numerical semantics unchanged","device_us_per_call unchanged","caller-selected device and current stream preserved"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. The four recorded failures (winner tree, sort-32+sort-64 selection network, dynamic `tl.gather` compaction, cumsum compaction) are all MLU590 device-side top-k selection evidence. None matches a host-side output-allocation-reuse change on Ascend910B4, so none is applied here.
- Consulted `references/invariants.md` Buffer/Device/Stream Lifecycle. The Host Plan names `state_owner`, `lifetime`, `allocation_reuse`, `cache_key`, `invalidation`, `concurrency`, `device_stream_behavior`, and `unchanged_behavior`, and limits reuse to exact shape/dtype/device matches, satisfying the cache-key and lifecycle requirements and forbidding implicit global or class-level caches.
- Consulted `prompts/coder_targets/triton_ascend.md`. This is a host-only change using no new Triton primitive; the profile's note "Reusing output buffers requires an explicit Host Plan with cache keys, invalidation, device/stream behavior, and concurrency assumptions" is satisfied here. `num_warps=1` and direct launch remain unchanged; `fast_libentry`/stream/context stay out of scope (Unknown on Ascend).
- Consulted `references/bottleneck-judgment.md`. device_ratio 0.072 (< 20%) classifies the candidate as host-bound, so the intervention targets a compressible host component (per-forward allocation), not device work. This is the same classification and lever that the groupedtopk-ascend and flexattention-ascend campaigns both used in their Round 2 output-alloc-reuse decisions (+18.21% and +14.71% respectively on this identical runtime).

## Rationale and Evidence

After Round 2 (report_002), the candidate runs at 0.368980 ms wall with 26.678 us device across 3 kernels (1 fused Triton kernel ~22 us plus 2 fp16 `w1/w2.to(dtype)` casts ~4.8 us). device_ratio is 0.072: ~342 us of the ~369 us wall is host-side launch, dispatch, allocation, and harness synchronization. The fused kernel is already at its structural floor (~22 us) and the only remaining device kernels are the two casts, so the compressible headroom is host-side.

The single most evidence-backed host lever is output buffer reuse: `forward` currently executes `out = torch.empty_like(hidden_states)` every call. On this identical Ascend910B4 runtime, the two sibling campaigns each removed their per-call output allocations and gained +18.21% (groupedtopk Round 2) and +14.71% (flexattention Round 2) — both clearly clearing the 5% threshold with a single attributable host-only change. This campaign's output allocation is the same class of compressible, per-forward host cost (the NPU caching allocator's host-side `empty` bookkeeping on a fixed `[83,128]` fp16 tensor).

Why this lever and not the others: (c) device-side changes are ruled out by the flexattention Round 3 negative result (tl.dot Cube halved device time but regressed wall -8.34% via a +55us host penalty), and device is only 7% of wall here. (b) the two fp16 casts are a separate mechanism (~4.8us device plus 2 host launches) and moving `w1/w2.to(dtype)` into `__init__` is correctness-delicate because the harness calls `load_state_dict` after construction; bundling it would break one-intervention attribution, so it is deferred. (d) abort is premature: unlike the two sibling campaigns that aborted only after exhausting allocation-reuse, this campaign has not yet applied the proven output-alloc-reuse lever.

The intervention is falsifiable and low-risk: `output_allocations_per_call` must drop toward 0 and `host_us_per_call` must decrease while `device_us_per_call` stays ~26.7 us and correctness passes. The cache key (num_tokens, hidden_size, dtype, device) is constant across the 100 benchmark repeats, so every forward after the first reuses the buffer; the harness compares v0/v1 outputs immediately after each forward, before any subsequent forward could overwrite the reused buffer. If the NPU caching allocator already makes `torch.empty_like` effectively free (so allocation is not a measurable host chunk), the wall gain falls below 5% and the hypothesis is falsified — which the targeted observable captures. Expected improvement is set conservatively at 10% (below the 14-18% priors, accounting for a single small output tensor).
