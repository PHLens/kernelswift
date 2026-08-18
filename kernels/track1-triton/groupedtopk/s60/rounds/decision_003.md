# Decision 003

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"003","reference_implementation":"triton_grouped_topk_002.py","reference_report":"rounds/report_002.md","language":"triton","backend":"gcu","target_profile":"triton_gcu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"host","change_family":"host-metadata-specialization"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"cache exact-shape/device/stream-compatible block_e, epg, and launch configuration in ModelNew host metadata state without changing the Triton kernel body, grid, constexprs, num_warps, public contract, or device/stream semantics","allowed_changes":["ModelNew instance-private host metadata cache","forward-side exact-key metadata lookup and miss path","reuse of cached block_e, epg, and launch configuration"],"invariants":["ModelNew public constructor and forward contract","output shapes, dtypes, layout, and numerical semantics","Triton kernel body remains unchanged","grid remains (tokens,)","all constexpr values and their meanings remain unchanged","num_warps remains 1","caller-selected GCU device and current stream semantics remain unchanged","existing output-pool ownership and lifecycle behavior remains unchanged","no module-global or cross-instance cache"],"expected_wall_improvement_pct":5.0}
```

## Unified Sketch

N/A: host-only change

## Host Plan

```json
{"applicability":"required","affected_scope":["ModelNew.__init__ metadata-cache state","ModelNew.forward metadata lookup and miss path","host launch-argument packaging"],"state_owner":"A private metadata cache owned by the individual ModelNew instance; no module-global, process-global, or cross-instance owner is permitted.","lifetime":"Create the cache with ModelNew and release it with that instance; entries contain immutable host launch metadata only and never own output or device buffers.","allocation_reuse":"After an exact cache-key hit, reuse the cached block_e, epg, grid, constexpr-value bundle, and launch configuration; on a miss compute and insert one immutable entry, while leaving the existing output lease pool and its storage-lifetime behavior unchanged.","cache_key":["exact gating_output.shape including tokens and experts","gating_output.dtype and the output dtype/layout requirements","topk, renormalize, num_expert_group, topk_group, scoring_func, and routed_scaling_factor","gating_output.device.type and device.index","caller current GCU stream identity","launch configuration including grid=(tokens,), BLOCK_E=block_e, epg, and num_warps=1"],"invalidation":"An entry is incompatible and must not be reused when any exact shape, dtype, routing configuration, device, stream, or launch-configuration component changes; use a separate entry on a miss, do not mutate an existing entry, and discard an entry only when it is not being looked up or used.","concurrency":"Protect lookup and insertion with ModelNew instance-owned synchronization; concurrent forwards may share an immutable metadata entry after a validated hit, but must not race its initialization, and all existing output-pool lease protections remain in force. Separate ModelNew instances never share metadata state.","device_stream_behavior":"Observe the caller-selected GCU device and current stream only for compatibility-key construction, preserve both for output allocation and the unchanged direct Triton launch, and perform no synchronize, stream switch, device switch, cross-stream wait, or altered device-context operation.","unchanged_behavior":["_grouped_topk_kernel body and its dataflow","grid=(tokens,) and all constexpr values E, n_group, epg, K, KG, renorm, scaling, and BLOCK_E","num_warps=1 and the direct Triton-GCU launcher","ModelNew public contract and get_inputs/get_init_inputs","output-pool allocation, lease, alias, retained-output, and concurrent-forward behavior","output device placement, current-stream ownership, shapes, dtypes, ordering, and tolerance semantics"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-003","intervention":"cache exact-shape/device/stream-compatible block_e, epg, and launch configuration in ModelNew host metadata state without changing the Triton kernel body, grid, constexprs, num_warps, public contract, or device/stream semantics","expected_causal_chain":["an exact-key miss computes shape-derived metadata and stores one immutable launch bundle","compatible repeated forwards hit the instance-private cache and avoid repeated block_e, epg, and launch-argument construction","the unchanged one-launch kernel path and output-pool lifecycle remain intact","host launch setup decreases and authoritative wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"metadata_cache_hit_rate","expectation":"after one exact-key miss, compatible repeated forwards hit; every shape, dtype, device, stream, or routing-key change misses"},{"name":"metadata_derivation_count_per_call","expectation":"decrease from once per forward to once per new exact key and zero on compatible hits"},{"name":"host_launch_setup_us_per_call","expectation":"decrease in a same-process targeted decomposition with matching call counts"},{"name":"runtime_launch_count_per_call","expectation":"remain 1.0 for accepted reference and candidate"}],"guardrails":["correctness:pass","output shapes, dtypes, layout, and numerical semantics unchanged","ModelNew public contract unchanged","kernel body, grid, constexprs, and num_warps unchanged","exact cache-key compatibility and invalidation pass","separate ModelNew instances do not share metadata state","concurrent forwards do not race cache initialization","caller-selected device and current stream preserved","existing output-pool lifetime and alias guardrails pass"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`; its recorded MLU selection and compaction regressions do not justify a GCU kernel-body change, so this round remains host-only.
- Consulted `references/bottleneck-judgment.md`; Round 002 provides a source-backed host metadata hypothesis but no host-time attribution. Targeted same-process decomposition is required and must not be replaced by GCU runtime launch time or a fabricated device ratio.
- The GCU profile records one direct launch per call and unavailable device duration. The decision therefore preserves the launch path and requests runtime-launch count as a guardrail, not as device-time evidence.
- Cache invalidation, instance ownership, concurrency, exact shape/device/stream compatibility, and caller stream preservation are explicit in the Host Plan. An unprovable compatibility or lifetime property is a capability miss, not permission for unsafe reuse.

## Rationale and Evidence

Round 002 was accepted against the canonical `triton_grouped_topk_001.py` reference with a `9.02136875254568%` unrounded median wall improvement. Its profiler showed one runtime launch per call for both implementations, while GCU device duration remained unavailable. The accepted `triton_grouped_topk_002.py` still computes `block_e = triton.next_power_of_2(experts)`, derives `epg`, and constructs the launch argument bundle on every forward at `forward` lines 192-209. These observations identify a falsifiable host metadata specialization hypothesis, not a measured host-time claim. Round 003 must compare against `triton_grouped_topk_002.py`, preserve its output-pool lifecycle, and accept only if correctness, all lifecycle and stream guardrails, targeted metadata evidence, and the unrounded wall median all pass the contract.
