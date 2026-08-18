# Decision 004

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"004","reference_implementation":"triton_grouped_topk_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"maca","target_profile":"triton_maca","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"host","change_family":"fast-path-dispatch-specialization"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"specialize only the fixed fast-path Python dispatch by removing hidden_states metadata eligibility checks that are semantically unnecessary because hidden_states contributes only the preserved leading-token assertion, comparing gating_output.shape directly without tuple materialization, and reading gating_output.device once for the CUDA check and both unchanged fresh output allocations","allowed_changes":["candidate-only ModelNew.forward fixed-fast-path predicate","retain the leading-token assertion, replace both tuple(shape) equality expressions with direct gating_output.shape == (83, 256), and remove hidden_states width, dtype, contiguity, and device-equality eligibility checks","bind gating_device = gating_output.device once per forward and reuse it for gating_device.type == 'cuda' and the device argument of both unchanged torch.empty calls"],"invariants":["canonical/reference/base/harness/project/team-state files are not modified","the complete Triton kernel, direct launch, grid (83,), argument order, T=83, BLOCK_E=256, and num_warps=1 are byte-equivalent to Round 001","constructor/signature, mutable constructor-attribute comparisons, exact grad predicate, canonical PyTorch fallback body, get_inputs, and get_init_inputs remain unchanged","the fast path retains exact gating shape, dtype, contiguity, CUDA-device, fixed-constructor, and grad-safety requirements","the initial token-count assertion remains first and exact; removing other hidden metadata eligibility checks is valid because base.py and the accepted kernel never read hidden_states afterward","exactly two independent fresh torch.empty outputs remain; shared backing, cache, pool, reuse, aliasing, and persistent mutable state are forbidden","output tuple, values, exact IDs, shapes, dtypes, contiguity, gating-device placement, input non-mutation, lifetime, concurrency, and cross-instance isolation remain compatible with base.py","every call failing a retained fast-path condition executes the byte-equivalent canonical fallback","caller-selected gating device/current stream are preserved; no synchronization, context, launcher, kernel, or allocation-family change is introduced"],"expected_wall_improvement_pct":6.0}
```

## Unified Sketch

N/A: host-only change

## Host Plan

```json
{"applicability":"required","affected_scope":["ModelNew.forward fixed-fast-path predicate","invocation-local gating_device binding and existing torch.empty device arguments"],"state_owner":"No state is introduced; gating_device and fast_path are invocation-local values, and each returned tensor owns one fresh allocation exactly as Round 001.","lifetime":"The local device object exists only during forward. Both independently allocated outputs retain ordinary per-call lifetime across later calls and model instances.","allocation_reuse":"None. Keep exactly two independent fresh torch.empty allocations per fixed call; do not share backing, cache, pool, recycle, alias, or reuse outputs.","cache_key":["not-applicable:no cache, pool, or reusable buffer exists"],"invalidation":"Not applicable because no state is retained. Recompute every mutable constructor predicate and allocate fresh outputs on every invocation.","concurrency":"Concurrent/reentrant forwards and distinct model instances share no candidate-owned mutable state; invocation locals and separate output allocations preserve isolation.","device_stream_behavior":"Read gating_output.device once, require type cuda, allocate both outputs on that exact device, and use the unchanged direct launch on the caller current stream without device or stream switching or synchronization.","unchanged_behavior":["leading token-count assertion and exception behavior","retained gating metadata, constructor-attribute, and grad-safety predicates","byte-equivalent canonical fallback outside the specialized fixed guard","two independent fresh output allocations and non-aliasing/lifetime behavior","kernel, direct launch, grid, T, BLOCK_E, num_warps, device, and current stream","public entry points, output contract, exact IDs, tolerated weights, and input non-mutation"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-004","intervention":"specialize only the fixed fast-path Python dispatch by removing hidden_states metadata eligibility checks that are semantically unnecessary because hidden_states contributes only the preserved leading-token assertion, comparing gating_output.shape directly without tuple materialization, and reading gating_output.device once for the CUDA check and both unchanged fresh output allocations","expected_causal_chain":["on every fixed benchmark forward, remove two Python tuple materializations, four hidden_states metadata eligibility queries, and four of five source-level tensor device-property reads while preserving the accepted kernel path","the separately scoped forward CPU duration decreases by at least 4.1 us/call with two allocations, one direct launch, and one device kernel unchanged","unrounded median benchmark wall_time decreases by at least 5% against accepted Round 001"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"fast_guard_shape_tuple_materializations_per_call","expectation":"decrease from exactly 2 to exactly 0 by source audit"},{"name":"fast_guard_hidden_metadata_eligibility_queries_per_call","expectation":"decrease from exactly 4 (shape, dtype, is_contiguous, device) to exactly 0, excluding the preserved token assertion and conditional requires_grad read"},{"name":"tensor_device_property_reads_per_fixed_call_source","expectation":"decrease from exactly 5 across canonical guard and allocations (one hidden, four gating) to exactly 1 local gating_device binding"},{"name":"forward_cpu_scope_inclusive_us_per_call","expectation":"decrease by at least 4.1 us/call from durable accepted 41.58952 us/call in comparable separate scopes; inclusive diagnostic only"},{"name":"newly_admitted_hidden_metadata_semantic_parity","expectation":"with eligible gating under no_grad, targeted hidden tensors varying nonleading width, dtype, contiguity, and device match base.py values/exact IDs, preserve token assertion, and demonstrably take fast path"},{"name":"retained_guard_and_fallback_equivalence","expectation":"source audit proves every gating/config/grad condition retained and fallback byte-equivalent; targeted nonfast cases match base including sigmoid, unsupported scoring error, grad-requiring gating input, and token mismatch"},{"name":"aten_empty_count_per_call","expectation":"remain exactly 2.0 with independent fresh storages and no Round 002 backing/view"},{"name":"candidate_kernel_count_per_call","expectation":"remain exactly 1.0"},{"name":"candidate_device_us_per_call","expectation":"no material regression; at most 1.05 times concurrently scoped accepted reference device us/call"},{"name":"mcModuleLaunchKernel_count_per_call","expectation":"remain exactly 1.0 with unchanged direct launch"},{"name":"kernel_launch_byte_equivalence","expectation":"kernel and launch grid/arguments/T/BLOCK_E/num_warps are byte-equivalent to canonical"},{"name":"reference_adapter_class_rename_only","expectation":"reference adapter SHA256 is 70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9 and only renames ModelNew to Model"}],"guardrails":["correctness:pass","unrounded median wall improvement is at least 5% against accepted Round 001 under measurement fingerprint 3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809","values pass frozen tolerance and IDs are exact on seed, existing tie probes, and newly admitted hidden-metadata cases","initial token assertion, retained gating/config/grad guard, and byte-equivalent fallback preserve public behavior and exceptions","kernel, launch, one-kernel/one-launch count, device time, and current-stream behavior do not regress","two independent fresh outputs remain; no cache, pool, reuse, shared backing, dtype view, alias, model state, or global mutable state","retained outputs, concurrent/reentrant forwards, and separate model instances preserve independent storage/lifetime","Round 002 allocation coalescing, Round 003 combined reduction, fast_libentry, and unrelated kernel/allocation/launcher changes are excluded","CPU profiler durations are inclusive and never added, subtracted, or used to reconstruct wall","immutable sources, profiler settings, and measurement fingerprint remain unchanged"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- The accepted 41.58952 us/call CPU scope is inclusive and the benchmark wall
  is 68.280 us/call. Source counts and CPU scope establish mechanism only;
  unrounded wall controls adoption.
- Do not precompute a constructor flag. The public routing attributes are
  mutable and must be checked on every call. Retain the grad predicate and
  leading token assertion exactly.
- Broader hidden-metadata eligibility follows source semantics: base.py reads
  only hidden_states.size(0), then computes entirely from gating_output.
  Targeted cases must prove newly admitted calls take fast path and match base.
- The single gating device object is invocation-local. It must not become model
  state, a device cache/context, or a stream change. Outputs stay separately
  and freshly allocated.
- Round 002 safety passed but wall regressed 13.711567434852972%; Round 003
  capability/correctness passed but wall improved only 0.04903708987159917%.
  Neither family is included, and neither unprofiled result gets a causal story.
- fast_libentry is unsupported. Direct launch, num_warps=1, and the complete
  accepted kernel are frozen.
- The anti-pattern catalog concerns kernel selection. This host-only change
  introduces no winner tree, sort, gather, compaction, or backend assumption.

## Rationale and Evidence

Round 001 remains canonical at 68.280 us/call wall, 41.58952 us/call inclusive
forward CPU scope, and one 10.7442822265625 us/call device kernel. A 5% gain is
3.414 us/call; the 6% hypothesis is 4.0968 us/call. On every fixed call the
canonical predicate visibly constructs two tuples, performs four unnecessary
hidden metadata eligibility queries, and reads tensor device properties five
times across the guard and allocations.

Both base.py and the accepted fallback use hidden_states only for the leading
token assertion; scores, IDs, weights, and output placement derive entirely
from gating_output. Removing only those eligibility queries, comparing gating
shape directly, and reusing one local device value is therefore auditable while
two allocations, launch, kernel, mutable configuration checks, grad behavior,
fallback, device, stream, and lifetime remain unchanged.

The expected saving is falsifiable: exact source counts must change, the
comparable CPU scope must fall by at least 4.1 us/call, and authoritative wall
must improve by at least 5%. Otherwise Round 004 is no-improvement and reaches
the configured three-miss stop.
