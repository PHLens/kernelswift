# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"002","reference_implementation":"triton_grouped_topk_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"maca","target_profile":"triton_maca","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"host","change_family":"fresh-allocation-coalescing"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"replace the fixed fast path's two per-forward torch.empty calls with one fresh int32 backing allocation and return non-overlapping contiguous fp32/int32 views from its two halves, without cache, reuse, pooling, or kernel changes","allowed_changes":["candidate-only ModelNew.forward fixed-fast-path output allocation and view construction","allocate one fresh flat int32 tensor of length 2*83*8 on gating_output.device for every fixed-fast-path call","construct weights as backing[:83*8].view(torch.float32).view(83,8) and IDs as backing[83*8:].view(83,8) before the unchanged launch"],"invariants":["triton_grouped_topk_001.py, reference_triton_grouped_topk_001.py, base.py, auto_bench.py, project.md, and team-state.md are not modified","the complete _grouped_topk_fixed_kernel definition is byte-equivalent to triton_grouped_topk_001.py","the launch grid (83,), argument order, T=83, BLOCK_E=256, and num_warps=1 are byte-equivalent to Round 001","ModelNew constructor, forward signature, exact fixed-fast-path guard, canonical PyTorch fallback, get_inputs, and get_init_inputs remain unchanged","exactly one new backing allocation occurs on every fixed-fast-path forward; no backing, output, cache, pool, or mutable state survives in the model or module","weights and IDs share one backing storage only within a call, occupy disjoint byte intervals [0,2656) and [2656,5312), and are each contiguous [83,8] views with fp32 and int32 dtype respectively","each returned view retains the backing lifetime; while prior outputs remain live, every later forward and every other model instance owns a distinct backing storage","mutating elements through either returned tensor cannot modify the other tensor because their byte spans do not overlap","caller-selected CUDA-compatible device and current stream are preserved; no device context, synchronization, launcher, or kernel change is introduced","MACA CUDA-tensor dtype-view support is an explicit capability gate; unsupported view construction is capability-miss and must not be hidden by changing the algorithm or silently restoring two allocations on the fixed benchmark"],"expected_wall_improvement_pct":6.0}
```

## Unified Sketch

N/A: host-only change

## Host Plan

```json
{"applicability":"required","affected_scope":["ModelNew.forward fixed-fast-path output allocation","returned topk_weights/topk_ids view construction"],"state_owner":"Each forward invocation owns one newly allocated local flat int32 backing tensor; the two returned tensor views retain that backing storage without storing it on ModelNew or in module-global state.","lifetime":"Create the backing at the start of every fixed-fast-path call after the existing guard; it remains alive as long as either returned view is live and becomes reclaimable only after both views are released.","allocation_reuse":"None. Allocate exactly one fresh 1328-element int32 backing tensor per fixed-fast-path call; do not cache, reuse, pool, resize, or retain it across calls or model instances.","cache_key":["not-applicable:no cache, reuse, or pool exists"],"invalidation":"Not applicable because no backing is retained after the returned views release it; every invocation unconditionally creates a new backing.","aliasing":"The outputs intentionally share the same backing storage within one call, but weights use backing elements [0,664) reinterpreted as fp32 and IDs use backing elements [664,1328) as int32. Their data byte spans are disjoint and neither tensor overlaps the other.","non_overlap":"Both dtypes are four bytes on this contract. Weights occupy backing bytes [0,2656); IDs occupy [2656,5312). Each view has shape [83,8], zero relative storage offset within its half, standard contiguous strides, and no writable element aliases the other output.","concurrency":"Concurrent or reentrant forwards, retained outputs from earlier calls, and separate ModelNew instances are safe because every invocation creates a distinct backing and no mutable allocation state is shared. Verification keeps outputs from call A live while call B runs and requires different backing storage identities.","device_stream_behavior":"Allocate the fresh backing on gating_output.device through torch.empty and preserve the caller's current stream. Keep the direct Triton launch, device selection behavior, launch order, and absence of explicit synchronization or device-context changes exactly as Round 001.","fallback":"All calls outside the unchanged fixed-fast-path guard execute the unchanged canonical PyTorch fallback. The fixed fast path has no two-allocation fallback: failure of .view(torch.float32) on the matched MACA CUDA tensor is capability-miss, not permission to change the kernel, algorithm, or allocation intervention.","capability_gate":"The matched runtime must permit a contiguous int32 CUDA-compatible tensor slice with aligned offset and equal element width to be viewed as float32 without copy, allocation, or kernel launch. The actual harness and targeted storage probe must establish this; support is not inferred from generic CUDA behavior.","unchanged_behavior":["public constructor and forward contract","fixed-fast-path guard and fallback semantics","output tuple order, shape, dtype, contiguity, values, IDs, device, and input non-mutation","_grouped_topk_fixed_kernel body, grid, arguments, T, BLOCK_E, and num_warps","caller-selected device/current stream and per-call output lifetime semantics"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-002","intervention":"replace the fixed fast path's two per-forward torch.empty calls with one fresh int32 backing allocation and return non-overlapping contiguous fp32/int32 views from its two halves, without cache, reuse, pooling, or kernel changes","expected_causal_chain":["the fixed fast path performs one fresh torch.empty instead of two while still returning two typed contiguous views","aten::empty count and its inclusive CPU-event duration decrease without adding a device kernel or launch","the accepted Triton kernel, launch count, device time, output semantics, and independent per-call lifetimes remain unchanged","unrounded median benchmark wall_time decreases by at least 5% against the accepted Round 001 implementation"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"aten_empty_count_per_call","expectation":"decrease from 2.0 to exactly 1.0 in the fixed-fast-path CPU scope"},{"name":"aten_empty_inclusive_us_per_call","expectation":"decrease from the accepted 10.03988 us/call; report as an inclusive diagnostic without adding it to or subtracting it from other events or wall time"},{"name":"triton_kernel_count_per_call","expectation":"remain exactly 1.0"},{"name":"candidate_device_us_per_call","expectation":"no material regression: no more than 1.05 times the concurrently profiled accepted reference device time, whose durable value is 10.7442822265625 us/call"},{"name":"mcModuleLaunchKernel_count_per_call","expectation":"remain exactly 1.0"},{"name":"mcModuleLaunchKernel_inclusive_us_per_call","expectation":"no material regression relative to the accepted 4.88562 us/call inclusive diagnostic"},{"name":"kernel_body_grid_num_warps_byte_equivalence","expectation":"_grouped_topk_fixed_kernel source and the (83,) launch with T=83, BLOCK_E=256, num_warps=1 are byte-equivalent to triton_grouped_topk_001.py"},{"name":"reference_adapter_class_rename_only","expectation":"reference_triton_grouped_topk_001.py SHA256 is 70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9 and its only byte diff from canonical is ModelNew renamed to Model"},{"name":"maca_cuda_dtype_view_capability","expectation":"the one-allocation int32-to-fp32 view path executes without copy, extra allocation, or device kernel; otherwise capability-miss"},{"name":"output_storage_spans_disjoint","expectation":"weights and IDs share one backing identity but their computed data-pointer byte intervals do not overlap and each tensor is contiguous [83,8] with fp32/int32 dtype"},{"name":"cross_call_live_outputs_distinct_backing","expectation":"with outputs from call A retained, call B outputs use a different backing storage; separate model instances likewise do not share backing"},{"name":"output_alias_mutation_isolation","expectation":"in-bounds mutations through one output view do not alter any element of the other output view"}],"guardrails":["correctness:pass","unrounded median wall improvement is at least 5% against triton_grouped_topk_001.py through the verified class-rename reference under measurement fingerprint 3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809","output values and exact IDs pass the frozen harness tolerance/tie contract","output tuple order, [83,8] shapes, fp32/int32 dtypes, contiguity, device, and input non-mutation are unchanged","the fixed fast path performs exactly one fresh allocation per call and introduces no cache, reuse, pool, global state, model state, or cross-call storage sharing","same-call output byte spans are disjoint and retained output lifetime remains valid after forward returns and after later forwards execute","kernel body, grid, arguments, T, BLOCK_E, num_warps, direct launch, device kernel count, and launch count remain Round-001-equivalent","canonical PyTorch fallback and full public constructor/forward behavior remain unchanged outside the fixed guard","caller-selected device and current stream are preserved","CPU profiler event durations are treated as inclusive and potentially nested/overlapping; they are not summed, subtracted, or used to reconstruct benchmark wall time","base.py, auto_bench.py, project.md, team-state.md, canonical/reference source files, profiler settings, and measurement fingerprint remain unchanged"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- `Tensor.view(dtype)` on the matched MACA CUDA-compatible tensor is not
  established by the target profile. This is an explicit compile/runtime
  capability gate. Coder and Verifier must not infer support from NVIDIA CUDA;
  failure is `capability-miss`, not permission to alter the kernel or restore
  two fixed-fast-path allocations.
- Same-call outputs intentionally share a storage object, which is distinct from
  overlapping tensor elements. Verification must compute data-pointer byte
  intervals, require strict non-overlap, confirm both views are contiguous, and
  keep prior outputs live while checking that later calls use another backing.
- The backing must be a local fresh allocation every call. Model attributes,
  module globals, allocator-side user caches, explicit pools, lazy reuse, and
  output recycling are outside the decision and would be a major deviation.
- The accepted trace reports `41.58952 us/call` for the inclusive CPU scope,
  `10.03988 us/call` for two inclusive `aten::empty` events, and
  `4.88562 us/call` for one inclusive launch event. These events may nest or
  overlap; they cannot be added together, subtracted, or used to reconstruct
  the authoritative `68.280 us/call` wall time.
- `reference_triton_grouped_topk_001.py` is a harness adapter, not a new
  algorithm. Its exact SHA and one-line `ModelNew` to `Model` diff must pass
  before timing so the comparison source remains the accepted canonical logic.
- Consulted `references/anti-patterns.md`; its four entries concern kernel
  selection dataflows on an MLU runtime. This host-only round changes none of
  those mechanisms and freezes the accepted MACA kernel byte-for-byte.
- Consulted `references/invariants.md`,
  `references/bottleneck-judgment.md`, and the target profile. Targeted CPU
  attribution is diagnostic, separately scoped Level 1 kernel evidence remains
  mandatory, and only unrounded wall median controls adoption.

## Rationale and Evidence

The accepted implementation's authoritative benchmark median is
`68.280 us/call`. Its profile contains one `10.7442822265625 us` device
kernel and a `41.58952 us/call` inclusive CPU scope, so the device ratio is
about `15.74%` and the current evidence is host-bound. Inside that CPU scope,
`aten::empty` occurs twice per call with `10.03988 us/call` inclusive
duration. One launch remains at `4.88562 us/call`.

A 5% adoption gain requires `3.414 us/call`; the proposed 6% expectation is
`4.0968 us/call`. Coalescing two same-sized output allocations into one fresh
allocation targets an observed host event large enough to justify that
falsifiable expectation, while retaining the kernel and every output lifetime.
The inclusive event duration does not predict an exact saving and is not added
to any other duration; the frozen benchmark wall median alone decides whether
the hypothesis succeeds.

