# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"002","reference_implementation":"triton_grouped_topk_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"gcu","target_profile":"triton_gcu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"host","change_family":"allocation-reuse"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"replace the two per-forward output tensor allocations with a model-instance-owned compatibility-keyed pool that leases one output-buffer pair per live forward","allowed_changes":["ModelNew private output-pool state initialized in __init__","ModelNew.forward output-buffer acquisition and release bookkeeping","model-local host locking and lease metadata required to preserve output lifetime and concurrent-forward semantics"],"invariants":["ModelNew constructor and forward public contract","grouped top-k numerical and tie semantics","output tuple structure, shapes, contiguous layout, and dtypes","caller-selected GCU device and current stream","Triton kernel body, launch grid, launch count, constexprs, and num_warps=1","outputs and aliases retained by a caller are never overwritten by a later or concurrent forward","separate ModelNew instances never share output-pool state","reference_triton_grouped_topk_001.py remains byte-equivalent to the canonical source except for the required ModelNew-to-Model class rename"],"expected_wall_improvement_pct":6.0}
```

## Unified Sketch

N/A: host-only change

## Host Plan

```json
{"applicability":"required","affected_scope":["ModelNew.__init__ private output-pool state","ModelNew.forward output-buffer acquisition and return path","model-local lease and lock bookkeeping"],"state_owner":"Each ModelNew instance exclusively owns its private pool and synchronization metadata; no module-global, process-global, or cross-instance cache is allowed.","lifetime":"The pool lives from ModelNew construction through model destruction. A reserved pair remains leased for the complete externally observable lifetime of both returned tensors and any storage-sharing aliases; a pair is idle only when the implementation can prove that no caller-visible tensor or alias still owns or can observe its storage.","allocation_reuse":"For a compatible call, atomically reserve an idle pair and launch the unchanged kernel into it. If every compatible pair is leased or in flight, or safe idleness cannot be proved, allocate a fresh weights/ids pair and add it to that instance's pool; never serialize callers or overwrite live storage merely to force reuse.","cache_key":["gating_output device type and device index","caller current GCU stream identity","output shape tokens x topk","weights dtype torch.float32","ids dtype torch.int32","contiguous output layout and strides"],"invalidation":"A cache-key mismatch must not reuse an existing pair. Device or current-stream changes select a distinct key. Idle incompatible entries may be discarded, but live leases remain valid until release. If standard GCU tensor/storage lifetime or current-stream identity cannot be proven safely, the call must allocate fresh outputs and Coder must report capability-miss rather than use an unsafe cache.","concurrency":"Protect only pool metadata with a ModelNew-local lock and reserve one distinct pair per in-flight forward. Concurrent forwards on one instance must never share a live pair; a pool miss allocates instead of serializing kernel execution. Separate model instances have disjoint pools, and retained outputs from earlier calls remain stable.","device_stream_behavior":"Allocate and launch on gating_output.device and the caller's current GCU stream. Reuse is permitted only within an exact device-and-stream cache key and only after the prior lease is safely idle, preserving same-stream ordering. Do not enter another device context, switch streams, insert cross-stream waits, call torch.gcu.synchronize(), or add any device operation or kernel launch.","unchanged_behavior":["ModelNew public constructor and forward signature","returned weights and ids tuple structure","weights shape [tokens,topk], fp32 dtype, contiguous layout, and GCU device","ids shape [tokens,topk], int32 dtype, contiguous layout, and GCU device","grouped softmax, group selection, expert top-k, renormalization, scaling, ordering, and tolerance behavior","the accepted Triton kernel body and one direct Triton-GCU launch per forward","caller-selected device, current stream, concurrent-forward behavior, and lifetime of previously returned outputs","reference adapter content other than the ModelNew-to-Model class rename"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-002","intervention":"replace the two per-forward output tensor allocations with a model-instance-owned compatibility-keyed pool that leases one output-buffer pair per live forward","expected_causal_chain":["after warmup, compatible sequential forwards reserve idle output pairs instead of executing two torch.empty allocations","per-forward host allocation work decreases without adding a device launch, synchronization, or stream transition","unrounded paired benchmark wall time decreases by at least five percent","correctness, retained-output lifetime, stream, device, and concurrent-forward guardrails remain satisfied"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"output_allocations_per_call","expectation":"decrease from two to zero after warmup for compatible sequential forwards"},{"name":"live_output_storage_conflicts","expectation":"remain zero when prior outputs or storage-sharing aliases are retained and when forwards overlap on one model instance"},{"name":"runtime_launch_count_per_call","expectation":"remain 1.0 with no added synchronization or device operation"}],"guardrails":["correctness:pass","public constructor, forward signature, and returned tuple structure unchanged","output shapes, contiguous layout, dtypes, and selected GCU device unchanged","caller current GCU stream preserved with no explicit synchronization or device-context switch","accepted Triton kernel body, grid, constexprs, and num_warps=1 unchanged","previously returned outputs and aliases remain unchanged after later compatible forwards","overlapping forwards on one model instance use distinct live storage and produce correct independent results","separate model instances do not share output storage","reference_triton_grouped_topk_001.py differs from canonical only by class ModelNew renamed to Model","GCU device duration remains unavailable and runtime launch duration is not relabeled as device kernel time"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`; its recorded failures concern MLU
  selection dataflows, not this host-only allocation lifecycle. This decision
  does not reinterpret those backend-specific failures as GCU evidence.
- Consulted `prompts/coder_targets/triton_gcu.md`; output caching, current-stream
  behavior, and concurrent model use are explicitly unproven. The Host Plan
  therefore requires exact device/stream keys, per-instance ownership, live-slot
  leases, and a capability-miss when safe lifetime or stream identity cannot be
  established.
- Consulted `references/invariants.md`; a single cached pair reused on every call
  is forbidden because it can overwrite retained outputs and race concurrent
  forwards. No global cache, device switch, stream switch, explicit synchronize,
  or cross-instance sharing is allowed.
- Consulted `references/bottleneck-judgment.md`; the GCU runtime-launch duration
  is diagnostic only. It is not device time and is not used to assert a host
  ratio or device ratio for this decision.

## Rationale and Evidence

The canonical source `triton_grouped_topk_001.py` executes two `torch.empty`
output allocations in every forward before its single direct Triton-GCU launch.
The harness reference fixture `reference_triton_grouped_topk_001.py` has SHA-256
`800ec0080e66589f6dfcf3a71ee79f08e01be68f145b4cb3c6c6b50dd7c03027` and is
byte-equivalent to canonical after the required `ModelNew` to `Model` class
rename; it is a verifier adapter and does not replace the canonical pointer.
The accepted `rounds/report_001.md` records a wall median of `0.273881 ms`, one
runtime launch per call, unavailable GCU device duration, and identifies the two
output allocations as an intentionally untested next-round lifecycle mechanism.
Those facts support a falsifiable allocation-reuse attempt, but they do not prove
a measured host/device ratio or a six-percent gain. The required allocation
count probe, paired wall timing, retained-output checks, alias checks, and
concurrent-forward checks decide the hypothesis. The kernel and runtime-launch
path remain outside the change boundary so any valid wall-time change is
attributable to output-buffer lifecycle.
