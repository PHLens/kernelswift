# Target Profile: triton_ascend

This profile records only capabilities observed on the available Ascend NPU
runtime and probe configuration. Evidence is local to the recorded runtime,
device architecture, shapes, dtypes, and compiler versions. Absence from
Supported is not evidence of support.

## Identity and Match

The decision and manifest must say `language: triton`, `backend: ascend`, and
`target_profile: triton_ascend`. Phase 0 must discover the Triton distribution
and version, the `torch_npu` package version, the active compiler/driver target
when available, and the Ascend device architecture. The recorded probe reports
an Ascend device `Ascend910B4` with `cube_core_num=20`, `vector_core_num=40`,
and `L2_cache_size=96MB`; this is evidence for that runtime only. A missing
runtime, backend mismatch, or architecture/profile mismatch is
`environment-blocked`, not a capability miss. Repository evidence does not
replace the current project's runtime fingerprint.

## Runtime and Launcher Conventions

- Import `torch_npu` before allocating NPU tensors or launching Triton kernels;
  the probe does not require a separate `triton_ascend` package import because
  the Ascend backend is registered inside `triton.backends` (the single
  registered backend on this runtime). `triton_ascend` is present only as a
  `.dist-info` metadata package; `import triton_ascend` fails, and the backend
  is reached through `triton.backends.backends['ascend']`.
- Use `device="npu"` for the selected accelerator and
  `torch_npu.npu.synchronize()` (or `torch.npu.synchronize()`) for the observed
  synchronization boundary.
- Direct Triton launch syntax `kernel[(grid,)](...)` was observed and is the
  proven launcher path for this project.
- The shared `base.py` reference writes tensors with `device="cuda"`; the harness
  `auto_bench.py` rewrites the `"cuda"` literal to the detected accelerator
  (`npu` here) before exec. Do not rely on a raw `"cuda"` device string in
  candidate `get_inputs`; use `"npu"` or derive from an input tensor's device.
- The actual harness loader must be used. GCU/MLU device strings, stream
  assumptions, and launcher behavior must not be inferred for Ascend.

## Profiler Evidence

On the recorded Ascend runtime, the stock `torch.profiler` path exports only
host-side `cpu_op` events (`aclnn*` / `aten::*` operator calls) and
`user_annotation` scopes; there are no `cat=kernel` device-duration events. NPU
AI Core kernel durations are instead captured by `torch_npu.profiler`, which
triggers the CANN msprof capture whose sqlite output
(`device_0/sqlite/ai_core_op_summary.db`, tables `task_time` + `ge_summary`)
records per-task kernel `duration_time` in nanoseconds. The harness sets
`ASCEND_WORK_PATH` to the campaign `log/` directory and prints the
`cann_profiling_data` path; `summarize_cann_trace.py` reads that sqlite and
reports `device_time_available=true` with `device_us_per_call`,
`kernel_count_per_call`, and a top-k kernel breakdown. Device time is therefore
available, but a raw `torch.profiler` chrome trace alone is not device evidence.
Benchmark wall time (unrounded median) controls adoption; profiler device time
is diagnostic evidence.

## Supported Primitives

Established by a local compile-and-execute probe with numeric checks on the
recorded runtime (Ascend910B4, torch_npu 2.7.1.post4, triton 3.2.0):

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| `tl.load` | Supported | Masked contiguous loads verified; bounds and alignment remain shape-specific. | ascend primitive probe (`load/store/arange/program_id`) | Local misuse is implementation repair; runtime mismatch is environment-blocked. |
| `tl.store` | Supported | Contiguous stores verified for the probed dtypes. | same probe | Local misuse is implementation repair. |
| `tl.arange` | Supported | Extents 64/128/256 verified; other extents remain Unknown. | same probe | Unsupported required extent is capability-miss. |
| `tl.program_id` | Supported | Axis 0 in single- and multi-program launches. | same probe | Incorrect mapping is implementation repair. |
| `tl.zeros` | Supported | `(64,)` float32 verified. | ascend primitive probe (`zeros/full/where`) | Unsupported required shape or dtype is capability-miss. |
| `tl.full` | Supported | Scalar fill `(64,)` float32 verified. | same probe | Unsupported required dtype is capability-miss. |
| `tl.where` | Supported | Masked selection verified. | same probe | Incorrect masking is implementation repair. |
| `tl.reshape` | Supported | Element-count-preserving `(16,)` to `(4,4)` verified. | ascend primitive probe (`reshape/broadcast/max(axis=1)`) | Invalid shape is implementation repair. |
| `tl.broadcast_to` | Supported | `(4,1)` to `(4,4)` broadcast verified. | same probe | Unsupported required shape is capability-miss. |
| `tl.max` | Supported | Axis-1 reduction over `(4,4)` and axis-0 scalar reduction verified. | same probe + `max/sum/argmax(axis=0)` | Unsupported required reduction is capability-miss. |
| `tl.sum` | Supported | Axis-0 scalar reduction verified. | `max/sum/argmax(axis=0)` | Unsupported required reduction is capability-miss. |
| `tl.argmax` | Supported | Axis-0 reduction over a float32 vector verified; tie behavior not characterized. | `max/sum/argmax(axis=0)` | Semantic mismatch is implementation repair or capability-miss according to the decision. |
| `tl.exp` | Supported | Elementwise float32 exponential verified. | ascend primitive probe (`exp`) | Unsupported required math is capability-miss. |
| `tl.static_range` | Supported | Compile-time loop `K=8` verified. | ascend primitive probe (`static_range`) | Unsupported required control construct is capability-miss. |
| `tl.dot` | Supported | `(16,16)@(16,16)` float32 verified. | ascend primitive probe (`dot`) | Unavailable required shape/dtype is capability-miss. |

## Constrained Primitives

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| `num_warps` | Constrained | `num_warps=1`, `2`, and `4` all compiled and ran correctly in the probe; performance tradeoffs are architecture-specific and must be measured per candidate. | ascend primitive probe (`num_warps=1/2/4`) | A required unavailable value is capability-miss; optional tuning falls back to a proven value. |
| `tl.argmax` tie behavior | Constrained | The probe uses unique values only; equal-value ordering is not established. | `max/sum/argmax(axis=0)` | A semantic tie mismatch is implementation repair or design rejection. |

## Unsupported Primitives

No primitive is established as universally Unsupported by current repository
evidence. Do not invent unsupported entries from GCU, MLU, CUDA, or vendor
assumptions. A matched local compile/runtime failure may establish an
Unsupported result for a future profile revision.

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| None established | Unsupported | Do not add an entry without a matched local failure or authoritative evidence. | Repository evidence review | A normative unsupported requirement is capability-miss. |

## Unknown Primitives

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| `tl.make_block_ptr` | Unknown | Memory-placement and support semantics require a matched Ascend probe. | No qualifying Ascend probe | An unprovable normative use is capability-miss. |
| `vectorize` | Unknown | Accepted values and backend effect require a matched Ascend probe. | No qualifying Ascend probe | An unprovable normative use is capability-miss. |
| `async_copy` | Unknown | Availability, synchronization, and memory semantics require a matched Ascend probe. | No qualifying Ascend probe | An unprovable normative use is capability-miss. |
| `num_stages` | Unknown | No Ascend architecture-specific legality or performance probe is recorded. | No qualifying Ascend probe | An unprovable normative use is capability-miss. |
| `fast_libentry` | Unknown | No Ascend probe establishes a fast launcher path. | No qualifying Ascend probe | A normative fast launcher requirement is capability-miss. |
| stream and context semantics | Unknown | The probe does not establish current-stream preservation or context ownership. | No qualifying Ascend probe | A normative unproven lifecycle requirement is capability-miss. |
| non-contiguous and mixed-precision behavior | Unknown | The probe uses contiguous float32 buffers only. | Ascend primitive probe | An unprovable normative use is capability-miss. |
| `tl.dot` dtypes beyond float32 | Unknown | Only float32 dot was probed. | `dot` probe | An unprovable required dtype is capability-miss. |

## Allowed Fallbacks

A fallback is allowed only when the decision leaves the choice non-normative and
preserves the Optimization Intent, Unified Sketch, Host Plan, and Evaluation
Contract. Any proven `num_warps` value may replace an optional unproven tuning
value. Direct Triton launch may replace optional `fast_libentry` only when the
Host Plan does not require launcher reduction. GCU/MLU device APIs, GCU/MLU
stream assumptions, algorithm changes, dataflow changes, and lifecycle changes
are not fallbacks; they require a new decision or are a major deviation.

## Target-specific Pitfalls

- `device="npu"` and `torch_npu.npu.synchronize()` are the observed Ascend APIs;
  GCU/MLU/CUDA strings and synchronization assumptions are not interchangeable.
- `torch_npu` and `triton` versions must be recorded in the runtime fingerprint.
  There is no importable `triton_ascend` module on this runtime; the backend is
  `triton.backends.backends['ascend']`. Do not write `import triton_ascend`.
- The shared `base.py` uses `device="cuda"` as a neutral placeholder; the harness
  rewrites it to `npu`. Candidate `get_inputs` should not hardcode `"cuda"`.
- The primitive probe covers small fixed shapes and one launch configuration per
  primitive. It does not prove arbitrary shapes, dtypes, grids, streams, or
  concurrent execution.
- The harness AST loader and Ascend launcher construction must be validated with
  the real harness, not a direct import alone.
- Reusing output buffers requires an explicit Host Plan with cache keys,
  invalidation, device/stream behavior, and concurrency assumptions.
- Ascend evidence does not automatically generalize to other NPU architectures.

## Evidence Ledger

| Claim | Repository evidence | Scope |
|---|---|---|
| NPU tensors, synchronization, direct Triton launch, loads, stores, indexing, zeros, full, where, reshape, broadcast, max, sum, argmax, exp, static_range, and dot executed with checked results. | ascend primitive probe (run on Ascend910B4) | One Ascend910B4 runtime, torch_npu 2.7.1.post4, triton 3.2.0, fixed float32 buffers, small shapes. |
| `num_warps=1`, `2`, and `4` compiled and executed correctly. | ascend primitive probe | One launch configuration family on the recorded runtime only. |
| `torch_npu` is the required import; `triton_ascend` is metadata-only and not importable. | runtime introspection | Package behavior and versions must be rediscovered in each project fingerprint. |
| Ascend support for block pointers, async copy, fast launcher, stages, streams, and mixed precision remains unproven. | This profile and absence of a qualifying probe. | Unknown is not treated as Unsupported or Supported. |
