# Target Profile: triton_gcu

This profile records only capabilities observed on the available Triton-GCU
runtime and probe configuration. Evidence is local to the recorded runtime,
device architecture, shapes, dtypes, and compiler versions. Absence from
Supported is not evidence of support.

## Identity and Match

The decision and manifest must say `language: triton`, `backend: gcu`, and
`target_profile: triton_gcu`. Phase 0 must discover the Triton distribution and
version, the `triton_gcu` and `torch_gcu` package versions, the active compiler
or driver target when available, and the GCU device architecture. The recorded
S60 probe reports a GCU device with `major=3, minor=0`; this is evidence for
that runtime only. A missing runtime, backend mismatch, or architecture/profile
mismatch is `environment-blocked`, not a capability miss. Repository evidence
does not replace the current project's runtime fingerprint.

## Runtime and Launcher Conventions

- Import `torch_gcu` before allocating GCU tensors or launching Triton kernels;
  the probe also imports `triton_gcu` explicitly.
- Use `device="gcu"` for the selected accelerator and
  `torch.gcu.synchronize()` for the observed synchronization boundary.
- Direct Triton launch syntax `kernel[(grid,)](...)` was observed. Both
  `from triton.runtime.fast_libentry import fast_libentry` and `from
  triton.runtime import fast_libentry` imports were unavailable in the recorded runtime; direct launch is the
  proven launcher path for this project.
- The actual harness loader must be used. MLU device strings, MLU stream
  assumptions, and MLU launcher behavior must not be inferred for GCU.
- Output caching, device context removal, stream behavior, and concurrent
  model-instance semantics are Host Plan concerns and remain unproven by the
  primitive probe.

## Profiler Evidence

On the recorded S60 runtime, `torch.profiler` with `CPU` and `PrivateUse1`
exports `gcu_runtime` launch events but no `cat=kernel` device-duration events.
The summary helper therefore reports normalized `runtime_launch_*` fields and
sets `device_time_available=false`; those launch durations are not device
kernel time and must not be used to compute `device_ratio`. A future matched
TOPS/TOPSPTI exporter may establish device duration and revise this limitation.

## Supported Primitives

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| `tl.load` | Supported | Masked contiguous load in the recorded 16-element float32 probe only. | `s60/groupedtopk/triton_grouped_topk_001.py` | Local misuse is implementation repair; runtime mismatch is environment-blocked. |
| `tl.store` | Supported | Contiguous stores with the recorded output shapes and dtypes only. | `s60/groupedtopk/triton_grouped_topk_001.py` | Local misuse is implementation repair; runtime mismatch is environment-blocked. |
| `tl.arange` | Supported | Extent 16 and extent 4 in the recorded probe; other extents remain Unknown. | `s60/groupedtopk/triton_grouped_topk_001.py` | Unsupported required extent is capability-miss. |
| `tl.program_id` | Supported | Axis 0 in a one-program launch. | `s60/groupedtopk/triton_grouped_topk_001.py` | Incorrect mapping is implementation repair. |
| `tl.zeros` | Supported | Shape `(16,)`, float32, in the recorded probe. | `s60/groupedtopk/triton_grouped_topk_001.py` | Unsupported required shape or dtype is capability-miss. |
| `tl.reshape` | Supported | Element-count-preserving `(16,)` to `(4,4)` reshape in the recorded probe. | `s60/groupedtopk/triton_grouped_topk_001.py` | Invalid shape is implementation repair. |
| `tl.max` | Supported | Axis-1 reduction over a `(4,4)` float32 matrix in the recorded probe. | `s60/groupedtopk/triton_grouped_topk_001.py` | Unsupported required reduction is capability-miss. |
| `tl.argmax` | Supported | Axis-0 reduction over the recorded float32 vector; tie behavior was not characterized. | `s60/groupedtopk/triton_grouped_topk_001.py` | Semantic mismatch is implementation repair or capability-miss according to the decision. |
| `tl.sum` | Supported | Scalar reductions in the recorded group-topk candidate; other shapes remain target-specific. | `s60/groupedtopk/triton_grouped_topk_001.py` | Unsupported required reduction is capability-miss. |
| `tl.exp` | Supported | Elementwise float32 exponential in the recorded group-topk candidate. | `s60/groupedtopk/triton_grouped_topk_001.py` | Unsupported required math is capability-miss. |
| `tl.where` | Supported | Masked selection in the recorded group-topk candidate. | `s60/groupedtopk/triton_grouped_topk_001.py` | Incorrect masking is implementation repair. |
| `tl.broadcast_to` | Supported | `(n_group,1)` to `(n_group,epg)` in the recorded shape. | `s60/groupedtopk/triton_grouped_topk_001.py` | Unsupported required shape is capability-miss. |
| `tl.full` | Supported | Compile-time scalar fill for the recorded output vector. | `s60/groupedtopk/triton_grouped_topk_001.py` | Unsupported required dtype is capability-miss. |
| `tl.static_range` | Supported | Compile-time loops with `KG=4` and `K=8` in the recorded candidate. | `s60/groupedtopk/triton_grouped_topk_001.py` | Unsupported required control construct is capability-miss. |

## Constrained Primitives

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| `num_warps` | Constrained | `num_warps=1` compiled and ran in the probe. Other values are Unknown on this architecture. | `s60/groupedtopk/triton_grouped_topk_001.py` | A required unavailable value is capability-miss; optional tuning falls back to 1. |
| `tl.argmax` tie behavior | Constrained | The probe uses unique values only; equal-value ordering is not established. | `s60/groupedtopk/triton_grouped_topk_001.py` | A semantic tie mismatch is implementation repair or design rejection. |
| reduction shapes and dtypes | Constrained | Only the recorded float32 vector/matrix reductions are proven. | `s60/groupedtopk/triton_grouped_topk_001.py` | An unproven normative shape or dtype is capability-miss. |

## Unsupported Primitives

No primitive is established as universally Unsupported by current repository
evidence. Do not invent unsupported entries from MLU, CUDA, or vendor
assumptions. A matched local compile/runtime failure may establish an
Unsupported result for a future profile revision.

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| None established | Unsupported | Do not add an entry without a matched local failure or authoritative evidence. | Repository evidence review | A normative unsupported requirement is capability-miss. |

## Unknown Primitives

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| `tl.dot` | Unknown | No qualifying GCU probe for the required shape or dtype. | No qualifying GCU probe | An unprovable normative use is capability-miss. |
| `tl.make_block_ptr` | Unknown | Memory-placement and support semantics require a matched GCU probe. | No qualifying GCU probe | An unprovable normative use is capability-miss. |
| `vectorize` | Unknown | Accepted values and backend effect require a matched GCU probe. | No qualifying GCU probe | An unprovable normative use is capability-miss. |
| `async_copy` | Unknown | Availability, synchronization, and memory semantics require a matched GCU probe. | No qualifying GCU probe | An unprovable normative use is capability-miss. |
| `num_stages` | Unknown | No GCU architecture-specific legality or performance probe is recorded. | No qualifying GCU probe | An unprovable normative use is capability-miss. |
| `fast_libentry` | Unknown | Both observed import paths failed on the recorded runtime; no alternate path is established. | S60 import probe output; direct launch candidate | A normative fast launcher requirement is capability-miss. |
| stream and context semantics | Unknown | The probe does not establish current-stream preservation or context ownership. | No qualifying GCU probe | A normative unproven lifecycle requirement is capability-miss. |
| non-contiguous and mixed-precision behavior | Unknown | The probe uses contiguous float32 buffers only. | `s60/groupedtopk/triton_grouped_topk_001.py` | An unprovable normative use is capability-miss. |

## Allowed Fallbacks

A fallback is allowed only when the decision leaves the choice non-normative and
preserves the Optimization Intent, Unified Sketch, Host Plan, and Evaluation
Contract. The observed `num_warps=1` may replace an optional unproven tuning
value. Direct Triton launch may replace optional `fast_libentry` only when the
Host Plan does not require launcher reduction. MLU device APIs, MLU stream
assumptions, algorithm changes, dataflow changes, and lifecycle changes are not
fallbacks; they require a new decision or are a major deviation.

## Target-specific Pitfalls

- `device="gcu"` and `torch.gcu.synchronize()` are the observed GCU APIs;
  MLU/CUDA strings and synchronization assumptions are not interchangeable.
- `torch_gcu` and `triton_gcu` package versions must be recorded in the runtime
  fingerprint. Importing only generic `torch` is insufficient evidence.
- The primitive probe covers one program, fixed 16-element float32 buffers, and
  one launch configuration. It does not prove arbitrary shapes, dtypes, grids,
  streams, or concurrent execution.
- The harness AST loader and GCU launcher construction must be validated with
  the real harness, not a direct import alone.
- Reusing output buffers requires an explicit Host Plan with cache keys,
  invalidation, device/stream behavior, and concurrency assumptions.
- S60 evidence does not automatically generalize to other GCU architectures.

## Evidence Ledger

| Claim | Repository evidence | Scope |
|---|---|---|
| GCU tensors, synchronization, direct Triton launch, loads, stores, indexing, zeros, reshape, max, and argmax executed with checked results. | `s60/groupedtopk/triton_grouped_topk_001.py` | One S60/GCU runtime, GCU architecture `major=3, minor=0`, fixed float32 buffers, one program. |
| Group-topk candidate compiled and ran with softmax math, reductions, masking, compile-time loops, and one direct module launch per token. | `s60/groupedtopk/triton_grouped_topk_001.py` and Round 001 harness smoke output. | Same S60/GCU runtime and exact `T=83,E=256,K=8` regime; not a general-shape guarantee. |
| `num_warps=1` compiled and executed. | `s60/groupedtopk/triton_grouped_topk_001.py` | One launch configuration on the recorded runtime only. |
| `torch_gcu` and `triton_gcu` are required imports in the recorded setup. | `s60/groupedtopk/triton_grouped_topk_001.py` | Package behavior and versions must be rediscovered in each project fingerprint. |
| GCU support for dot, block pointers, async copy, fast launcher, stages, streams, and mixed precision remains unproven. | This profile and absence of a qualifying probe. | Unknown is not treated as Unsupported or Supported. |
