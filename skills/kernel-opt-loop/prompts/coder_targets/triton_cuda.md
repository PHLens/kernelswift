# Target Profile: triton_cuda

This profile records only capabilities observed on a matched Triton runtime that
launches through the `torch.cuda` API surface. Current repository evidence is
from an Iluvatar BI-V150 / CoreX environment. Absence from Supported is not
proof of support.

## Identity and Match

The decision and manifest must say `language: triton`, `backend: cuda`, and
`target_profile: triton_cuda`. Phase 0 must discover the Triton distribution and
version, the active `torch` runtime, the active compiler or driver target and
version when available, and the CUDA-visible device architecture. The recorded
BI150 probe reports `Iluvatar BI-V150` with `major=7, minor=1`,
`multi_processor_count=16`, and `total_memory=17179869184` bytes. A missing
runtime, backend mismatch, or architecture/profile mismatch is
`environment-blocked`, not a capability miss. Repository evidence does not
replace the current project's runtime fingerprint.

## Runtime and Launcher Conventions

- On the recorded BI150 host, a fresh shell must set `COREX_VERSION=4.4.0` and
  source `/usr/local/corex/enable` before importing `torch` or `triton`;
  without that bootstrap, imports and `ixsmi` fail.
- Use `device="cuda"` for the observed accelerator path and
  `torch.cuda.synchronize()` for the observed synchronization boundary.
- Direct Triton launch syntax `kernel[(grid,)](...)` was observed.
- `fast_libentry` import behavior is unproven on this profile revision; do not
  require it without a matched local probe.
- The actual harness AST loader must be used. `torch.cuda` API compatibility
  alone does not prove launcher, stream, or profiler semantics.
- Output caching, device-context removal, current-stream behavior, and
  concurrent model-instance semantics are Host Plan concerns and are not proven
  by the primitive probe.

## Profiler Evidence

No matched BI150 profiler export is yet recorded in this repository. Do not
assume `cat=kernel` device-duration availability, vendor-specific trace fields,
or a stable device-time interpretation until a local scoped export proves them.

## Supported Primitives

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| `tl.load` | Supported | Contiguous one-dimensional float32 loads at extent `256`. | `scripts/bi150_triton_smoke.py`; `scripts/bi150_groupedtopk_probe.py` | Local misuse is implementation repair; runtime mismatch is environment-blocked. |
| `tl.store` | Supported | Contiguous stores for one-dimensional extents `256`, `8`, and `1` with float32/int32 outputs. | `scripts/bi150_triton_smoke.py`; `scripts/bi150_groupedtopk_probe.py` | Local misuse is implementation repair; runtime mismatch is environment-blocked. |
| `tl.arange` | Supported | Extents `256` and `8` in one-dimensional launches. | `scripts/bi150_triton_smoke.py`; `scripts/bi150_groupedtopk_probe.py` | Unsupported required extent is capability-miss. |
| `tl.program_id` | Supported | Axis `0` in a one-dimensional launch. | `scripts/bi150_triton_smoke.py` | Incorrect mapping is implementation repair. |
| `tl.reshape` | Supported | Element-count-preserving `(256,)` to `(8,32)` and `(8,)` to `(8,1)` reshapes. | `scripts/bi150_groupedtopk_probe.py` | Invalid shape is implementation repair. |
| `tl.max` | Supported | Axis-`1` reduction over `(8,32)` and axis-`0` reduction over `(256,)`; float32 only. | `scripts/bi150_groupedtopk_probe.py` | Unsupported required reduction shape/dtype is capability-miss. |
| `tl.sum` | Supported | Axis-`1` reduction over `(8,32)` and axis-`0` reduction over `(256,)`; float32 only. | `scripts/bi150_groupedtopk_probe.py` | Unsupported required reduction shape/dtype is capability-miss. |
| `tl.exp` | Supported | Elementwise float32 exponential over a contiguous `(256,)` vector. | `scripts/bi150_groupedtopk_probe.py` | Unsupported required math is capability-miss. |
| `tl.sqrt` / `tl.sin` / `tl.cos` | Supported | Elementwise float32 transcendentals over contiguous vectors; lower to the same libdevice intrinsics as `torch.sqrt/sin/cos` with bit-identical results (max abs diff 0.0). | `scripts/bi150_tl_dot_probe_bf16.py` (file-backed transcendental probe); `kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_002.py` (round 002 accepted) | Unsupported required math is capability-miss; a numerically-divergent lowering is implementation repair. |
| `tl.argmax` | Supported | Axis-`0` argmax over an `(8,)` float32 vector with a unique maximum. | `scripts/bi150_groupedtopk_probe.py` | Incorrect index mapping is implementation repair; unproven tie semantics remain constrained. |
| `tl.zeros` | Supported | One-dimensional `(8,)` float32 value tensor. | `scripts/bi150_groupedtopk_probe.py` | Unsupported required shape/dtype is capability-miss. |
| `tl.full` | Supported | One-dimensional `(8,)` and `(8,32)` float32 fills. | `scripts/bi150_groupedtopk_probe.py` | Unsupported required shape/dtype is capability-miss. |
| `tl.where` | Supported | Boolean selection over `(8,)` and `(8,32)` float32 values. | `scripts/bi150_groupedtopk_probe.py` | Incorrect masking is implementation repair. |
| `tl.broadcast_to` | Supported | `(8,1)` to `(8,32)` broadcast in the grouped probe. | `scripts/bi150_groupedtopk_probe.py` | Unsupported required shape is capability-miss. |
| `tl.static_range` | Supported | Compile-time loop with four iterations in the grouped probe. | `scripts/bi150_groupedtopk_probe.py` | Unsupported required control construct is capability-miss. |
| `tl.dot` | Supported | `(32,32) @ (32,32)` matmul with fp32 inputs → exact (`0.0` max abs err); bf16 inputs → fp32 accumulate with `9.5e-7` max abs err, `1.2e-6` max rel err. | `scripts/bi150_tl_dot_probe2.py`; `scripts/bi150_tl_dot_probe_bf16.py` | Unsupported required shape/dtype is capability-miss; incorrect result is implementation repair. |

## Constrained Primitives

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| `masked elementwise indexing` | Constrained | The matched probe covers contiguous `(8,32)` masking and reshape-backed storage only; gather/scatter, arbitrary multidimensional indexing, and aliasing remain unproven. | `scripts/bi150_groupedtopk_probe.py` | An unproven normative indexing requirement is capability-miss. |
| `argmax tie and repeated selection` | Constrained | The probe covers one unique maximum only; repeated top-k selection and PyTorch-compatible tie ordering remain to be established. | `scripts/bi150_groupedtopk_probe.py` | An unproven normative tie or selection requirement is capability-miss. |
| `launch configuration` | Constrained | The matched probes use a one-dimensional grid and direct launch; no explicit `num_warps` or `num_stages` hint is established. | `scripts/bi150_triton_smoke.py`; `scripts/bi150_groupedtopk_probe.py` | An unavailable required launch requirement is capability-miss; an incorrect optional setting is implementation repair. |
| `dtype and layout regime` | Constrained | Contiguous float32 vectors, `(8,32)` row layout, and `(32,32)` bf16 matmul inputs are proven; other mixed precision, non-contiguous inputs, and arbitrary layouts remain unproven. | `scripts/bi150_groupedtopk_probe.py`; `scripts/bi150_tl_dot_probe_bf16.py` | An unproven normative shape, layout, or dtype requirement is capability-miss. |
| `torch.compile` | Constrained | File-backed CUDA add-one functions compile and execute through the CoreX Torch 2.7.1 runtime with default and `reduce-overhead` modes. Graph coverage, graph breaks, cache behavior, streams, and grouped-topk lowering remain unproven. | `scripts/bi150_torch_compile_probe.py`; `scripts/bi150_torch_compile_reduce_overhead_probe.py` | A missing required lifecycle or lowering property is capability-miss; compile/runtime failure is environment-blocked. |

## Unsupported Primitives

No primitive is established as universally Unsupported by current repository
criteria. Do not invent CUDA-vendor restrictions or NVIDIA-specific allowances
without matched local evidence.

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| None established | Unsupported | Do not add an entry without a matched local failure or authoritative evidence. | Repository evidence review | A normative unsupported requirement is capability-miss. |

## Unknown Primitives

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| `tl.make_block_ptr` | Unknown | Memory-placement and pointer semantics require a matched local probe. | No qualifying BI150 probe | An unprovable normative use is capability-miss. |
| `vectorize` | Unknown | Accepted values and backend effect require a matched local probe. | No qualifying BI150 probe | An unprovable normative use is capability-miss. |
| `async_copy` | Unknown | Availability, synchronization, and memory semantics require a matched local probe. | No qualifying BI150 probe | An unprovable normative use is capability-miss. |
| `num_warps` | Unknown | No explicit working value is recorded on this profile revision. | No qualifying BI150 probe | A normative unproven hint is capability-miss. |
| `num_stages` | Unknown | No explicit working value is recorded on this profile revision. | No qualifying BI150 probe | A normative unproven hint is capability-miss. |
| `fast_libentry` | Unknown | The profile records direct launch only; no matched `fast_libentry` import or execution probe is recorded. | No qualifying BI150 probe | A normative fast-launcher requirement is capability-miss. |
| stream and context semantics | Unknown | The matched probe does not establish current-stream preservation or context ownership. | No qualifying BI150 probe | A normative unproven lifecycle requirement is capability-miss. |
| non-contiguous and mixed-precision behavior | Unknown | The matched probe uses contiguous `fp32` vectors only. | `scripts/bi150_triton_smoke.py` | An unprovable normative use is capability-miss. |

## Allowed Fallbacks

A fallback is allowed only when the decision leaves the choice non-normative and
preserves the Optimization Intent, Unified Sketch, Host Plan, and Evaluation
Contract. Direct Triton launch may replace optional `fast_libentry` only when
the Host Plan does not require launcher reduction. Algorithm changes, dataflow
changes, lifecycle changes, vendor-specific API assumptions, and profiler-field
assumptions are not fallbacks; they require a new decision or are a major
deviation.

## Target-specific Pitfalls

- `torch.cuda` API compatibility is not evidence of NVIDIA-specific behavior;
  match the current runtime fingerprint and recorded device identity.
- On the recorded BI150 host, forgetting `export COREX_VERSION=4.4.0` and
  `. /usr/local/corex/enable` makes `torch` and `triton` imports fail and leaves
  `ixsmi` unable to load `libixml.so`.
- A quick Triton probe defined through stdin failed with `could not get source
  code`; use a file-backed local probe when validating target behavior.
- The harness AST loader and launch construction must be validated with the real
  harness, not a direct import alone.
- No matched profiler export is recorded; do not plan mechanism observables that
  require unproven device-time fields.
- `num_warps`, `num_stages`, block pointers, and mixed precision remain
  unproven on this profile revision; repeated argmax tie behavior remains
  constrained rather than proven.

## Evidence Ledger

| Claim | Repository evidence | Scope |
|---|---|---|
| BI150 shells require a CoreX environment bootstrap before `torch`, `triton`, and `ixsmi` work. | `docs/bi150-kernel-opt-loop-prep.md` | Recorded BI150 / CoreX 4.4.0 environment only. |
| After bootstrap, `torch.cuda` exposes `Iluvatar BI-V150` with `major=7, minor=1`, `multi_processor_count=16`, and `total_memory=17179869184`. | `docs/bi150-kernel-opt-loop-prep.md`; `scripts/bi150_triton_smoke.py` | One BI150 device on the recorded runtime only. |
| Direct Triton launch, `tl.program_id`, `tl.arange`, `tl.load`, and `tl.store` execute with checked results on the recorded BI150 runtime. | `scripts/bi150_triton_smoke.py`; `docs/bi150-kernel-opt-loop-prep.md` | One-dimensional vector-add probe with contiguous `fp32` tensors and `device="cuda"` only. |
| Grouped-topk-shaped reductions, reshape, argmax, exp, sum, where, broadcast, full, zeros, static-range, and masked stores execute with checked results on BI150. | `scripts/bi150_groupedtopk_probe.py` | One program over a contiguous float32 vector of length 256 reshaped to `(8,32)`; unique argmax maximum only. |
| `torch.cuda.synchronize()` is the observed synchronization boundary for the matched probes. | `scripts/bi150_triton_smoke.py`; `scripts/bi150_groupedtopk_probe.py` | Same BI150 runtime and probe regime only. |
| `torch.compile` compiles and executes file-backed CUDA add-one functions with exact output in default and `reduce-overhead` modes on the recorded BI150 runtime. | `scripts/bi150_torch_compile_probe.py`; `scripts/bi150_torch_compile_reduce_overhead_probe.py` | Basic compiler-mode availability only; grouped-topk graph coverage and lifecycle remain constrained. |
| `tl.dot` executes `(32,32) @ (32,32)` matmul with exact `fp32` results and near-exact `bf16`-input results (fp32 accumulate). | `scripts/bi150_tl_dot_probe2.py`; `scripts/bi150_tl_dot_probe_bf16.py` | One square-tile matmul shape on the recorded BI150 runtime; other shapes/layouts unproven. |
| Block pointers, explicit launch hints, stream/context lifecycle semantics, mixed precision, and profiler interpretation beyond the recorded CUDA trace remain unproven. | This profile and absence of a qualifying probe. | Unknown is not treated as Supported or Unsupported. |
