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
| `tl.load` | Supported | Masked contiguous one-dimensional float32 loads in the recorded vector-add probe only. | `scripts/bi150_triton_smoke.py`; `docs/bi150-kernel-opt-loop-prep.md` | Local misuse is implementation repair; runtime mismatch is environment-blocked. |
| `tl.store` | Supported | Contiguous stores with the recorded output shape and dtype only. | `scripts/bi150_triton_smoke.py`; `docs/bi150-kernel-opt-loop-prep.md` | Local misuse is implementation repair; runtime mismatch is environment-blocked. |
| `tl.arange` | Supported | Extent `256` in the recorded one-dimensional probe only. | `scripts/bi150_triton_smoke.py`; `docs/bi150-kernel-opt-loop-prep.md` | Unsupported required extent is capability-miss. |
| `tl.program_id` | Supported | Axis `0` in a one-dimensional launch only. | `scripts/bi150_triton_smoke.py`; `docs/bi150-kernel-opt-loop-prep.md` | Incorrect mapping is implementation repair. |

## Constrained Primitives

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| `masked elementwise indexing` | Constrained | The probe covers `offs < n_elements` masking only; gather/scatter, multidimensional indexing, and aliasing behavior remain unproven. | `scripts/bi150_triton_smoke.py`; `docs/bi150-kernel-opt-loop-prep.md` | An unproven normative indexing requirement is capability-miss. |
| `launch configuration` | Constrained | The recorded probe uses a one-dimensional grid and `BLOCK=256`; no explicit `num_warps` or `num_stages` hint is established. | `scripts/bi150_triton_smoke.py`; `docs/bi150-kernel-opt-loop-prep.md` | An unavailable required launch requirement is capability-miss; an incorrect optional setting is implementation repair. |
| `dtype and layout regime` | Constrained | Only contiguous `fp32` vectors are proven by the matched probe. | `scripts/bi150_triton_smoke.py`; `docs/bi150-kernel-opt-loop-prep.md` | An unproven normative shape, layout, or dtype requirement is capability-miss. |

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
| `tl.zeros` | Unknown | No qualifying BI150 probe for the required shape or dtype. | No qualifying BI150 probe | An unprovable normative use is capability-miss. |
| `tl.reshape` | Unknown | No matched probe establishes legal shapes or layout-preserving behavior. | No qualifying BI150 probe | An unprovable normative use is capability-miss. |
| `tl.max` | Unknown | No matched reduction probe is recorded. | No qualifying BI150 probe | An unprovable normative reduction is capability-miss. |
| `tl.argmax` | Unknown | No matched reduction probe or tie-behavior characterization is recorded. | No qualifying BI150 probe | An unprovable normative reduction is capability-miss. |
| `tl.sum` | Unknown | No matched reduction probe is recorded. | No qualifying BI150 probe | An unprovable normative reduction is capability-miss. |
| `tl.exp` | Unknown | No matched transcendental-math probe is recorded. | No qualifying BI150 probe | An unprovable normative math requirement is capability-miss. |
| `tl.where` | Unknown | No matched masked-select probe is recorded. | No qualifying BI150 probe | An unprovable normative mask requirement is capability-miss. |
| `tl.dot` | Unknown | No qualifying probe for the required shape, dtype, or lowering path. | No qualifying BI150 probe | An unprovable normative use is capability-miss. |
| `tl.broadcast_to` | Unknown | No matched broadcast probe is recorded. | No qualifying BI150 probe | An unprovable normative shape requirement is capability-miss. |
| `tl.full` | Unknown | No matched fill probe is recorded. | No qualifying BI150 probe | An unprovable normative fill requirement is capability-miss. |
| `tl.static_range` | Unknown | No matched compile-time loop probe is recorded. | No qualifying BI150 probe | An unprovable normative control requirement is capability-miss. |
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
- `num_warps`, `num_stages`, reductions, `tl.dot`, block pointers, and mixed
  precision remain unproven on this profile revision.

## Evidence Ledger

| Claim | Repository evidence | Scope |
|---|---|---|
| BI150 shells require a CoreX environment bootstrap before `torch`, `triton`, and `ixsmi` work. | `docs/bi150-kernel-opt-loop-prep.md` | Recorded BI150 / CoreX 4.4.0 environment only. |
| After bootstrap, `torch.cuda` exposes `Iluvatar BI-V150` with `major=7, minor=1`, `multi_processor_count=16`, and `total_memory=17179869184`. | `docs/bi150-kernel-opt-loop-prep.md`; `scripts/bi150_triton_smoke.py` | One BI150 device on the recorded runtime only. |
| Direct Triton launch, `tl.program_id`, `tl.arange`, `tl.load`, and `tl.store` execute with checked results on the recorded BI150 runtime. | `scripts/bi150_triton_smoke.py`; `docs/bi150-kernel-opt-loop-prep.md` | One-dimensional vector-add probe with contiguous `fp32` tensors and `device="cuda"` only. |
| `torch.cuda.synchronize()` is the observed synchronization boundary for the matched probe. | `scripts/bi150_triton_smoke.py`; `docs/bi150-kernel-opt-loop-prep.md` | Same BI150 runtime and probe only. |
| `fast_libentry`, reductions, dot, block pointers, explicit launch hints, stream semantics, and profiler interpretation remain unproven. | This profile and absence of a qualifying probe. | Unknown is not treated as Supported or Unsupported. |
