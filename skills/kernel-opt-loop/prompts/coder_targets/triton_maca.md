# Target Profile: triton_maca

> **Migration status**: this Markdown page is an explanatory rendering of the
> target. It has no vNext canonical implementation profile yet — no reviewed
> `profiles/<id>/profile.yaml`, executable versioned probe suite, or approved
> evidence record exists under `skills/kernel-opt-loop/profiles/`. Claims below
> are historical human context until a machine-readable profile is promoted.


This profile records only capabilities observed on the matched Triton-MACA
runtime. Evidence is local to the recorded runtime, MetaX C500 architecture,
shapes, dtypes, compiler, and launch configuration. Absence from Supported is
not evidence of support.

## Identity and Match

The decision and manifest must say `language: triton`, `backend: maca`, and
`target_profile: triton_maca`. Phase 0 must discover the Triton distribution
and version, active compiler or driver target, MACA version, PyTorch build,
device name, architecture, and target warp size. A missing `MACA_PATH`, missing
runtime, backend mismatch, or architecture/profile mismatch is
`environment-blocked`, not a capability miss.

The recorded probe used PyTorch `2.8.0+metax3.5.3.9`, Triton package
`3.0.0+metax3.5.3.9` (`triton.__version__ == 3.0.0`), MACA `3.5.3.26`, and a
MetaX C500. Triton's active target was
`GPUTarget(backend='maca', arch=80, warp_size=64)`. These values are evidence
for that probe only and must be rediscovered for every run.

## Runtime and Launcher Conventions

- `MACA_PATH` must be defined before importing Triton. On the recorded host it
  is `/opt/maca`; non-interactive SSH commands did not source the required
  environment and failed in `triton/backends/metax/driver.py` before import.
- Use the selected environment's absolute interpreter. The recorded
  interpreter is `/opt/conda/bin/python` after the MACA environment is loaded.
- MetaX tensors use PyTorch's `device="cuda"` compatibility surface and
  `torch.cuda.synchronize()` on the recorded runtime. Preserve caller-selected
  device and current-stream behavior.
- Direct launch syntax `kernel[(grid,)](...)` is the proven launcher path.
  Both observed `fast_libentry` import locations were unavailable.
- The actual harness AST loader must be used. It retains imports, classes, and
  functions but removes nonliteral module assignments.
- Output caching, launcher replacement, device-context removal, and stream or
  concurrency assumptions are Host Plan concerns and require explicit cache
  keys, invalidation, aliasing, device, and stream semantics.

## Profiler Evidence

On the recorded C500 runtime, `torch.profiler` with `CPU` and `CUDA` activities
exported a trace containing `cat=kernel`, `cuda_runtime`, and `ac2g` events for
the primitive probe. Kernel durations are available for normalized device-time
evidence when the run's trace contains the expected scoped iterations.

## Supported Primitives

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| `tl.load` | Supported | Contiguous float32 load with extent 256 in the recorded probe. Bounds and masking remain shape-specific. | C500 primitive probe | Local misuse is implementation repair; runtime mismatch is environment-blocked. |
| `tl.store` | Supported | Contiguous float32 and int32 stores with extents 256, 8, and 1. | C500 primitive probe | Local misuse is implementation repair. |
| `tl.arange` | Supported | Extents 256 and 8 in the recorded probe. Other extents remain Unknown. | C500 primitive probe | Unsupported required extent is capability-miss. |
| `tl.reshape` | Supported | Element-count-preserving `(256,) -> (8,32)`, `(8,) -> (8,1)`, and `(8,32) -> (256,)`. | C500 primitive probe | Invalid shape is implementation repair. |
| `tl.broadcast_to` | Supported | `(8,1) -> (8,32)` float32 broadcast in the recorded probe. | C500 primitive probe | Unsupported required shape is capability-miss. |
| `tl.max` | Supported | Axis-1 reduction over `(8,32)` float32 values. | C500 primitive probe | Unsupported required reduction is capability-miss. |
| `tl.argmax` | Supported | Axis-0 reduction over eight unique float32 group maxima. Equal-value ordering is not established. | C500 primitive probe | Semantic tie mismatch is implementation repair or capability-miss according to the decision. |
| `tl.sum` | Supported | Scalar reduction over 256 float32 values. | C500 primitive probe | Unsupported required reduction is capability-miss. |
| `tl.exp` | Supported | Elementwise float32 exponential over 256 values. | C500 primitive probe | Unsupported required math is capability-miss. |
| `tl.where` | Supported | Float32 selection over `(8,32)` with a `-inf` sentinel. | C500 primitive probe | Incorrect masking is implementation repair. |
| `tl.static_range` | Supported | Compile-time loop with two iterations in the recorded probe. | C500 primitive probe | Unsupported required control construct is capability-miss. |

## Constrained Primitives

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| `num_warps` | Constrained | `num_warps=1` compiled and executed with target warp size 64. Other values are Unknown. | C500 primitive probe | Required unavailable values are capability-miss; optional tuning falls back to 1. |
| `tl.argmax` tie behavior | Constrained | Only unique maxima were checked. The project decision must preserve deterministic expert-id semantics for equal scores. | C500 primitive probe | Tie mismatch is implementation repair or design rejection. |
| reduction shapes and dtypes | Constrained | Only the recorded float32 extents and axes are proven. | C500 primitive probe | Unproven normative use is capability-miss. |

## Unsupported Primitives

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| `fast_libentry` | Unsupported | Neither `triton.runtime.fast_libentry` nor `triton.runtime.fast_libentry.fast_libentry` was importable in the matched runtime. Use direct launch. | C500 import probe | A normative fast-launcher requirement is capability-miss. |

## Unknown Primitives

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| `tl.zeros` | Unknown | No qualifying execution probe. | No qualifying C500 probe | Unprovable normative use is capability-miss. |
| `tl.full` | Unknown | No qualifying execution probe. | No qualifying C500 probe | Unprovable normative use is capability-miss. |
| `tl.dot` | Unknown | No qualifying execution probe for the required shape or dtype. | No qualifying C500 probe | Unprovable normative use is capability-miss. |
| `tl.make_block_ptr` | Unknown | Memory-placement and support semantics require a matched probe. | No qualifying C500 probe | Unprovable normative use is capability-miss. |
| `vectorize` | Unknown | Accepted values and backend effect require a matched probe. | No qualifying C500 probe | Unprovable normative use is capability-miss. |
| `async_copy` | Unknown | Availability, synchronization, and memory semantics require a matched probe. | No qualifying C500 probe | Unprovable normative use is capability-miss. |
| `num_stages` | Unknown | No C500-specific legality or performance probe is recorded. | No qualifying C500 probe | Unprovable normative use is capability-miss. |
| non-contiguous and mixed-precision behavior | Unknown | The primitive probe used contiguous float32 buffers only. | No qualifying C500 probe | Unprovable normative use is capability-miss. |

## Allowed Fallbacks

A fallback is allowed only when the decision leaves the choice non-normative
and preserves Optimization Intent, Unified Sketch, Host Plan, and Evaluation
Contract. The observed `num_warps=1` may replace an optional unproven tuning
value. Direct Triton launch may replace optional launcher tuning. Algorithm,
dataflow, output-buffer lifecycle, observable semantics, and concurrent model
behavior are not fallbacks; they require a new decision or are a major
deviation.

## Target-specific Pitfalls

- Importing Triton before setting `MACA_PATH` fails during backend discovery;
  this is an environment block, not a kernel compile failure.
- The CUDA-compatible PyTorch surface does not make CUDA target profiles or
  NVIDIA tuning assumptions valid for MACA.
- Target warp size is 64 on the recorded C500; do not infer warp-32 layouts or
  launch parameters from NVIDIA examples.
- Direct launch is the proven path. `fast_libentry` assumptions from MLU are
  invalid on the matched runtime.
- The harness loader and profiler must be validated through the actual
  `auto_bench.py` path, not direct import alone.
- Reusing output buffers requires an explicit Host Plan and cannot assume a
  model instance is safe across concurrent forwards or streams.

## Evidence Ledger

| Claim | Repository or run evidence | Scope |
|---|---|---|
| MACA runtime identity, direct launch, target arch 80, and warp size 64 were discovered. | Phase 0 C500 runtime probe and `rounds/report_000.md` reproduction commands | One matched MetaX C500 environment. |
| Loads, stores, indexing, reshape, broadcast, max, argmax, sum, exp, where, static loops, and one-warp launch executed with checked results. | Phase 0 C500 primitive probe | Exact recorded float32 shapes and launch only. |
| Profiler exported kernel-category events. | Phase 0 C500 profiler probe | One primitive kernel; full baseline trace remains project-specific. |
| Both known `fast_libentry` imports were unavailable. | Phase 0 C500 import probe | Recorded Triton package only. |
| Dot, block pointers, async copy, stages, other warp counts, non-contiguous, and mixed-precision behavior remain unproven. | This profile and absence of qualifying probes | Unknown is not treated as Unsupported or Supported. |
