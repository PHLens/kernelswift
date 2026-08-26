# Target Profile: triton_mlu

> **Machine-readable authority**: this Markdown page is an explanatory rendering
> of the target. The canonical machine-readable implementation profile is
> `profiles/triton_mlu/profile.yaml` under `skills/kernel-opt-loop/`. Its
> `capability_matrix`, `probe_catalog`, and reviewed `evidence/` records are the
> authoritative capability facts; descriptions below are historical human
> context and are never treated as approved probe evidence.

This is the MLU target profile. It records capabilities observed in this
repository and the checks required before using them. A run selects exactly one
matching profile; this profile is not a fallback for other backends. Absence from Supported is
not evidence of support.

## Identity and Match

The decision and manifest must say `language: triton`, `backend: mlu`, and
`target_profile: triton_mlu`. Phase 0 must discover the Triton distribution and
version, active driver/compiler backend target and version when available, and
MLU device architecture. A missing runtime or identity mismatch is
`environment-blocked`, not a capability miss. Repository evidence below does
not replace a match against the current project's runtime fingerprint.

## Runtime and Launcher Conventions

- Import torch MLU support before launching and preserve the caller-selected
  device and current stream.
- The selected interpreter requires runtime introspection before choosing an import form.
- The repository contains both
  `from triton.runtime import fast_libentry` and
  `from triton.runtime.fast_libentry import fast_libentry`.
- The harness AST loader may remove nonliteral module assignments. The observed
  compatible pattern initializes `fast_libentry()(_kernel)` from a retained
  class body when the loader requires it.
- Output caching and removal of a redundant device context are Host Plan
  patterns. They are not Sketch primitive translations and require explicit
  ownership, cache-key, invalidation, concurrency, device, and stream semantics.

## Supported Primitives

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| `tl.load` | Supported | Validate mask, bounds, dtype, and alignment for the project shape. | `mlu/groupedtopk/triton_grouped_topk_004.py`; `mlu/fused_moe/triton_fused_moe_005.py`; `mlu/flexattention/triton_flexattention_003.py` | Local misuse is implementation repair; runtime absence is environment-blocked. |
| `tl.store` | Supported | Preserve output shape, dtype, bounds, and aliasing contract. | Same three kernel files and their `log.md` files | Local misuse is implementation repair. |
| `tl.arange` | Supported | Extent and masking remain shape-specific. | Same three kernel files | Unsupported extent discovered locally is capability-miss. |
| `tl.program_id` | Supported | Grid mapping must preserve the decision's control structure. | Same three kernel files | Incorrect mapping is implementation repair. |
| `tl.dot` | Supported | Inputs are 2-D with matching inner dimensions; dtype and shape restrictions must be probed for the current runtime. | `mlu/fused_moe/triton_fused_moe_005.py`; `mlu/flexattention/triton_flexattention_003.py`; `mlu/flexattention/log.md` Entry 002 | Unavailable required shape/dtype is capability-miss. |
| `tl.argmax` | Supported | Tie behavior and masking must preserve project semantics. | `mlu/groupedtopk/triton_grouped_topk_004.py`; `mlu/groupedtopk/log.md` Entry 003 | Semantic mismatch is major-deviation or implementation repair according to design impact. |
| `tl.reshape` | Supported | Logical element count must be unchanged; this is not a storage-placement claim. | All three evidence kernels | Invalid shape is implementation repair. |
| `tl.zeros` | Supported | A value-producing tensor operation only; dtype and shape remain constrained by the current compiler. | `mlu/groupedtopk/triton_grouped_topk_004.py`; `mlu/fused_moe/triton_fused_moe_005.py` | Unsupported required shape/dtype is capability-miss. |

## Constrained Primitives

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| `num_warps` | Constrained | `num_warps=1` is locally used. `num_warps=2` failed in the flexattention experiment. Every other value is Unknown until probed on the matched runtime and architecture. | `mlu/groupedtopk/triton_grouped_topk_004.py`; `mlu/fused_moe/triton_fused_moe_005.py`; `mlu/flexattention/log.md` Entry 004 | A required unavailable value is capability-miss; optional tuning falls back to a proven value. |
| `num_stages` | Constrained | `num_stages=2` compiled and ran in the recorded experiment but produced less than 5% wall improvement; legality and value are architecture-specific. | `mlu/flexattention/triton_flexattention_004.py`; `mlu/flexattention/log.md` Entry 004 | Compile failure is capability-miss; poor performance is Verifier evidence. |

## Unsupported Primitives

No primitive is declared universally Unsupported by current repository
evidence. A matched local probe may establish an Unsupported result for the
current profile revision; until then unproven items remain Unknown.

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| None established | Unsupported | Do not invent entries from other Triton vendors. | Repository evidence review | A normative unsupported requirement is capability-miss. |

## Unknown Primitives

| Primitive | Status | Constraint | Evidence | Failure classification |
|---|---|---|---|---|
| `tl.make_block_ptr` | Unknown | Do not infer memory placement or support. A local MLU compile and execution probe is required. | No qualifying repository probe | Unprovable normative use is capability-miss. |
| `vectorize` | Unknown | Meaning, accepted values, and backend effect require a local probe. | No qualifying repository probe | Unprovable normative use is capability-miss. |
| `async_copy` | Unknown | Availability, synchronization, and memory semantics require a local probe. | No qualifying repository probe | Unprovable normative use is capability-miss. |

## Allowed Fallbacks

A fallback is allowed only when the decision leaves the choice non-normative and
the fallback preserves Optimization Intent, Unified Sketch, Host Plan, and the
Evaluation Contract. Proven `num_warps=1` may replace an optional unproven tuning
value. Ordinary Triton launch may replace optional `fast_libentry` only when the
Host Plan does not require launcher reduction. Any algorithm, dataflow, buffer
lifecycle, or observable change is `major-deviation`, not a fallback.

## Target-specific Pitfalls

- The harness may strip module-level launcher construction; validate with the
  actual loader rather than direct import alone.
- `fast_libentry` import location varies between observed environments.
- A `tl.dot` port must preserve two-dimensional shapes and accepted dtypes.
- Argmax sentinels must not contribute nonzero indices to reductions.
- `num_warps=2` failed in the recorded flexattention runtime; do not generalize
  settings from CUDA Triton.
- Reusing an output buffer requires a complete Host Plan and cannot assume a
  model instance is shared safely across concurrent forwards.
- Removing `torch.mlu.device()` is valid only when the caller already owns device
  selection and current-stream behavior.

## Evidence Ledger

| Claim | Repository evidence | Scope |
|---|---|---|
| Basic loads, stores, indexing, reshape, argmax, and fast launcher execute on the recorded MLU setup. | `mlu/groupedtopk/triton_grouped_topk_004.py`; `mlu/groupedtopk/log.md` | Recorded grouped-top-k shapes and runtime only. |
| Dot, zeros, loads, stores, and one-warp launch execute on the recorded MLU setup. | `mlu/fused_moe/triton_fused_moe_005.py`; `mlu/fused_moe/log.md` | Recorded fused-MoE shapes and runtime only. |
| Dot attention and host-side launcher/cache patterns execute on the recorded MLU setup. | `mlu/flexattention/triton_flexattention_003.py`; `mlu/flexattention/log.md` | Recorded flexattention shapes and runtime only. |
| Two stages ran but did not clear adoption threshold; two warps failed. | `mlu/flexattention/triton_flexattention_004.py`; `mlu/flexattention/log.md` Entry 004 | Evidence for the recorded runtime; other architectures remain Unknown. |
