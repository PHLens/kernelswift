# Coder Result 002

Result: `candidate-ready`

- round: `002`
- source_canonical: `triton_rotary_001.py`
- source_canonical_sha256: `51a9a33b82f550abfd80400bb0748b74fd181d0f3c4fd4b5d70b4ca1f5d6984e`
- decision: `rounds/design_002.md`
- decision_sha256: `ee54366c2034343ad6206e02f9c8dd0d6340178d66ef6d70283e78932f61c4d0`
- candidate: `triton_rotary_002.py`
- candidate_sha256: `98e86e48f00a25254561ad5bf8ef91824c87e5bb9dfa0279f221633d910b07ab`
- selected_profile: `triton_ascend`
- runtime_fingerprint: `project.md#runtime-fingerprint`
- measurement_fingerprint: `pending` (Verifier-owned)

## Primitive and Hint Conformance

| Primitive / Hint | Status | Note |
|---|---|---|
| `tl.load` | Supported | Scalar `timestamps[b*SEQ+t]`; vector `batch_freq_base[0:DIM]`, `position_angles[t,0:DIM]` (each DIM-wide, loaded exactly once). |
| `tl.store` | Supported | Contiguous `cos_out`/`sin_out` half-row stores: `[0:DIM]` and `[DIM:2DIM]`. |
| `tl.arange` | Supported | Extent `DIM=64` (verified extent in the ascend primitive probe). |
| `tl.program_id` | Supported | Axis 0 only (1D grid `(B*SEQ,)`). |
| `tl.cos` / `tl.sin` | Supported | Elementwise fp32 trig; compiled and executed correctly on Ascend (accuracy PASS). |
| `num_warps=4` | Constrained (proven) | Sketch hint; re-swept 1/2/4 (see attempt ledger); all compile/run, 4 retained as within noise of 2. |

No Unsupported or Unknown primitive is made normative. `tl.dot`, `tl.where`,
`tl.make_block_ptr`, `async_copy`, `num_stages`, `fast_libentry`, and `vectorize`
are not used. No `tl.where` branch remains (the Round 1 select is removed by
writing the two 64-wide halves directly). Direct Triton launch used; no
`import triton_ascend`. `get_inputs` uses `device="npu"`.

## Conformance Notes (candidate-ready)

Small target-language accommodations that preserve all normative semantics:

- **Row-per-program grid (the design's own restructure).** `grid = (B*SEQ,)`, each
  program owns one output row of `2*DIM=128` columns. `b = pid // SEQ` and
  `t = pid % SEQ` are computed once per program as scalars (not per-lane). This
  removes all per-lane integer division/modulo; the only remaining scalar
  div/mod is one each per program. The `angle = -ts*2π` scalar is likewise
  computed once per program and broadcast to the row — valid because in
  `base.py` the angle depends only on `(b, t)`, never on the column `c`.
- **Two contiguous 64-wide halves loaded exactly once each.** `batch_freq_base[0:DIM]`
  and `position_angles[t,0:DIM]` are loaded as single `tl.arange(0,DIM)` vectors;
  each of the 128 frequencies is loaded once (Round 1 loaded `2×16384` and
  discarded half via `tl.where`). The two output halves are written directly
  (`cos_out[row_base+0:DIM]` and `cos_out[row_base+DIM:2DIM]`), with no `tl.where`.
- **`batch_freq_base` register_buffer (interleaved `inv_freq`, `[dim]`).** Added in
  `__init__` as `inv_freq.repeat_interleave(2)`, mirroring the existing
  `position_angles` precompute, to eliminate the kernel's `c // 2` integer
  division. It is registered `persistent=False` because it is a pure derived cache
  of `inv_freq` (no learned state). This keeps it a `register_buffer` (moves with
  `.to(npu)`, participates in `_apply`) while keeping `state_dict()` identical to
  the reference `{inv_freq, position_angles}`, so the harness's
  `model_new.load_state_dict(model.state_dict())` (strict) still matches. All three
  frequency tables remain precomputed once in `__init__`, preserving the
  "model state, not per-forward" invariant.
- **`num_warps=4` (sketch hint) retained after re-sweep.** A 1/2/4 sweep found all
  three within measurement noise (~0.31-0.34 ms candidate median); 4 was retained
  as it matched the sketch and was not worse than 2.
- **`TWO_PI` as a float32 kernel scalar** (unchanged from Round 1): `base.py`
  computes the angle in float64 then casts to float32; the kernel computes in
  float32 directly, negligible against `atol=rtol=1e-2`. Accuracy passes.

## Local Gate

| Gate | Command | Exit | Result |
|---|---|---|---|
| `ast.parse` / `py_compile` | `python3 -m py_compile triton_rotary_002.py` | 0 | pass |
| Harness loader | real `auto_bench.py` AST loader (`_filter_module_ast`) | 0 | pass (kernel `FunctionDef`, `ModelNew` `ClassDef`, imports all retained) |
| Correctness + compile/warm-up smoke | `auto_bench.py --v0_file .../base.py --v1_file .../triton_rotary_002.py --warmup 1 --repeat 3 --full-traceback` | 0 | `PASS accuracy` |
| Steady-state smoke (num_warps=4) | `--warmup 50 --repeat 100` | 0 | `PASS accuracy; ~1.75x-1.88x` (smoke, non-authoritative) |

The real harness loader retains the kernel's `@triton.jit` `FunctionDef`, the
`ModelNew` `ClassDef`, and all top-level imports; accuracy passes against the
reference under `atol=1e-2, rtol=1e-2` (tuple output compared element-wise:
`cos` then `sin`).

## Attempt Ledger

| Attempt | Command | Exit | Defect | Before SHA256 | After SHA256 |
|---:|---|---:|---|---|---|
| 1 | `py_compile triton_rotary_002.py` | 0 | none | - | row-per-program variant, num_warps=4 |
| 2 | `auto_bench.py ... --warmup 50 --repeat 100` (num_warps=4) | 0 | none; accuracy PASS ~1.75-1.91x | (num_warps=4) | (num_warps=4) |
| 3 | num_warps=2 sweep (`--warmup 50 --repeat 100` ×3) | 0 | none; v1 ≈ 0.328-0.332 ms | (num_warps=4) | (num_warps=2) |
| 4 | num_warps=1 sweep | 0 | none; v1 ≈ 0.336 ms | (num_warps=2) | (num_warps=1) |
| 5 | num_warps=4 sweep (×3) | 0 | none; v1 ≈ 0.312-0.337 ms | (num_warps=1) | (num_warps=4) |
| 6 | revert to num_warps=4 + final correctness/warm-up smoke | 0 | none; accuracy PASS | (num_warps=4) | `98e86e48f00a25254561ad5bf8ef91824c87e5bb9dfa0279f221633d910b07ab` |

No semantic defect occurred. The num_warps sweep (attempts 3-5) showed 1/2/4 all
within noise (~0.31-0.34 ms); 4 was retained as the sketch's hint and was not
worse than 2. The restructure removed the per-lane integer division, the dual
frequency load, and the `tl.where` select, with no change to the numerical result,
dataflow, lifecycle, or Evaluation Contract.

## Reason Code

`candidate-ready`: the candidate conforms to the immutable design (row-per-program
fused kernel that scalarizes `b/t/ts/angle` per program, loads each frequency once
as two contiguous 64-wide halves via a precomputed `batch_freq_base` buffer, and
writes `cos`/`sin` with no `tl.where`). Correctness and the local compile/warm-up
smoke gate pass against the real harness.
