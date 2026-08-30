# Coder Result 001

Result: `candidate-ready`

- round: `001`
- source_canonical: `baseline_adapter.py`
- source_canonical_sha256: `40c1a8bfbd9a0e957f21ae8ac686aa4c378a28299fd1f053d1e35b5fa8c443e0`
- decision: `rounds/design_001.md`
- decision_sha256: `fdf2f9f9a5660cea68d7546206f9767ddbbe94f61cba0ae34056b4c4b9825786`
- candidate: `triton_rotary_001.py`
- candidate_sha256: `51a9a33b82f550abfd80400bb0748b74fd181d0f3c4fd4b5d70b4ca1f5d6984e`
- selected_profile: `triton_ascend`
- runtime_fingerprint: `project.md#runtime-fingerprint`
- measurement_fingerprint: `pending` (Verifier-owned)

## Primitive and Hint Conformance

| Primitive / Hint | Status | Note |
|---|---|---|
| `tl.load` | Supported | Scalar + vector loads: `inv_freq[half]`, `position_angles[t, c-dim]`, `timestamps[b,t]`. |
| `tl.store` | Supported | Contiguous `cos_out`/`sin_out` `[N]` fp32 stores, masked by `idx < N`. |
| `tl.arange` | Supported | Extent `BLOCK=128` (verified extent in the ascend primitive probe). |
| `tl.program_id` | Supported | Axis 0 only (1D grid `(cdiv(N, BLOCK),)`). |
| `tl.minimum` / `tl.maximum` | Supported | Int32 index clamping for the out-of-bounds-side loads (discarded by `tl.where`). |
| `tl.where` | Supported | Masked selection of batch vs time frequency source. |
| `tl.cos` / `tl.sin` | Supported | Elementwise fp32 trig; compiled and executed correctly on Ascend (accuracy PASS). |
| `num_warps=1` | Constrained (proven) | Tuning fallback from the sketch's `num_warps=4`; 1/2/4 all proven on Ascend (see profile). |

No Unsupported or Unknown primitive is made normative. `tl.dot`, `tl.make_block_ptr`,
`async_copy`, `num_stages`, `fast_libentry`, and `vectorize` are not used. Direct
Triton launch (the proven Ascend launcher path) is used; no `import triton_ascend`
(metadata-only, non-importable). `get_inputs` uses `device="npu"`.

## Conformance Notes (candidate-ready)

Small target-language accommodations that preserve all normative semantics:

- **Blocked realization of the parallel loop.** The Unified Sketch writes
  `parallel idx over B*SEQ*2DIM` with `guard idx < B*SEQ*2DIM` and
  `num_warps=4`. A literal one-program-per-element realization (16384 scalar
  programs) is correct but leaves the kernel host/launch-bound (measured ~0.45x,
  slower than baseline). The sketch's own `num_warps` hint implies a blocked,
  multi-element program. The kernel therefore groups `BLOCK=128` contiguous flat
  output indices per program (`idx = pid*BLOCK + arange(0,BLOCK)`, `mask = idx < N`),
  which is the standard Triton realization of "parallel over N elements": every
  element is still computed by exactly one program-lane, the decode
  `(b, t, c)`, the `is_time` selection, the `-timestamps*2pi` angle, and the
  `cos`/`sin` writes are byte-for-byte the same operations. This changes no
  algorithm, dataflow, lifecycle, or Evaluation Contract; it only coalesces the
  per-element work into vectorized loads/stores. `BLOCK=128` equals one full
  `(b,t)` row (`2DIM=128`), so `c` spans `[0,128)` and the writes are fully
  contiguous.
- **`num_warps=1` instead of `num_warps=4`.** `num_warps` is an optional tuning
  hint (profile: "Constrained"; "optional tuning falls back to a proven value").
  For a 128-element block, `num_warps=1` (32 threads, 4 elems/thread) measured
  faster than 2 or 4 (see attempt ledger). All three values are proven to compile
  and run on Ascend.
- **In-bounds clamping of the discarded branch loads.** The sketch loads
  `inv_freq[half]` (`half = c//2`) and `position_angles[t, c-dim]` before the
  `is_time` selection. For `c >= dim`, `half` reaches `[dim/2, dim)` which is out
  of `inv_freq`'s `[dim/2]` range; for `c < dim`, `c-dim` is negative. The kernel
  clamps the load index into the valid range (`half_safe = min(half, dim//2-1)`,
  `c_minus_dim_safe = max(c-dim, 0)`) and discards the clamped value via
  `tl.where`. The selected branch value is bit-identical to the unclamped load, so
  this preserves exact semantics with no out-of-bounds access.
- **`TWO_PI` as a float32 kernel scalar.** `base.py` computes
  `(-timestamps * 2*math.pi)` in float64 then `.to(freqs)` back to float32. The
  kernel computes `-ts * TWO_PI` in float32 directly. The fp32 rounding of `2π`
  (~1e-7 relative) is negligible against `atol=rtol=1e-2`; accuracy passes.
- **`inv_freq` / `position_angles` remain `register_buffer`s**, precomputed once
  in `__init__` and passed to the kernel as input tensors (no per-forward
  recomputation), preserving the Phase 0 "model state, not per-forward" semantics.

## Local Gate

| Gate | Command | Exit | Result |
|---|---|---|---|
| `ast.parse` / `py_compile` | `python3 -m py_compile triton_rotary_001.py` | 0 | pass |
| Harness loader | real `auto_bench.py` AST loader (`_filter_module_ast`) | 0 | pass (kernel `FunctionDef`, `ModelNew` `ClassDef`, imports all retained) |
| Correctness + compile/warm-up smoke | `auto_bench.py --v0_file .../base.py --v1_file .../triton_rotary_001.py --warmup 1 --repeat 3 --full-traceback` | 0 | `PASS accuracy` |
| Steady-state smoke (BLOCK=128, num_warps=1) | `--warmup 50 --repeat 100` | 0 | `PASS accuracy; ~1.8x-2.0x speedup` (smoke, non-authoritative) |

The real harness loader retains the kernel's `@triton.jit` `FunctionDef`, the
`ModelNew` `ClassDef`, and all top-level imports; accuracy passes against the
reference under `atol=1e-2, rtol=1e-2` (tuple output compared element-wise:
`cos` then `sin`).

## Attempt Ledger

| Attempt | Command | Exit | Defect | Before SHA256 | After SHA256 |
|---:|---|---:|---|---|---|
| 1 | `py_compile triton_rotary_001.py` | 0 | none | - | scalar-per-program variant (16384 programs, num_warps=4) |
| 2 | `auto_bench.py ... --warmup 1 --repeat 3 --full-traceback` | 0 | none; accuracy PASS, but ~0.45x (slower) | (scalar variant) | (scalar variant) |
| 3 | `auto_bench.py ... --warmup 50 --repeat 100` | 0 | none; confirmed ~0.43x — scalar grid is launch-bound | (scalar variant) | (scalar variant) |
| 4 | rework to blocked kernel (BLOCK sweep 256/512/1024/2048, num_warps=4) | 0 | none; accuracy PASS, ~1.4x-1.8x | (scalar variant) | (blocked variant) |
| 5 | BLOCK sweep 64/128 + num_warps sweep 4/2/1 | 0 | none; best = BLOCK=128, num_warps=1 (~1.8x-2.0x) | (blocked variant) | `51a9a33b82f550abfd80400bb0748b74fd181d0f3c4fd4b5d70b4ca1f5d6984e` |

The defect at attempts 2-3 was a local realization issue (scalar grid left the
kernel launch-bound), not a semantic change: the blocked kernel computes the
identical elementwise chain over the identical elements. No algorithm, dataflow,
or lifecycle change was made; `inv_freq`/`position_angles` remain register_buffers
and the output tuple `(cos, sin)` of `[4,32,128]` fp32 is unchanged.

## Reason Code

`candidate-ready`: the candidate conforms to the immutable design (single fused
Triton kernel reading precomputed `inv_freq`/`position_angles` buffers and
`timestamps`, writing `cos` and `sin` directly, collapsing the 14-kernel elementwise
chain). Correctness and the local compile/warm-up smoke gate pass against the real
harness.
