# Coder Result 001

## Identity

- role: Coder
- round: "001"
- project: mhc_head_compute_mix (bi150, track1-triton)
- target_profile: triton_cuda
- runtime_fingerprint_ref: project.md#runtime-fingerprint
- result: `candidate-ready`

## Source and Decision

- source canonical (last_accepted_kernel): `baseline_adapter.py`
- source canonical SHA-256: `ceebdc6185de4c980156a7833073678a0964fb7ccb5edf74b42be6156652eaed`
- decision: `rounds/decision_001.md`
- decision validation: `valid=true` (via `validate_decision.py --expected-profile triton_cuda`)
- base (immutable reference): `../base.py`, SHA-256 `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5`

## Candidate

- path: `triton_mhc_head_compute_mix_001.py`
- SHA-256: `a98b1b12593d858ca29c787afa939a3ae0061df4ec6b51aa9a0fe7fa43c6b473`

## Implementation

A single fused Triton kernel `_mhc_head_compute_mix_kernel` replaces the entire
`ModelNew.forward` dataflow (head-compute pre/post elementwise + the 20-round
Sinkhorn iteration), collapsing the ~133 per-call tiny-kernel launches into one
kernel over a 16-program 1D grid.

### Kernel structure

- One program per `(b, s)` position (`n = tl.program_id(0)`, N = 16, grid = (16,)).
- Loads `hc_scale` (3 scalars), the per-element `mixes[n, :]` slice (`pre [0:4]`,
  `post [4:8]`, `comb [8:24]`) and the corresponding `hc_base` slices with
  `tl.arange(0, 4)` and `tl.load`.
- Computes and stores `pre` (`sigmoid(x0*s0 + base0) + eps`) and
  `post` (`2*sigmoid(x1*s1 + base1)`) in the same kernel.
- Builds the `[4,4]` `comb` tile (`comb_x * s2 + comb_b`) entirely in registers
  via `rows = idx[:, None]`, `cols = idx[None, :]` indexing.
- Stable softmax: `row_max = tl.max(comb, axis=1, keep_dims=True)`;
  `comb = tl.exp(comb - row_max)`.
- Sinkhorn normalization stays in registers for all 20 rounds; the `[4,4]` tile
  never round-trips to global memory between normalization rounds.
- Stores `comb` to the `[16,4,4]` output buffer.

### eps asymmetric placement (correctness-critical)

The exact base.py semantics are preserved:

- First explicit row-normalize: `comb = comb / row_sum + eps`
  (eps added to the **matrix**, every element +eps).
- First explicit col-normalize: `comb = comb / (col_sum + eps)`
  (eps added to the **denominator**).
- All 19 looped row/col normalizations: `comb = comb / (sum + eps)`
  (eps always in the **denominator**).

### Loop construct: static_range -> tl.range (conformance note)

The decision's Optimization Intent specified a compile-time `tl.static_range`
loop (19 iterations). Empirically on the BI150 / CoreX Triton 3.1.0 runtime:

- `tl.static_range(4)` compiles ~instantly (the only count the profile had
  verified as Supported).
- `tl.static_range(8)` compiles in ~100s.
- `tl.static_range(19)` does **not** complete within 300s (compile-time unroll
  blow-up), leaving the harness smoke effectively hung.

This is the exact capability risk the decision pre-identified ("the primary
capability risk"). I fell back to the dynamic `tl.range(ITERS)` loop, which is
explicitly listed as an allowed degradation in the task brief ("可降级为动态循环
tl.range 或减少融合范围"). The dynamic loop preserves the exact algorithm,
dataflow, normalization count (19), and eps placement; it changes only the loop
*control construct* (compile-time unroll vs runtime loop), not any normative
semantics. The `[4,4]` `comb` tile remains in registers across all iterations
either way. This is recorded as a conformance note, not a major deviation.

## Gate Evidence

| Gate | Command | Result |
|---|---|---|
| decision validation | `validate_decision.py .../decision_001.md --expected-profile triton_cuda` | `valid=true` |
| AST syntax check | `python3 -m py_compile triton_mhc_head_compute_mix_001.py` | PASS |
| harness loader (real) | `auto_bench.py` (AST loader `_filter_module_ast` retains `@triton.jit` FunctionDef) | PASS (loaded, compiled, launched) |
| harness smoke (accuracy + timing) | `auto_bench.py --v0_file base.py --v1_file triton_..._001.py --warmup 50 --repeat 100 --full-traceback` | **PASS accuracy**; v0=1.424529 ms, v1=0.180098 ms, speedup=7.910x |

## Attempt Ledger

| # | Command | Exit | Defect | Before SHA | After SHA |
|---|---|---|---|---|---|
| 1 | harness smoke with `tl.static_range(ITERS)` (ITERS=19) | timeout (>300s, EXIT 124) | compile-time unroll blow-up at 19 iterations | (initial write) | (unchanged) |
| 2 | isolated probe `tl.static_range(4)` | 0 | none | - | - |
| 3 | isolated probe `tl.static_range(8)` | 0 (~99.7s) | compile time grows superlinearly with unroll count | - | - |
| 4 | isolated probe `tl.range(19)` dynamic | 0 | none | - | - |
| 5 | edit: `tl.static_range(ITERS)` -> `tl.range(ITERS)` + note | 0 | none | (pre-edit) | `a98b1b12...` |
| 6 | harness smoke (final candidate) | 0 | none | - | `a98b1b12...` (PASS) |

## Conformance

- Primitive conformance (vs `triton_cuda` profile): all primitives used are
  Supported — `tl.load`, `tl.store`, `tl.arange`, `tl.program_id`, `tl.max`,
  `tl.sum`, `tl.exp`, `tl.sigmoid` (elementwise fp32), and 2D broadcast indexing
  via `idx[:, None]`/`idx[None, :]` (reshape-backed register indexing, consistent
  with the profile's constrained masking/reshape evidence). `tl.range` is a
  standard Triton runtime loop, not an unproven construct. No Unknown primitive
  (`num_warps`, `num_stages`, block pointers, mixed precision, gather/scatter) is
  required or introduced.
- `tl.static_range` at 19 iterations was attempted first (the decision's
  normative construct) and is the sole degraded item; the fallback is explicitly
  authorized and preserves all normative semantics.
- Invariants preserved: `ModelNew(hc_mult=4, sinkhorn_iters=20, eps=1e-6)`
  public constructor; forward signature `(mixes, hc_scale, hc_base) ->
  (pre, post, comb)`; output shapes `pre/post [2,8,4]`, `comb [2,8,4,4]`, all
  fp32; exact 20-round Sinkhorn semantics with asymmetric eps placement; inputs
  not mutated (read-only loads, outputs written to fresh buffers); caller-selected
  device/stream preserved (outputs allocated on `x.device`, no stream/context
  manipulation).
- No `base.py`, `baseline_adapter.py`, `decision_001.md`, `team-state.md`,
  `project.md`, or harness files were modified.

## Handoff

- Result: `candidate-ready`
- Next role: Verifier (authoritative runtime measurement, Level 1 device-time and
  kernel-count observables to confirm the fusion dropped kernel_count_per_call
  and device_us_per_call as the Evaluation Contract expects).
- Candidate is NOT adopted by Coder; adoption requires Verifier evidence and
  Orchestrator decision.
