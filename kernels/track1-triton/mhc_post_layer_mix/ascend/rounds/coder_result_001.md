# Coder Result 001

- round: `001`
- result: `candidate-ready`
- change_family: `kernel-fusion`
- bottleneck_class: `device-bound`

## Source and Decision

| Artifact | Path | SHA-256 |
|---|---|---|
| canonical source (last accepted) | `baseline_adapter.py` | `a4f0aa8ac2d59c57059223b1710d20718af1b0f892cd7c373174e531c927133e` |
| immutable base (v0 reference) | `../base.py` | `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3` |
| decision | `rounds/decision_001.md` | `6c9bf2b10c30b3a1205fe3c94f3ba4b6dc8abe1f2753b41536bdf7e29acb32ad` |

## Candidate

| Field | Value |
|---|---|
| path | `candidate_001.py` |
| SHA-256 | `b74e407348d424c9265ddf831b245cda90297a48bdbaa576fa7e6b57b5d121f9` |
| target_profile | `triton_ascend` |
| runtime fingerprint | `project.md#runtime-fingerprint` (Ascend910B4, torch_npu 2.7.1.post4, triton 3.2.0) |

## Implementation Summary

Single Triton kernel `mhc_fused_kernel` fuses the six baseline kernels
(3 Cast + BatchMatMul + Add + Mul) into one launch.

- Grid: `(n_batch,)` where `n_batch = n0*n1 = 8192`; each program owns one
  `(a,b)` pair and loops over all c-blocks with a compile-time `tl.static_range`
  (`BLOCK_C=256`, `C=1280`, 5 iterations).
- The small per-`(a,b)` weight tiles `comb_res_mix[4,4]` (fp32) and
  `post_layer_mix[4,1]` (fp32) are loaded exactly once, outside the c-loop.
- `x` (bf16) and `residual` (bf16) are loaded once and cast to fp32 in registers.
- The contraction over `m=4` uses an **explicit 4-way fp32 FMA reduction** (no
  `tl.dot`, per the decision: `m=4` is below the probed `(16,16)@(16,16)` shape).
  The einsum `'abmn,abmc->abnc'` contracts over `m`, so the weight is
  `cm[m,n]`; loaded as four contiguous rows `cm[m,:]`.
- Single fp32→bf16 cast, then store `out[4,BLOCK_C]`.

## Primitive / Hint Conformance

| Primitive / hint | Decision requirement | Used | Status |
|---|---|---|---|
| matmul lowering | explicit 4-way fp32 reduction (default); `tl.dot` only as non-normative fallback | explicit reduction (broadcast mul + add), no `tl.dot` | conform |
| `tl.load` | masked contiguous loads | contiguous loads; no mask (C % BLOCK_C == 0, bounds exact) | conform |
| `tl.store` | contiguous stores | contiguous store | conform |
| `tl.arange` | extents 64/128/256 verified | extents 4 and 256 | conform |
| `tl.static_range` | compile-time loop (K=8 verified) | 5 iterations over c | conform |
| `num_warps` | proven 1/2/4 | 4 | conform |
| `tl.dot`, `num_stages`, `vectorize`, `make_block_ptr`, `async_copy` | Unknown / not required | not used | conform |

## Numerical Guardrail

- fp32 accumulation before a single bf16 cast (matches reference).
- Harness correctness `PASS` at atol=1e-2 / rtol=1e-2.

## Local Gate

| Gate | Command | Status |
|---|---|---|
| ast.parse | `ast.parse(candidate_001.py)` | PASS |
| harness loader | `auto_bench.load_ks_module(candidate_001.py)` (AST filter + device rewrite) | PASS |
| warmup/compile smoke | `auto_bench.py --v0_file base.py --v1_file candidate_001.py` | PASS accuracy |

## Attempt Ledger

| # | Command (summary) | Exit | Defect | Before hash | After hash |
|---|---|---|---|---|---|
| 1 | `auto_bench.py --warmup 5 --repeat 10` (2D grid, BLOCK_C=256) | 0 | correctness fail: max_abs_diff=26.66 (contraction orientation: used `cm[n,m]` instead of `cm[m,n]`) | `(initial)` | — |
| 2 | fix weight orientation (contiguous `cm[m,:]` rows); re-run smoke | 0 | correctness PASS but 0.926x (redundant per-c-block weight loads + 40960 tiny programs) | — | `(intermediate)` |
| 3 | restructure to loop-over-c, grid `(8192,)`, load weights once | 0 | none | `(intermediate)` | `b74e4073...` |

## Final Smoke Result (informational; authoritative timing is Verifier's)

- `--warmup 50 --repeat 100`: `PASS accuracy; v0=3.206825 ms, v1=0.880515 ms, speedup=3.642x`
- Device-bound fusion removes the 3 Cast kernels + Add + Mul materialization and
  the separate BatchMatMul launch, collapsing 6 kernels to 1.

## Reason Code

`candidate-ready` — candidate conforms to the immutable design (kernel-fusion,
explicit fp32 reduction, single bf16 cast, `ModelNew` public contract preserved).
No major deviation; no capability-miss; environment fingerprint matches.
