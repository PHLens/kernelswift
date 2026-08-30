# Coder Result 001

## Result

- result: `candidate-ready`
- round: `001`
- reason_code: `null`

## Source and Decision

- source_canonical: `baseline_adapter.py` (last_accepted_kernel)
- source_canonical_sha256: `2c0c121e2e4504e791fee3675ac1cd54d1322059fc537555578a91cabd2a24e6`
- decision_path: `rounds/decision_001.md`
- decision_sha256: `9f3795f57808c0ada1ffaa6c02cfea507f9026cd8a09684987f0e30d3074da5a`
- selected_profile: `triton_maca`
- runtime_fingerprint_ref: `project.md#runtime-fingerprint`

## Candidate

- candidate_path: `kernels/track1-triton/mhc_post_layer_mix/maca/triton_mhc_001.py`
- candidate_sha256: `e54e5b2e553449134eb3b6679d6ed759e30fd2dd42499f9a21716ae57216b944`

## Conformance Notes

1. **comb_res_mix indexing (m/n subscript ordering)** — The decision prose and the
   sketch write the contraction as `comb_res_mix[b,p,n,m] * residual[b,p,m,h]`.
   This transposes `comb_res_mix` relative to the actual `base.py` einsum
   `'abmn,abmc->abnc'`, which maps `comb_res_mix` dims to `(a,b,m,n)` (dim2 = m,
   dim3 = n). Numerical verification against `torch.einsum` confirmed the
   correct lowering is `comb_res_mix[b,p,m,n] * residual[b,p,m,h]` (sum over
   `m`, dim2). The candidate implements `comb_res_mix[b,p,m,n]` to reproduce
   `base.py` exactly. This is a conformance note, not a semantic deviation: the
   harness compares against `base.py`, and `base.py` is the normative source of
   truth. The candidate passes the harness accuracy gate.

2. **Primitive usage** — Uses only `tl.load`, `tl.store`, `tl.arange`,
   `tl.static_range`, and scalar int/float arithmetic. No `tl.dot`, no
   block-ptr, no async-copy, no `tl.zeros`/`tl.full` (accumulator built via
   `tl.zeros`-free `tl.arange`-shaped scalar product). `num_warps=1` honored.
   `tl.arange(0, BLOCK)` with `BLOCK=1024` — extent 1024 is not explicitly
   listed in the probe's proven extents (256, 8), but `tl.arange` is a
   universally supported Triton primitive and the decision explicitly permits
   `BLOCK` of 1024/2048; the harness smoke compiled and ran correctly.

3. **Fast-path guard** — Guard requires the exact benchmark shapes/dtypes,
   contiguity, `cuda` device, and no-grad (or non-requires-grad inputs). The
   fallback is the unchanged `torch.einsum` path copied verbatim from
   `baseline_adapter.py`.

4. **AST loader retention** — All required entry points (`ModelNew`,
   `get_inputs`, `get_init_inputs`) are top-level `ClassDef`/`FunctionDef`
   nodes, retained by the harness `_filter_module_ast`. No non-literal top-level
   assignments are used.

## Attempt Ledger

| # | Command | Exit | Defect | Before SHA-256 | After SHA-256 |
|---|---|---|---|---|---|
| 1 | `ast.parse` + harness `_filter_module_ast` retention check | 0 | none | `null` | `e54e5b2e553449134eb3b6679d6ed759e30fd2dd42499f9a21716ae57216b944` |
| 2 | `auto_bench.py --warmup 2 --repeat 3 --full-traceback` | 0 | none (accuracy PASS) | `e54e5b2e553449134eb3b6679d6ed759e30fd2dd42499f9a21716ae57216b944` | `e54e5b2e553449134eb3b6679d6ed759e30fd2dd42499f9a21716ae57216b944` |

## Local Gate Evidence

- `ast.parse`: PASS (no syntax error).
- Harness loader retention: `ModelNew`, `get_inputs`, `get_init_inputs` retained
  as top-level defs.
- Harness compile + correctness smoke:
  ```
  PASS accuracy; v0=7.656202 ms, v1=0.254279 ms, speedup=30.109x
  Summary: 1 passed, 0 failed, 1 total.
  ```
  (wall numbers are smoke-only; authoritative benchmark belongs to Verifier.)

## Notes

- Coder never returns `accepted`; this is `candidate-ready` only. Authoritative
  runtime evidence is Verifier's responsibility.
