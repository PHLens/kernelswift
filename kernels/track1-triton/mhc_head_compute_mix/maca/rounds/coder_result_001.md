# Coder Result 001

- result: `candidate-ready`
- round: `001`
- source_canonical_path: `baseline_adapter.py`
- source_canonical_sha256: `c3cc90deb93c69f092e876ddc0057ddf6c2d7ac28020f1b7fd1ed260bca72fee`
- decision_path: `rounds/decision_001.md`
- decision_sha256: `<decision_001.md>`
- selected_profile: `triton_maca`
- runtime_fingerprint_ref: `project.md#runtime-fingerprint`
- candidate_path: `triton_mhcc_001.py`
- candidate_sha256: `f29b71c87712aa3f674c2ec6e448bf1026a81986ecaa070645e691d66e969c39`
- reason_code: `candidate-ready`

## Conformance Notes

### Optimization Intent and Sketch
Fused the entire MHCHeadComputeMix forward into a single Triton kernel, one
program per `(b, s)` position (16 programs). `grid = (n,)` with `n = 16`,
`num_warps=1`, direct launch `kernel[(grid,)](...)`. The Sinkhorn loop is a
genuine `for _ in tl.static_range(sinkhorn_iters - 1)` (19 additional pairs)
preceded by the unconditional first row/col pair, matching `base.py`'s exact
loop structure (1 first pair + 19 subsequent pairs = 20 total).

### Exact fp32 math (eps placement asymmetry)
Reproduced line-by-line from `base.py`:
- `pre[i] = sigmoid(x[i]*s0 + base[i]) + eps` (eps AFTER sigmoid).
- `post[i] = 2*sigmoid(x[hc+i]*s1 + base[hc+i])`.
- `comb = x[8+j*4+i]*s2 + base[8+j*4+i]`; `row_max = max_i(comb)`; `comb = exp(comb - row_max)`.
- FIRST row normalize: `comb = comb/row_sum + eps` (eps AFTER division) — the special case.
- FIRST col normalize: `comb = comb/(col_sum + eps)` (eps INSIDE denominator).
- 19 subsequent pairs: `comb = comb/(row_sum + eps)` then `comb = comb/(col_sum + eps)` (eps INSIDE denominator both).
- sigmoid implemented as `1.0 / (1.0 + tl.exp(-v))` — identical to `torch.sigmoid`.

### Primitive conformance (triton_maca profile)
Used only proven primitives:
- `tl.load` / `tl.store` — scalar contiguous fp32 (Supported).
- `tl.exp` — elementwise fp32 (Supported).
- `tl.maximum` — scalar max over 4 elements, avoiding unproven 4-element `tl.max`/`tl.sum` reductions (manual reduction used, per decision guidance).
- `tl.static_range` — compile-time loop (Supported).
- `num_warps=1` (Constrained, only proven value).
- `tl.program_id` — scalar program id.
- scalar `+ - * /` math on fp32 scalars.
- NO `tl.dot`, NO `tl.sum`/`tl.max` vector reductions, NO `tl.zeros`/`tl.full`, NO `tl.arange` (avoided because arange requires power-of-2 extents on this runtime), NO `tl.make_block_ptr`/`async_copy`.

### Deviations / accommodations (conformance notes, not design changes)
1. The decision sketch suggested `tl.arange`/vector tiles and `tl.sum` over 4
   elements. On this runtime `tl.arange` requires a power-of-2 range (24 and 4
   elements are not power-of-2), and 4-element `tl.sum` is unproven. Per the
   decision's explicit direction ("use explicit manual reduction over the
   4-element dims"), the kernel uses fully-unrolled scalar registers
   (`c00..c33`) with manual `+`/`tl.maximum` reductions. This preserves all
   normative semantics (algorithm, dataflow, eps placement, loop count) and is a
   conformance note, not a design revision.
2. The Sinkhorn loop body is unrolled per-iteration (the `static_range` controls
   the 19-pair repetition; each pair's row/col steps are written explicitly).
   This is semantically identical to a nested `static_range(hc)` loop and
   preserves the exact 20 total normalization pairs.

### Public contract
`ModelNew(hc_mult=4, sinkhorn_iters=20, eps=1e-6)`, `forward(mixes, hc_scale,
hc_base) -> (pre, post, comb)`, plus `get_inputs` and `get_init_inputs` all
retained by the AST loader (verified via `load_ks_module`).

### Guardrails
- Inputs not mutated (kernel reads only; outputs are fresh `torch.empty`).
- Caller device/stream preserved: outputs allocated on `x.device`, kernel
  launched on current stream (no stream switching, no device context removal).
- Fast-path guard: `hc==4`, `mix_hc==24`, `sinkhorn_iters==20`, `eps==1e-6`,
  fp32, contiguous cuda, no grad. Otherwise falls back to the unchanged PyTorch
  reference path (copied verbatim from `baseline_adapter.py`).

## Local Gate

1. `ast.parse` on candidate: `ast.parse OK` (exit 0).
2. Harness loader smoke (compile + correctness):
   ```
   cd /root/kernelswift-mhcc && /opt/conda/bin/python auto_bench.py \
     --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py \
     --v1_file kernels/track1-triton/mhc_head_compute_mix/maca/triton_mhcc_001.py \
     --warmup 2 --repeat 3 --full-traceback
   ```
   Output: `PASS accuracy; v0=1.768580 ms, v1=0.139910 ms, speedup=12.641x`
   `Summary: 1 passed, 0 failed, 1 total.` (exit 0)
3. AST loader retention: `ModelNew` (type), `get_inputs` (function),
   `get_init_inputs` (function) all present.

## Attempt Ledger

| Attempt | Command | Exit | Defect | Before SHA256 | After SHA256 |
|---:|---|---|---|---|---|
| 1 | ast.parse + harness smoke | 1 | `tl.arange` requires power-of-2 range (24 not power-of-2) | `n/a` | `n/a` (vector arange approach) |
| 2 | ast.parse + harness smoke | 1 | `ListComp` unsupported AST node in Triton | `n/a` | `n/a` (list-comprehension approach) |
| 3 | ast.parse + harness smoke | 0 | (none) — scalar unrolled registers | `n/a` | `f29b71c87712aa3f674c2ec6e448bf1026a81986ecaa070645e691d66e969c39` |

Attempts 1 and 2 were non-semantic (compiler/language-construct
accommodations) and did not change the algorithm, dataflow, eps placement, or
loop count. Attempt 3 is the final candidate.
