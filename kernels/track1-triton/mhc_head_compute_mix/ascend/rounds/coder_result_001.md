# Coder Result 001

Result: `candidate-ready`

## Identity

- Round: `001`
- Result: `candidate-ready`
- Decision: `rounds/decision_001.md` (`proceed`)
- Decision SHA256: `cfce60f6110bb21802b878f61a6238d89fed0320835560d2cfbd723107b881ef`
- Source canonical (`last_accepted_kernel`): `baseline_adapter.py`
- Source canonical SHA256: `c3cc90deb93c69f092e876ddc0057ddf6c2d7ac28020f1b7fd1ed260bca72fee`
- Candidate: `candidate_001.py`
- Candidate SHA256: `3eda8a14dede15a91f1a04c37bc5ff178a83fc87ecb7137b3569756c17f94f10`
- Selected profile: `triton_ascend`
- Runtime fingerprint: `project.md#runtime-fingerprint` (Ascend910B4, torch_npu 2.7.1.post4, triton 3.2.0)

## Optimization Intent Realized

- change_family: `sinkhorn-loop-fusion`
- bottleneck_class: `host-bound` (device_ratio ≈ 0.082, 136 kernels/call)
- intervention: entire forward collapsed into a single Triton kernel over a
  16-program grid (one program per `(b,s)` row), with the 20-iteration Sinkhorn
  row/column normalization inside a compile-time `tl.static_range` loop.
  136 → 1 kernel launch.

## Sketch / Primitive Conformance

| Primitive / Hint | Decision requirement | Implementation | Verdict |
|---|---|---|---|
| grid | 16 programs, one per `(b,s)` | `grid = (n_rows,)`, `n_rows = b*s = 16` | conform |
| `tl.static_range` | compile-time Sinkhorn loop (no relaunch) | `for _ in tl.static_range(0, SINKHORN_ITERS - 1)` = 19 iterations | conform |
| `tl.sum` axis | row/col 4x4 reductions | `tl.sum(comb, axis=1)` (row), `tl.sum(comb, axis=0)` (col) | conform |
| `tl.max` axis=1 | row-stabilized softmax | `row_max = tl.max(comb, axis=1)` | conform |
| `tl.exp` | softmax numerator | `comb = tl.exp(comb - row_max[:, None])` | conform |
| `tl.reshape` | `(16,) -> (4,4)` and back | `tl.reshape(raw, (HC, HC))`, `tl.reshape(comb, (HC*HC,))` | conform |
| `tl.broadcast_to` | broadcast base `(1,4,4)` | implicit via broadcasting `[:, None]` / `[None, :]` | conform (implicit broadcast) |
| `num_warps=1` | target hint | `num_warps=1` | conform |

Conformance note (non-normative, under `candidate-ready`): the sketch's
`tl.broadcast_to` is realized via Triton's implicit broadcasting of `[:, None]` /
`[None, :]` indexing rather than an explicit `tl.broadcast_to` call. This is a
target-language accommodation preserving identical semantics; it is not a design
change. All required primitives are `Supported`/`Constrained` on the
`triton_ascend` profile.

## Numerical Semantics Preserved (exact)

1. `pre = sigmoid(x[:,:4]*s0 + base[:4]) + eps` — `+eps` on `pre` only. ✓
2. `post = 2*sigmoid(x[:,4:8]*s1 + base[4:8])` — factor 2, no `+eps`. ✓
3. `comb = reshape(x[:,8:24],4,4)*s2 + reshape(base[8:24],4,4)`. ✓
4. `row_max = amax(-1)`; `comb = exp(comb - row_max)`. ✓
5. `comb = comb / sum(-1) + eps` (softmax row + eps). ✓
6. `comb = comb / (sum(-2) + eps)` (first column normalize). ✓
7. `static_range(19)`: `comb/(sum(-1)+eps)`; `comb/(sum(-2)+eps)` → total 20 row
   + 20 column normalizations. ✓
8. Output tuple `(pre[2,8,4], post[2,8,4], comb[2,8,4,4])` fp32. ✓
9. `ModelNew(hc_mult=4, sinkhorn_iters=20, eps=1e-6)`, `get_inputs`,
   `get_init_inputs` preserved. ✓

## Local Gate

| Gate | Command | Exit | Observation |
|---|---|---|---|
| ast.parse | `ast.parse(candidate_001.py)` | 0 | pass |
| harness loader | `load_ks_module(candidate_001.py)` | 0 | `ModelNew`/`get_inputs`/`get_init_inputs` resolved |
| warmup/compile smoke | forward ×3 + sync + kernel count | 0 | 1 Triton kernel (`_mhc_head_compute_mix_kernel`); numerics match base (pre/post maxdiff 0.0, comb maxdiff 8.9e-8) |
| full harness correctness | `auto_bench.py --warmup 5 --repeat 10` | 0 | PASS accuracy; v0=3.88 ms, v1=0.39 ms, speedup=9.83x |

The full harness smoke (warmup 5 / repeat 10) already reports `PASS accuracy`.
Authoritative measurement (warmup 50 / repeat 100) and profiler evidence remain
Verifier-owned and are not produced here.

## Attempt Ledger

| Attempt | Command | Exit | Defect | Before SHA | After SHA | Outcome |
|---:|---|---|---|---|---|---|
| 1 | write `candidate_001.py` + ast.parse + harness loader + forward smoke | 0 | none | n/a | `3eda8a14dede15a91f1a04c37bc5ff178a83fc87ecb7137b3569756c17f94f10` | candidate-ready |

## Reason Code

- `candidate-ready`: the candidate conforms to the immutable decision 001 design
  (single fused kernel, static_range Sinkhorn loop, exact numerical semantics,
  preserved public contract). No major deviation, no capability miss.

## Notes for Verifier

- Candidate path: `kernels/track1-triton/mhc_head_compute_mix/ascend/candidate_001.py`
- Expected mechanism observable: `kernel_count_per_call` should drop from 136 to 1
  (plus a handful of host-side `aten::reshape/view/to/empty` ops for the
  reshape/cast/alloc in forward).
- `get_inputs()` uses `device="npu"` (not `"cuda"`), per target-profile pitfall.
