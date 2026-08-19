# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"maca","target_profile":"triton_maca","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"sinkhorn-loop-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"fuse the entire MHCHeadComputeMix forward (sigmoid gates, exp/row_max stabilization, and the 20-iteration Sinkhorn alternating normalization) into a single Triton kernel, one program per (b,s) position, collapsing 133 library launches per call into 1 fused kernel launch","allowed_changes":["kernel dataflow","fused Sinkhorn loop","in-kernel reductions","single-kernel launch"],"invariants":["ModelNew public contract","output tuple shape/dtype/device","exact fp32 Sinkhorn semantics","input non-mutation","caller-selected device and current stream preserved"],"expected_wall_improvement_pct":30.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor mixes shape=[16,24] dtype=fp32 layout=row_major memory=global
tensor hc_scale shape=[3] dtype=fp32 layout=contiguous memory=global
tensor hc_base shape=[24] dtype=fp32 layout=contiguous memory=global
tensor pre shape=[16,4] dtype=fp32 layout=row_major memory=global
tensor post shape=[16,4] dtype=fp32 layout=row_major memory=global
tensor comb shape=[16,4,4] dtype=fp32 layout=row_major memory=global
scalar hc value=4
scalar sinkhorn_iters value=20
scalar eps value=1e-6
tile xrow shape=[24] dtype=fp32 memory=register
tile comb_local shape=[16] dtype=fp32 memory=register
tile pre_local shape=[4] dtype=fp32 memory=register
tile post_local shape=[4] dtype=fp32 memory=register
tile row_sums shape=[4] dtype=fp32 memory=register
tile col_sums shape=[4] dtype=fp32 memory=register

# O Operations
load xrow <- mixes[pos,0:24]
load s0 <- hc_scale[0]
load s1 <- hc_scale[1]
load s2 <- hc_scale[2]
load base <- hc_base[0:24]
compute pre_local[i] = sigmoid(xrow[i]*s0 + base[i]) + eps for i in 0:4
compute post_local[i] = 2*sigmoid(xrow[4+i]*s1 + base[4+i]) for i in 0:4
compute comb_local[j*4+i] = xrow[8+j*4+i]*s2 + base[8+j*4+i] for i in 0:4 and j in 0:4
compute row_max[j] = max over i of comb_local[j*4+i]
compute comb_local[j*4+i] = exp(comb_local[j*4+i] - row_max[j])
compute row_sums[j] = sum over i of comb_local[j*4+i]
compute comb_local[j*4+i] = comb_local[j*4+i] / row_sums[j] + eps
compute col_sums[i] = sum over j of comb_local[j*4+i]
compute comb_local[j*4+i] = comb_local[j*4+i] / (col_sums[i] + eps)
compute row_sums[j] = sum over i of comb_local[j*4+i] for it in 0..(sinkhorn_iters-1)
compute comb_local[j*4+i] = comb_local[j*4+i] / (row_sums[j] + eps) for it in 0..(sinkhorn_iters-1)
compute col_sums[i] = sum over j of comb_local[j*4+i] for it in 0..(sinkhorn_iters-1)
compute comb_local[j*4+i] = comb_local[j*4+i] / (col_sums[i] + eps) for it in 0..(sinkhorn_iters-1)
store pre[pos,0:4] <- pre_local
store post[pos,0:4] <- post_local
store comb[pos,0:16] <- comb_local

# C Control
parallel pos over 16
guard pos < 16
for it in 0..(sinkhorn_iters-1)
end

# H Target Hints
target=triton_maca
num_warps=1
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change: the fused Triton kernel replaces the reference forward body; all PyTorch host-side allocation of pre/post/comb remains inside forward and preserves caller-selected device and current stream"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the entire MHCHeadComputeMix forward (sigmoid gates, exp/row_max stabilization, and the 20-iteration Sinkhorn alternating normalization) into a single Triton kernel, one program per (b,s) position, collapsing 133 library launches per call into 1 fused kernel launch","expected_causal_chain":["the 133-library-kernel Sinkhorn/normalization chain collapses into one fused Triton kernel","kernel count per call drops from 133 to a tiny constant","host launch and dispatch overhead that dominated ~65% of wall time is eliminated","device time drops because reductions become in-kernel over 4-element dims instead of separate reduce_kernel_maca launches","benchmark wall time drops well beyond the 5% adoption threshold"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"candidate_kernel_count_per_call","expectation":"decrease from ~133 to 1 or a tiny constant"},{"name":"candidate_device_us_per_call","expectation":"decrease from ~534.685"},{"name":"fused_triton_kernel_count_per_call","expectation":"equal to 1.0"},{"name":"sinkhorn_sum_div_kernel_us_per_call","expectation":"decrease from ~500 toward 0"},{"name":"correctness","expectation":"pre/post/comb allclose atol=1e-2 rtol=1e-2"}],"guardrails":["correctness:pass","output tuple shape dtype device unchanged","inputs not mutated","caller-selected device and current stream preserved"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. The catalog entries concern dynamic
  gather/cumsum/sort-network lowering regressions on an MLU590-H8 runtime; none of
  the listed preconditions (dynamic compaction, partial selection networks,
  prefix-sum compaction) match this decision. This intervention uses only static
  compile-time loops over the tiny hc=4 dims and the 20-iteration Sinkhorn loop,
  with no dynamic indexing or sorting.
- Consulted `references/invariants.md`. The Sinkhorn loop count is an exact
  semantic invariant: the first normalization pair runs unconditionally, then
  exactly `sinkhorn_iters - 1 = 19` additional (row, column) pairs. The loop body
  must remain a data-dependent loop over `static_range(20)` (or the equivalent
  `static_range(sinkhorn_iters)`), NOT a hand-unrolled chain, so the exact 20
  total normalization pairs are preserved.
- Target-profile pitfalls honored: `num_warps=1` (warp_size 64) is the only proven
  launch config; direct launch `kernel[(grid,)](...)` is the proven launcher; the
  harness AST loader must be used; `MACA_PATH` must be set before importing Triton.

## Rationale and Evidence

The accepted baseline (report_000.md) is host-bound: `device_ratio ~0.35`
(`device_us_per_call ~534.685 us` out of `~1515 us` wall), so roughly 65% of wall
time is host launch/dispatch overhead rather than device compute. The operator is
tiny — the comb matrix is `[2,8,4,4]` (16 positions of a 4x4 matrix) — yet each
forward launches 133 kernels. The 20-iteration Sinkhorn loop accounts for ~100 of
those 133 kernels: 20 `reduce_kernel_maca` (sum dim -1), 20 (sum dim -2), 41
`+eps` self-add elementwise kernels, and 40 `DivFunctor` elementwise kernels,
totaling ~500 us/call of device time that is pure reduction/division overhead.

Fusing the entire forward into one Triton kernel, with one program per (b,s)
position (16 programs) and the Sinkhorn loop expressed as an in-kernel
`for` over `static_range(20)`, collapses 133 launches to 1 and moves all
reductions/divisions into registers over the 4-element dims. The mechanism
observables directly track this: `candidate_kernel_count_per_call` (133 -> 1 or a
tiny constant), `candidate_device_us_per_call` (drop from 534.685 us), and
`sinkhorn_sum_div_kernel_us_per_call` (drop from ~500 us toward 0). Because
~65% of wall time is launch overhead, eliminating 132 launches is expected to
yield a wall improvement far exceeding the 5% threshold (estimated ~30%).

The comb tensor is per-(b,s) position: shape `[2,8,4,4]` = 16 positions, each an
independent 4x4 matrix normalized over its last two dims. This maps cleanly to
one Triton program per position. The Sinkhorn loop is data-dependent and must
execute exactly 20 total normalization pairs (the first pair unconditional, then
19 more), preserving `base.py`'s exact fp32 semantics including the specific
`+eps` placements: after the first row-normalize `comb = comb/comb.sum(-1) + eps`,
the column step `comb = comb/(comb.sum(-2) + eps)`, and every subsequent row/column
step `comb/(comb.sum(axis) + eps)`.

The profile lists `tl.sum` as Supported for a 256-element scalar reduction, but
the 4-element reduction shape is unproven; the Coder is directed to use explicit
manual reduction over the 4-element dims (hc=4 is tiny) where `tl.sum` is not
proven, and to use only proven primitives (`tl.load`, `tl.store`, `tl.arange`,
`tl.exp`, `tl.max`, `tl.static_range`, scalar math). `tl.dot` is Unknown but not
needed. Non-benchmark shapes (any `mix_hc != 24`) must fall back to the unchanged
PyTorch reference path so the public contract is preserved.
