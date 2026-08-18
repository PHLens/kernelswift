# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"sinkhorn-loop-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"collapse the entire forward (sigmoid gates, row-stabilized softmax, and the 20-iteration Sinkhorn row/column normalization loop) into a single Triton kernel with an internal static_range loop, reducing 136 per-call kernel launches to 1","allowed_changes":["ModelNew.forward kernel implementation","kernel dataflow and control flow"],"invariants":["ModelNew public constructor and forward signature","output tuple structure (pre,post,comb) and fp32 dtype and shapes","exact numerical semantics including +eps placement, softmax row_max stabilization, and 20 row + 20 column normalizations","caller-selected device and current stream preserved"],"expected_wall_improvement_pct":40.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor mixes shape=[16,24] dtype=fp32 layout=row_major memory=global
tensor hc_scale shape=[3] dtype=fp32 layout=contiguous memory=global
tensor hc_base shape=[24] dtype=fp32 layout=contiguous memory=global
tensor out_pre shape=[16,4] dtype=fp32 layout=row_major memory=global
tensor out_post shape=[16,4] dtype=fp32 layout=row_major memory=global
tensor out_comb shape=[16,4,4] dtype=fp32 layout=row_major memory=global
tile xrow shape=[24] dtype=fp32 memory=register
tile base shape=[24] dtype=fp32 memory=register
tile pre shape=[4] dtype=fp32 memory=register
tile post shape=[4] dtype=fp32 memory=register
tile comb shape=[4,4] dtype=fp32 memory=register
scalar s0 dtype=fp32 memory=register
scalar s1 dtype=fp32 memory=register
scalar s2 dtype=fp32 memory=register
scalar eps dtype=fp32 memory=register

# O Operations
load xrow <- mixes[pid,0:24]
load base <- hc_base[0:24]
load s0 <- hc_scale[0]
load s1 <- hc_scale[1]
load s2 <- hc_scale[2]
compute pre = sigmoid(xrow[0:4] * s0 + base[0:4]) + eps
compute post = 2 * sigmoid(xrow[4:8] * s1 + base[4:8])
compute comb = reshape(xrow[8:24],4,4) * s2 + broadcast(base[8:24],4,4)
compute row_max = max(comb,axis=1)
compute comb = exp(comb - row_max)
compute comb = comb / sum(comb,axis=1) + eps
compute comb = comb / (sum(comb,axis=0) + eps)
compute comb = rowcolnormalize(comb,eps)  # repeated 19 times inside the Sinkhorn loop
store out_pre[pid,0:4] <- pre
store out_post[pid,0:4] <- post
store out_comb[pid,0:4,0:4] <- comb

# C Control
parallel pid over 16
guard pid < 16
for iter over static_range(19)
end

# H Target Hints
target=triton_ascend
num_warps=1
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change: the forward is rewritten as a single fused Triton kernel with an internal static_range Sinkhorn loop; no buffer caching, allocation reuse, or host lifecycle change is introduced"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"collapse the entire forward (sigmoid gates, row-stabilized softmax, and the 20-iteration Sinkhorn row/column normalization loop) into a single Triton kernel with an internal static_range loop, reducing 136 per-call kernel launches to 1","expected_causal_chain":["the Python for-loop and per-iteration torch reductions are replaced by one kernel launch","kernel_count_per_call drops from 136 to 1","per-launch host overhead (launcher/alloc/sync) collapses","wall time decreases despite near-identical device compute"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output dtype and shape unchanged","numerical semantics preserved (+eps only on pre and Sinkhorn denominators, factor 2 on post, row_max-stabilized softmax, 20 row + 20 column normalizations)"],"profiling_level":"summary"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`; no matching failure invalidates this path. The catalog entries (winner trees, full sort networks, dynamic gather, cumsum compaction) concern selection/compaction lowering on a grouped top-k MLU workload and are unrelated to a small 4x4 Sinkhorn normalization fused into one kernel.
- The `triton_ascend` target profile confirms every primitive this sketch requires is Supported or Constrained on the recorded Ascend910B4 runtime: `tl.sum` (axis-0), `tl.max` (axis-1 over (4,4)), `tl.exp`, `tl.reshape`, `tl.broadcast_to`, `tl.static_range` (compile-time loop), and `num_warps=1`. No Unknown or Unsupported primitive is required.
- Pitfall guarded: the Sinkhorn loop must use a compile-time `static_range` (iterations are the fixed constructor arg `sinkhorn_iters=20`), not a data-dependent `for`, so the loop unrolls/loops inside one kernel rather than re-launching.
- Pitfall guarded: `get_inputs()` in the candidate must not hardcode `"cuda"`; use `"npu"` or derive from an input tensor's device (the harness already rewrites `"cuda"`, but the profile recommends explicit `npu`).

## Rationale and Evidence

Report 000 is authoritative: the operator is strongly host-bound with `device_ratio ≈ 0.082` (device_us_per_call ≈ 281 us vs wall ≈ 3406 us), so roughly 92% of benchmark wall time is host-side (launcher/alloc/sync/context), not device compute. The single dominant cost driver is launch count: `kernel_count_per_call = 136`, produced by the Python `for _ in range(sinkhorn_iters - 1)` loop in `forward` that issues two tiny torch reductions (`sum(-1)`, `sum(-2)`) plus a division per iteration, plus the fixed head (sigmoid, amax, exp, adds). The top device kernel `aclnnReduceSum` appears 40 times per call (169 us/call) at ~4.2 us each — each individual kernel is tiny, so the per-launch overhead far exceeds the compute.

The intervention fuses the entire forward into one Triton kernel over a 16-program grid (one program per `(b,s)` row; the 16 rows are independent because the Sinkhorn normalization reduces only the last two `4x4` axes). Each program loads its 24-float `mixes` row and the shared 24-float `hc_base`/3-float `hc_scale`, computes `pre` and `post` gates, builds and row-stabilizes the `4x4` combination matrix, and runs the 20 row + 20 column normalizations inside a single `tl.static_range` loop. This collapses 136 launches to 1, directly attacking the host-bound wall time. The 4x4 reduction is expressible with the already-proven `tl.sum`/`tl.max` (the loop count is a compile-time constant, so no re-launch). Numerical exactness is preserved: `+eps` only on `pre` and Sinkhorn denominators, factor 2 on `post` with no `+eps`, `row_max` subtracted before `exp`, and exactly 20 row + 20 column normalizations (softmax row + first column normalize, then 19 loop iterations of row+column). The expected improvement (40%) is large because ~92% of wall time is host-side launch overhead that this fusion removes; even a partial realization exceeds the 5% adoption threshold.
