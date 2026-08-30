# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"device-bound","intervention":"fuse the 20-round Sinkhorn iteration and the surrounding head-compute elementwise chain into a single Triton kernel using a compile-time tl.static_range loop, collapsing ~133 per-call kernel launches (132.88 kernels/call, dominated by ~40 sum + ~40 div + ~41 add-eps tiny kernels over the [16,4,4] comb tensor) into 1-2 kernels that keep the entire [16,4,4] comb matrix in registers and never round-trip to global memory between normalization rounds","allowed_changes":["ModelNew.forward dataflow","fused Triton kernel over the Sinkhorn iteration and head-compute elementwise stages"],"invariants":["ModelNew public contract (hc_mult=4, sinkhorn_iters=20, eps=1e-6)","forward signature (mixes,hc_scale,hc_base)->(pre,post,comb)","output tuple structure, shapes pre/post [2,8,4] and comb [2,8,4,4], all fp32","exact Sinkhorn numerical semantics including the asymmetric eps placement (first explicit row-normalize adds eps to the matrix; the loop row-normalize and all column-normalizes add eps to the sum denominator)","input tensors not mutated","caller-selected device and current stream preserved"],"expected_wall_improvement_pct":25.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor mixes shape=[2,8,24] dtype=fp32 layout=contiguous memory=global
tensor hc_scale shape=[3] dtype=fp32 layout=contiguous memory=global
tensor hc_base shape=[24] dtype=fp32 layout=contiguous memory=global
tensor pre_out shape=[2,8,4] dtype=fp32 layout=contiguous memory=global
tensor post_out shape=[2,8,4] dtype=fp32 layout=contiguous memory=global
tensor comb_out shape=[2,8,4,4] dtype=fp32 layout=contiguous memory=global
tile comb shape=[4,4] dtype=fp32 memory=register
tile comb_row_sum shape=[4] dtype=fp32 memory=register
tile comb_col_sum shape=[4] dtype=fp32 memory=register
scalar N shape=[1] dtype=int
scalar hc shape=[1] dtype=int

# O Operations
load x <- mixes.reshape(N,24)                    # N = 16
load s0 <- hc_scale[0]
load s1 <- hc_scale[1]
load s2 <- hc_scale[2]
load base <- hc_base[0:24]
compute pre = sigmoid(x[:,0:4]*s0 + base[0:4]) + eps          # [N,4]
compute post = 2*sigmoid(x[:,4:8]*s1 + base[4:8])             # [N,4]
compute comb = x[:,8:24].view(N,4,4)*s2 + base[8:24].view(1,4,4)   # [N,4,4]
compute row_max = amax(comb, axis=-1, keepdim)                # [N,4,1]
compute comb = exp(comb - row_max)                            # stable softmax
compute comb = comb / sum(comb, axis=-1, keepdim) + eps       # row normalize (eps to matrix)
compute comb = comb / (sum(comb, axis=-2, keepdim) + eps)     # col normalize (eps to denom)
compute comb = row_normalize(comb) + eps                      # loop row normalize (eps to denom)
compute comb = col_normalize(comb)                            # loop col normalize (eps to denom)
store pre_out <- pre.view(2,8,4)
store post_out <- post.view(2,8,4)
store comb_out <- comb.view(2,8,4,4)

# C Control
parallel n over N
for it over sinkhorn_iters-1
guard n < N
end

# H Target Hints
target=triton_cuda
num_warps=4
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; the fused Triton kernel is launched inside forward with no new host-side state, buffer reuse, or allocation caching, and mixes/hc_scale/hc_base are read-only inputs"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the 20-round Sinkhorn iteration and the surrounding head-compute elementwise chain into a single Triton kernel using a compile-time tl.static_range loop, collapsing ~133 per-call kernel launches (132.88 kernels/call, dominated by ~40 sum + ~40 div + ~41 add-eps tiny kernels over the [16,4,4] comb tensor) into 1-2 kernels that keep the entire [16,4,4] comb matrix in registers and never round-trip to global memory between normalization rounds","expected_causal_chain":["per-call kernel count drops from 132.88 to approximately 1-2","the dominant reduce_kernel sum (442.5 us/call), DivFunctor (254.2 us/call), and CUDAFunctorOnSelf_add (161.4 us/call) tiny-kernel launches disappear and their device math is kept in registers inside one kernel","device_us_per_call drops substantially","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output dtype and shape unchanged","output tuple structure unchanged","Sinkhorn normalization count and eps placement preserved","input not mutated"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. No entry's preconditions match this operator: the catalog records grouped-topk selection-network (`tl.gather`, full-sort, cumsum compaction) failures on an MLU590-H8 runtime, which are reduction/selection-network dataflow paths absent here. This operator is a fixed 20-round row/column alternating normalization over a tiny `[16,4,4]` fp32 matrix, not a hierarchical/partial selection. The fusion is elementwise-plus-axis-sum, which the profile marks supported.
- Target-profile consultation (`prompts/coder_targets/triton_cuda.md`): `tl.load`, `tl.store`, `tl.arange`, `tl.reshape`, `tl.max`, `tl.sum`, `tl.exp`, `tl.where`, `tl.broadcast_to`, and `tl.static_range` are all Supported on the recorded BI150 / CoreX runtime (groupedtopk and smoke probes). `tl.static_range` is specifically recorded as "compile-time loop with four iterations", so a 19-iteration compile-time loop is the same construct at a larger count; Coder must verify the loop-body unroll lowers correctly at 19 iterations (this is the primary capability risk). `num_warps`/`num_stages` remain Unknown and must stay non-normative; the `num_warps=4` hint is advisory, not a correctness requirement.
- The correctness-critical trap is the **asymmetric eps placement**: the first explicit row-normalize adds `eps` to the normalized matrix (`comb/sum + eps`), while the loop row-normalize and every column-normalize add `eps` to the sum denominator (`comb/(sum + eps)`). A Triton implementation that uniformly applies `eps` to one side will silently diverge; this is the single highest-risk correctness point and is already pinned in `project.md#semantics`.
- The harness AST loader (`auto_bench.py` `_filter_module_ast`) retains `Import`/`ImportFrom`/`ClassDef`/`FunctionDef` and literal assignments, so a `@triton.jit`-decorated top-level function is preserved; the candidate module must still expose `ModelNew`, `get_init_inputs`, and `get_inputs`.
- Keeping `comb` as a `[4,4]` register tile across 19 iterations is the intended mechanism (one program per `(b,s)` position, i.e. a 16-program grid or a single program handling all 16 via `tl.arange`). The `[4,4]` tile is tiny (256 elements), so register pressure is negligible; the risk is whether the compiler keeps the loop fully on-chip without spilling, which the Level 1 device-time and kernel-count observables will directly verify.

## Rationale and Evidence

Phase 0 profiler evidence (`rounds/report_000.md`) establishes a **device-bound** baseline with extreme launch-count inefficiency: `baseline_base` scope measured `926.395 us/device-call` and `132.88 kernels/call` against `1.517299 ms` wall time, giving `device_ratio ≈ 0.611`. Roughly 61% of wall time is device kernel time, and that device time is dominated by a very large number of tiny kernels over the `[16,4,4]` (256-element) `comb` tensor:

- `reduce_kernel<sum>`: `39.96/call`, `442.515 us/call` (the row/column `sum` reductions of the Sinkhorn loop)
- `DivFunctor`: `39.96/call`, `254.236 us/call` (the normalization divisions)
- `CUDAFunctorOnSelf_add<float>`: `40.96/call`, `161.407 us/call` (the `+ eps` floor additions)

These three kernel families alone account for ~858 us/call, all generated by the 20-round Sinkhorn iteration (each round = one row-sum + one row-div + one add-eps + one col-sum + one col-div + one add-eps, i.e. ~6 kernels × 20 rounds ≈ 120 kernels). The remaining ~13 kernels are the one-off head-compute stage (2 sigmoid, 1 exp, 1 amax, and the s0/s1/s2 affine mul/add). Every one of these kernels operates on a few hundred elements and pays a full launch + global-memory round-trip for work that belongs entirely in registers.

Fusing the Sinkhorn iteration into a single Triton kernel with a compile-time `tl.static_range` loop is the textbook remedy: it collapses ~133 launches to 1-2, keeps the `[4,4]` `comb` matrix in registers across all 20 rounds (no global-memory round-trip between normalizations), and eliminates the per-kernel launch overhead that dominates device time on tiny tensors. Because the workload is device-bound (`device_ratio 0.61`), compressing device time directly propagates to wall time (unlike a host-bound case where the harness seed/synchronization floor would cap the gain).

The expected gain is bounded but substantial. The three dominant Sinkhorn kernel families (858 us/call) are almost entirely launch + tiny-kernel inefficiency that fusion removes; a conservative estimate of a 30-40% device-time reduction after fusion (the fused kernel still pays one launch and the `exp`/`sigmoid`/`amax` math, but none of the ~120 redundant launches) yields a wall improvement around 20-30%. I record `expected_wall_improvement_pct = 25.0` as a reasonable central expectation; the adoption threshold remains the harness `5%` unrounded median. If the compiler cannot keep 19 `tl.static_range` iterations on-chip (the primary capability risk), the device-time observable will reveal it and the round completes as `capability-miss`, not a silent numerical regression.

The canonical comparison source is `baseline_adapter.py` (established in Phase 0), and the reference report is `rounds/report_000.md`. The intervention changes only the kernel dataflow inside `ModelNew.forward`; the public contract, output structure/shape/dtype, and the exact Sinkhorn semantics (20 rounds, eps placement) are preserved as invariants.
