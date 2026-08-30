# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"fuse the per-expert feedforward (gate/up GEMM, SiLU gating, down GEMM) and weighted reduce into a single per-token Triton kernel using elementwise tl.sum outer-products, eliminating the mask/gather/scatter dispatch and the 16 per-expert MatMul launches","allowed_changes":["kernel dataflow","ModelNew.forward dispatch"],"invariants":["ModelNew public constructor and forward contract","output shape [83,128] fp16","softmax+topk+renormalize routing semantics","weighted top-k reduce over exactly 2 experts","atol=1e-2 rtol=1e-2 tolerance"],"expected_wall_improvement_pct":30.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor hidden shape=[T,H] dtype=fp16 layout=contiguous memory=global
tensor topk_ids shape=[T,K] dtype=int32 layout=contiguous memory=global
tensor topk_weights shape=[T,K] dtype=fp16 layout=contiguous memory=global
tensor w1 shape=[E,2I,H] dtype=fp16 layout=contiguous memory=global
tensor w2 shape=[E,H,I] dtype=fp16 layout=contiguous memory=global
tensor out shape=[T,H] dtype=fp16 layout=contiguous memory=global
tile x shape=[H] dtype=fp16 memory=register
tile out_acc shape=[H] dtype=fp32 memory=register

# O Operations
load x <- hidden[token,0:H]
compute out_acc = zeros([H], fp32)
load expert_id <- topk_ids[token,k]
load weight <- topk_weights[token,k]
load w1_block <- w1[expert_id,0:2I,0:H]
compute gate_up = sum(x[None,:] * w1_block, axis=1)
compute gate = gate_up[0:I]
compute up = gate_up[I:2I]
compute act = silu(gate) * up
load w2_block <- w2[expert_id,0:H,0:I]
compute out_k = sum(act[None,:] * w2_block, axis=1)
compute out_acc += weight * out_k
store out[token,0:H] <- out_acc.to(fp16)

# C Control
parallel token over T
guard token < T
for k over K
end

# H Target Hints
target=triton_ascend
num_warps=1
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; routing stays in PyTorch inside forward and no output buffer reuse or caching is introduced"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the per-expert feedforward (gate/up GEMM, SiLU gating, down GEMM) and weighted reduce into a single per-token Triton kernel using elementwise tl.sum outer-products, eliminating the mask/gather/scatter dispatch and the 16 per-expert MatMul launches","expected_causal_chain":["the per-expert Python loop with Nonzero/Index/IndexPut mask-gather-scatter and 16 MatMul launches disappears","kernel_count_per_call decreases from 126 toward 1","device_us_per_call decreases from ~744 us","wall_time_ms decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease from 126 toward 1"},{"name":"device_us_per_call","expectation":"decrease from ~744 us/call"},{"name":"aclnnNonzeroV2_presence","expectation":"absent from candidate scope"},{"name":"aclnnIndexPutImpl_presence","expectation":"absent from candidate scope"}],"guardrails":["correctness:pass","output shape [83,128] fp16 unchanged","softmax+topk+renormalize routing semantics preserved","weighted top-k reduce over exactly 2 experts preserved"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. The recorded failures (winner tree, sort-32/64 selection network, dynamic `tl.gather` compaction, cumsum compaction) all concern grouped top-k selection on MLU590; none applies to this decision, which performs no expert selection inside the kernel (top-2 routing stays in PyTorch) and no dynamic gather/compaction.
- Consulted the `triton_ascend` target profile. This decision uses only Supported primitives (`tl.load`, `tl.store`, `tl.arange`, `tl.zeros`, `tl.program_id`, `tl.exp` for SiLU, `tl.sum`, `tl.static_range`, `tl.reshape`). It deliberately avoids `tl.dot` (the Cube path, which the two prior Ascend campaigns found carries a host/launch penalty that can negate device savings) and avoids `fast_libentry` (Unknown on Ascend, per profile) and `num_stages`/`tl.trans` (Unknown). `num_warps=1` is proven on Ascend.
- Two prior Ascend campaigns (groupedtopk, flexattention) both confirmed: single-kernel fusion of decomposed `aclnn*` libraries is the dominant round-1 win; `tl.sum` rank-1 outer-products with `num_warps=1` and direct Triton launch are the proven path. This decision follows that exact pattern.
- SiLU must be computed as `gate * (1/(1+exp(-gate)))` using `tl.exp`, not `F.silu`, since the computation is inside the Triton kernel.

## Rationale and Evidence

Baseline (report_000) is host-bound: wall 7.159 ms with device only ~744 us/call (device_ratio 0.104), spread across 126 kernels per forward call. The per-expert Python `for e in range(8)` loop is the structural cause: it emits, per expert, a `flat_ids == e` mask (`aclnnNonzeroV2` NonZero + MemSet, ~337 us/call combined), an `x_rep[mask]` gather (`aclnnIndex`, ~82 us/call), and an `expert_out[mask] = ...` scatter (`aclnnIndexPutImpl`, ~57 us/call), plus 16 `aclnnMatmul` launches (2 per expert). Mask+gather+scatter alone account for ~476 us of the ~744 us device time, and the 126-kernel dispatch is what keeps 90% of wall time on the host.

A single per-token Triton kernel with `grid=(T,)` removes the mask/gather/scatter entirely: each program loads its token's top-2 expert ids and weights, loads the two experts' `w1`/`w2` blocks, computes gate/up GEMM, SiLU, and down GEMM via elementwise `tl.sum(x[None,:]*w, axis=1)` outer-products, and accumulates the weighted result. Routing (softmax + top-2 + renormalize) stays in PyTorch in `forward` for this round. This collapses the 126-kernel, ~744 us device workload to a single kernel.

The expected outcome is directly evidenced by the sibling MLU campaign (round 1 per-token fusion: 6.94 ms -> 0.564 ms, 12.3x) and the two prior Ascend campaigns (groupedtopk round 1: 19 -> 1 kernel, device 172.8 -> 34.6 us, wall +54.9%; flexattention round 1: 8.72 -> 1 kernel, device 145 -> 54 us, wall +18.5%). fused_moe's baseline has strictly more kernel count (126) and a larger mask/gather/scatter share than either, so the fusion win should be at least as large. `expected_wall_improvement_pct` is set conservatively at 30% (well above the 5% adoption threshold) because the device-time fraction is only 10% and a portion of the remaining ~6.4 ms wall is host launch/dispatch that a single kernel reduces but does not fully eliminate.
