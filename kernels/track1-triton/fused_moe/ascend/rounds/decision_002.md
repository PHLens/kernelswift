# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"002","reference_implementation":"triton_fused_moe_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"fuse routing (softmax over E=8, top-2 argmax selection, renormalize) into the existing per-token Triton kernel so forward passes router_logits directly and the 11 PyTorch routing kernels disappear; keep the elementwise tl.sum FFN path and do NOT introduce tl.dot","allowed_changes":["kernel dataflow","ModelNew.forward routing removal"],"invariants":["ModelNew public constructor and forward contract","output shape [83,128] fp16","softmax+topk+renormalize routing semantics over E=8 experts","weighted top-k reduce over exactly 2 experts","elementwise tl.sum FFN path retained (no tl.dot)","atol=1e-2 rtol=1e-2 tolerance"],"expected_wall_improvement_pct":25.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor hidden shape=[T,H] dtype=fp16 layout=contiguous memory=global
tensor router_logits shape=[T,E] dtype=fp32 layout=contiguous memory=global
tensor w1 shape=[E,2I,H] dtype=fp16 layout=contiguous memory=global
tensor w2 shape=[E,H,I] dtype=fp16 layout=contiguous memory=global
tensor out shape=[T,H] dtype=fp16 layout=contiguous memory=global
tile x shape=[H] dtype=fp16 memory=register
tile scores shape=[E] dtype=fp32 memory=register
tile topk_ids shape=[K] dtype=int32 memory=register
tile topk_weights shape=[K] dtype=fp32 memory=register
tile out_acc shape=[H] dtype=fp32 memory=register

# O Operations
load logits <- router_logits[token,0:E]
compute scores = softmax(logits)
compute topk_ids,topk_weights = topk(scores,K)
compute topk_weights = renormalize(topk_weights)
load x <- hidden[token,0:H]
compute out_acc = zeros([H], fp32)
load expert_id <- topk_ids[k]
load weight <- topk_weights[k]
load gate_block <- w1[expert_id,0:I,0:H]
load up_block <- w1[expert_id,I:2I,0:H]
compute gate = sum(x[None,:] * gate_block, axis=1)
compute up = sum(x[None,:] * up_block, axis=1)
compute act = silu(gate) * up
load w2_block <- w2[expert_id,0:H,0:I]
compute out_k = sum(act[None,:] * w2_block, axis=1)
compute out_acc += weight * out_k
store out[token,0:H] <- out_acc.to(fp16)

# C Control
parallel token over T
guard token < T
for k over K

# H Target Hints
target=triton_ascend
num_warps=1
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; forward simplifies to a single kernel launch on router_logits and no buffer reuse, caching, or device/stream change is introduced"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-002","intervention":"fuse routing (softmax over E=8, top-2 argmax selection, renormalize) into the existing per-token Triton kernel so forward passes router_logits directly and the 11 PyTorch routing kernels disappear; keep the elementwise tl.sum FFN path and do NOT introduce tl.dot","expected_causal_chain":["the 11 PyTorch routing kernels (Topk, Softmax, ReduceSum, Div, Cast, GatherElements) disappear","kernel_count_per_call decreases from 12 toward 1","device_us_per_call decreases from ~97 us toward the single fused kernel (~20 us plus routing compute)","host-side routing dispatch decreases","wall_time_ms decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease from 12 toward 1"},{"name":"device_us_per_call","expectation":"decrease from ~97 us/call"},{"name":"aclnnTopk_presence","expectation":"absent from candidate scope"},{"name":"aclnnSoftmax_presence","expectation":"absent from candidate scope"}],"guardrails":["correctness:pass","output shape [83,128] fp16 unchanged","softmax+topk+renormalize routing semantics over E=8 preserved","weighted top-k reduce over exactly 2 experts preserved","no tl.dot introduced"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. The four recorded failures — winner tree for repeated expert selection, sort-32 plus sort-64 selection network, dynamic `tl.gather` compaction, and cumsum compaction — all concern **grouped top-k selection over 256 experts (8 groups of 32) on MLU590**, with hierarchical reduction, full sort networks, or dynamic gather. None applies here: this decision selects top-2 over only **E=8 experts** using a simple per-token repeated argmax loop, which is the exact approach that succeeded on MLU (fused_moe round 2) and on the S60 GCU grouped-topk candidate. No winner tree, no sort network, no dynamic gather/compaction is introduced.
- Consulted the `triton_ascend` target profile. `tl.max`, `tl.exp`, `tl.sum`, `tl.argmax`, `tl.where`, `tl.static_range`, `tl.load`, `tl.store`, `tl.arange`, `tl.zeros`, `tl.reshape`, `tl.broadcast_to` are all Supported on Ascend910B4. `tl.argmax` is Supported but its tie behavior is "not characterized"; the masked-sum argmax formulation (`tl.sum(tl.where(remaining == best_val, e_idx, 0))`) is an allowed equivalent fallback and is what the MLU round-2 kernel used. For fp32 `torch.randn` logits, softmax scores are distinct with probability ~1, so `torch.topk`'s index-order tie-break is not exercised and either formulation matches within tolerance.
- Consulted the flexattention Ascend Round 3 report: replacing `tl.sum` with `tl.dot` (Cube/BMM) halved device time (54.43 -> 24.05 us) but regressed wall time -8.34% because the Cube-unit launch/dispatch path added ~55 us/call of host cost. This decision therefore **explicitly keeps the elementwise `tl.sum` FFN path** and does not introduce `tl.dot`.
- Consulted `references/bottleneck-judgment.md`: the candidate device_ratio after round 1 is 0.171 (host-bound), so the win depends on reducing both device work (routing ~77 us) and host-side routing launch/dispatch (11 launches). Removing the routing kernels is a single attributable cause (routing), consistent with selecting one intervention.

## Rationale and Evidence

After Round 1 (report_001), the candidate runs at 0.569590 ms wall with 97.366 us device across 12 kernels/call. The single fused `_fused_moe_per_token_kernel` is only ~20 us/call; the remaining ~77 us / 11 kernels are entirely routing: `aclnnTopk` (~39 us), `aclnnSoftmax` (~11 us), `aclnnInplaceCopy_Cast` (~11 us, from the `topk_ids.to(int32)` and `topk_weights.to(fp16)` casts), `aclnnReduceSum` (~7 us, the renormalize denominator), `aclnnTopk_GatherElements` (~6 us), plus `aclnnDiv` and `aclnnTopk_Cast`. Routing is now the dominant remaining cost in both device time (~79% of device) and kernel count (11 of 12).

Fusing routing into the per-token kernel removes all 11 routing kernels: each program loads its `router_logits[token, 0:8]`, computes softmax (max-subtract + exp + sum), selects top-2 by repeated argmax over E=8, renormalizes, then runs the existing elementwise `tl.sum` FFN and weighted reduce. `forward` simplifies to casting `w1`/`w2` and a single kernel launch on `router_logits`. This is byte-for-byte the MLU round-2 intervention, which on MLU took wall from 0.564 ms to 0.218 ms (12.3x -> 31.9x). On Ascend, the routing device cost (~77 us) is ~13.5% of the 569 us wall by itself, and removing 11 host-side routing launches adds further savings, so an expected 25% wall improvement is a conservative, falsifiable estimate well above the 5% adoption threshold.

The two risks are explicitly ruled out: (1) in-kernel top-k selection anti-patterns do not apply at E=8 with a simple argmax loop (the failed patterns are 256-expert hierarchical/sort/gather approaches on MLU590); (2) `tl.dot` is not introduced, avoiding the flexattention Round 3 host-penalty regression. The softmax/argmax primitives are all Supported on Ascend.
