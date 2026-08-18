# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"triton-attention-rewrite"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"replace F.scaled_dot_product_attention and its surrounding view/transpose/reshape with a single Triton attention kernel that reads the native [bsz, seq, num_heads*head_size] contiguous layout via strided loads and writes the same layout directly, eliminating the three .contiguous() transpose kernels and the output InplaceCopy transpose kernel","allowed_changes":["ModelNew.forward computation","attention kernel dataflow and layout access"],"invariants":["ModelNew(num_heads=8, head_size=64, num_kv_heads=8) public contract","output Tensor[2,83,512] fp16 on caller-selected device","numerical semantics equal within atol=1e-2, rtol=1e-2 (equal_nan=True)","non-GQA head arithmetic fixed at num_heads=8, head_size=64"],"expected_wall_improvement_pct":15.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor query shape=[2,83,512] dtype=fp16 layout=contiguous memory=global
tensor key shape=[2,83,512] dtype=fp16 layout=contiguous memory=global
tensor value shape=[2,83,512] dtype=fp16 layout=contiguous memory=global
tensor output shape=[2,83,512] dtype=fp16 layout=contiguous memory=global
tile q shape=[64] dtype=fp32 memory=register
tile scores shape=[83] dtype=fp32 memory=register
tile v shape=[64] dtype=fp32 memory=register
tile acc shape=[64] dtype=fp32 memory=register

# O Operations
load q <- query[batch, seq, head*64:(head+1)*64]
compute scores[k] = dot(q, key[batch, k, head*64:(head+1)*64]) * 0.125
compute scores = softmax(scores) over k in 0..83
compute acc = sum_k scores[k] * value[batch, k, head*64:(head+1)*64]
store output[batch, seq, head*64:(head+1)*64] <- acc

# C Control
parallel batch over 2
parallel head over 8
parallel seq over 83
for k over 83
end

# H Target Hints
target=triton_ascend
num_warps=1
num_stages=2
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; forward calls a single Triton kernel in place of the SDPA call plus surrounding layout ops, with no host-side state, allocation cache, or stream change"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"replace F.scaled_dot_product_attention and its surrounding view/transpose/reshape with a single Triton attention kernel that reads the native [bsz, seq, num_heads*head_size] contiguous layout via strided loads and writes the same layout directly, eliminating the three .contiguous() transpose kernels and the output InplaceCopy transpose kernel","expected_causal_chain":["the three aclnnFlashAttentionScore_TransposeAiCore_Transpose input-transpose kernels and the one aclnnInplaceCopy_TransposeAiCore_Transpose output-transpose kernel disappear","kernel_count_per_call decreases from ~6.7 toward ~1","device_us_per_call decreases by eliminating ~62 us of layout shuffle","host launch gaps between per-call kernels shrink","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"decrease"},{"name":"transpose_wrapper_kernel_count_per_call","expectation":"decrease toward zero"}],"guardrails":["correctness:pass","output dtype and shape unchanged","fp16 output within atol=1e-2 rtol=1e-2","non-GQA head arithmetic preserved"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. No listed failure matches this path: the catalog entries (winner tree, sort networks, dynamic gather, cumsum compaction) target grouped top-k on MLU590 and concern on-chip selection/compaction, not attention layout fusion.
- Recognized risk not in the catalog: writing a Triton attention kernel on Ascend910B4 must still beat the residual native flash-attention dispatch plus host overhead. This intervention deliberately does NOT try to beat the FA math (23.5 us); it targets the ~62 us of pure layout transpose/copy that the native SDPA path adds around the FA kernel. Because seq=83 is tiny, a non-flash attention (materialized 83x83 scores) is correct and simpler than tiled flash attention, lowering implementation risk.

## Rationale and Evidence

Verifier report_000 establishes the baseline: wall ~0.3206 ms, device ~108-120 us/call, device_ratio ~0.33, and ~6.7 kernels/call. Host launch/sync dominates wall, but the device-time breakdown is layout-dominated: three `aclnnFlashAttentionScore_TransposeAiCore_Transpose` kernels (48 us/call) are the `.contiguous()` materialization of the strided `view(...).transpose(1,2)` q/k/v, and one `aclnnInplaceCopy_TransposeAiCore_Transpose` (14 us/call) is the output `transpose(1,2).reshape(...)` copy. Together these four layout kernels are ~62 us of the ~110 us device time, while the actual flash-attention compute is only ~23.5 us. The reference therefore spends more than half its device time shuffling layout that exists only because the caller's `[bsz, seq, hidden]` layout differs from the native FA kernel's required `[bsz, num_heads, seq, head_size]` contiguous layout.

A single Triton kernel can read the original `[bsz, seq, hidden]` contiguous inputs with strided loads (indexing `head*64 + d` directly), compute `softmax(q@k^T * 0.125) @ v` per (batch, head, seq), and write the output in the same `[bsz, seq, hidden]` layout, removing all four layout kernels and collapsing ~6.7 launches toward one. Because seq=83 and head_size=64 are small, the 83x83 score matrix fits trivially on chip, so a non-flash (materialized-softmax) formulation is correct and low-risk, avoiding the complexity of online-softmax tiling. The expected effect is a device-time reduction of roughly 62 us plus a reduction in per-kernel host launch gaps, projected as ~15% wall improvement — comfortably above the 5% adoption threshold. The native FA kernel itself is not the target; the layout overhead around it is.
