# Decision 003

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"003","reference_implementation":"triton_flexattention_002.py","reference_report":"rounds/report_002.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"dot-bmm"}
```

## Optimization Intent

```json
{"bottleneck_class":"device-bound","intervention":"replace the elementwise tl.sum rank-1 reductions for QK^T and AV with tl.dot matrix multiplies on a multi-token-per-program layout (BLOCK_M=16 tokens per program), routing both matmuls through the Ascend Cube (BMM) hardware unit instead of the vector path","allowed_changes":["ModelNew.forward grid mapping","kernel dataflow (elementwise-reduce to tl.dot)"],"invariants":["ModelNew public contract (num_heads=8, head_size=64, scale=None, num_kv_heads=8)","output shape [83,512] and fp16 dtype","causal numerical semantics (scale=1/sqrt(head_size), lower-triangular mask)","output buffer cache behavior (host plan from round 002)","get_inputs and get_init_inputs entry points"],"expected_wall_improvement_pct":8.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor q shape=[T,H,D] dtype=fp16 layout=contiguous memory=global
tensor k shape=[T,H,D] dtype=fp16 layout=contiguous memory=global
tensor v shape=[T,H,D] dtype=fp16 layout=contiguous memory=global
tensor out shape=[T,H,D] dtype=fp16 layout=contiguous memory=global
scalar scale dtype=fp32 value=0.125
tile q_tile shape=[BLOCK_M,BLOCK_D] dtype=fp32 memory=register
tile k_tile shape=[BLOCK_KV,BLOCK_D] dtype=fp32 memory=register
tile v_tile shape=[BLOCK_KV,BLOCK_D] dtype=fp32 memory=register
tile scores shape=[BLOCK_M,BLOCK_KV] dtype=fp32 memory=register
tile probs shape=[BLOCK_M,BLOCK_KV] dtype=fp32 memory=register
tile acc shape=[BLOCK_M,BLOCK_D] dtype=fp32 memory=register

# O Operations
load q_tile <- q[m_off, head, 0:D]
load k_tile <- k[k_off, head, 0:D]
load v_tile <- v[k_off, head, 0:D]
compute scores = tl.dot(q_tile, trans(k_tile)) * scale
compute scores = where(k_idx <= token_idx, scores, -inf)
compute scores = scores - max(scores, axis=1)
compute probs = exp(scores)
compute probs = probs / sum(probs, axis=1)
compute acc = tl.dot(probs, v_tile)
store out[m_off, head, 0:D] <- acc

# C Control
parallel pid over TM_BLOCKS*H
guard pid < TM_BLOCKS*H
guard m_off + BLOCK_M <= T
for head over H
end

# H Target Hints
target=triton_ascend
num_warps=1
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change: output buffer cache from round 002 is retained unchanged; no new allocator, stream, or context behavior"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-003","intervention":"replace the elementwise tl.sum rank-1 reductions for QK^T and AV with tl.dot matrix multiplies on a multi-token-per-program layout (BLOCK_M=16 tokens per program), routing both matmuls through the Ascend Cube (BMM) hardware unit instead of the vector path","expected_causal_chain":["QK^T and AV matmuls run on the Cube matrix unit instead of elementwise vector reductions","device_us_per_call decreases from ~54 us toward the ~25 us fused core floor","wall_time_ms decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"device_us_per_call","expectation":"decrease"},{"name":"kernel_count_per_call","expectation":"remain 1 (still a single fused kernel)"},{"name":"output_allocations_per_call","expectation":"remain 0 (host cache unchanged)"}],"guardrails":["correctness:pass","output dtype and shape unchanged","causal semantics preserved","kernel_count_per_call unchanged at 1","output buffer cache behavior unchanged"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. The recorded failures (winner-tree selection, sort-32/sort-64 selection network, dynamic gather, cumsum compaction) are all grouped-top-k *selection* optimizations that regressed on MLU590-H8, and do not bear on routing QK^T/AV GEMMs through `tl.dot`. No listed failure invalidates this path.
- Consulted `prompts/coder_targets/triton_ascend.md`. `tl.dot` fp32 is Supported with the probe proving `(16,16)@(16,16)`; the sketch uses M=BLOCK_M=16 (the proven M dimension) with contraction/output dims K=64 and N=128. This is the same dtype (fp32) and same M as the probe, with a larger but structurally identical contraction. If the Coder finds that `(16,64)@(64,128)` does not compile or run on this runtime, that is a legitimate `capability-miss` to report back, not a silent substitution.
- Consulted `references/bottleneck-judgment.md` (fused_moe worked example rounds 3-5) and the groupedtopk-ascend `decision_003.md` abort. The groupedtopk abort proved the *host* launch/dispatch path (~107 us fixed, `fast_libentry` Unknown) is incompressible on this exact runtime. Crucially, that abort did **not** rule out device-side work: groupedtopk's device kernel was already near its floor (35 us, no structural 2x headroom) and its available device optimizations were selection networks that regress. This flexattention campaign differs: the device kernel is 54.64 us, ~2x the ~25 us `aclnnFlashAttentionScore` core, and the headroom maps to a specific proven primitive (`tl.dot` Cube path) rather than a regression-prone selection network.

## Rationale and Evidence

`rounds/report_002.md` shows the candidate is host-bound (device_ratio 0.194), with host ~227 us/call dominated by the harness `sync_devices()` and the fixed Triton launch/dispatch path, while the forward's own host enqueue is already negligible (~0.04 us/call). The groupedtopk-ascend campaign on this identical runtime already aborted after proving that host path is incompressible (`fast_libentry`/stream/context are `Unknown`, direct launch already in use). Host-side optimization is therefore exhausted — no falsifiable host intervention with ≥5% expected gain remains.

The one remaining, evidence-backed lever is the device kernel itself. It still uses `tl.sum(q[None,:] * k, axis=1)` and `tl.sum(probs[:,None] * v, axis=0)` — elementwise multiply-plus-reduce — for the two GEMMs, running entirely on the vector path with `num_warps=1`. The Ascend910B4 has 20 Cube (matrix) cores (`cube_core_num=20`) that this path never touches. The theoretical `aclnnFlashAttentionScore` core from report_000 runs the same fused QK^T/softmax/AV at ~25 us, confirming ~30 us of structural device headroom.

Routing both matmuls through `tl.dot` on a multi-token-per-program layout (BLOCK_M=16, giving `tl.dot` a non-degenerate M=16 — the exact M dimension the profile already proves) moves QK^T and AV onto the Cube matrix unit. The MLU sibling's Round 2 did precisely this and cut device time ~47% (96.2 → 50.9 us) for a ~20% wall gain; the fused_moe sibling's Round 4 likewise moved GEMM to `tl.dot` and cut device time. This is the strongest matched cross-backend evidence available.

Expected effect: device falls from 54.64 us toward ~30 us (a ~25 us saving, conservative — a partial win toward the 25 us core). With host ~227 us fixed, wall falls from 281.9 us toward ~257 us, an ~8.8% improvement, clearing the 5% threshold. The mechanism is falsifiable: `device_us_per_call` must decrease while `kernel_count_per_call` stays 1 and the output-buffer cache stays at 0 allocations — isolating the change to the kernel dataflow. Correctness is preserved because the multi-token layout computes the identical causal SDPA (`scale=0.125`, lower-triangular mask, fp32 accumulation) and stores the same `[83,512]` fp16 output; the BLOCK_M=16 tiling only changes how the 83 tokens are partitioned across programs (ceil(83/16)=6 token blocks × 8 heads), with masked guards for the final partial block.
