# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"fuse causal SDPA into a single Triton kernel that computes QK^T, applies the causal mask, softmax, and AV in one launch, eliminating the three TransposeAiCore copies, the standalone aclnnTriu mask, the OnesLike, and the inter-kernel sync waits","allowed_changes":["ModelNew.forward","kernel dataflow"],"invariants":["ModelNew public contract (num_heads=8, head_size=64, scale=None, num_kv_heads=8)","output shape [83,512] and fp16 dtype","causal numerical semantics (scale=1/sqrt(head_size), lower-triangular mask)","get_inputs and get_init_inputs entry points"],"expected_wall_improvement_pct":15.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor q shape=[T,H,D] dtype=fp16 layout=contiguous memory=global
tensor k shape=[T,H,D] dtype=fp16 layout=contiguous memory=global
tensor v shape=[T,H,D] dtype=fp16 layout=contiguous memory=global
tensor out shape=[T,H,D] dtype=fp16 layout=contiguous memory=global
scalar scale dtype=fp32 value=0.125
tile q_tile shape=[1,BLOCK_D] dtype=fp16 memory=register
tile k_tile shape=[BLOCK_K,BLOCK_D] dtype=fp16 memory=register
tile v_tile shape=[BLOCK_K,BLOCK_D] dtype=fp16 memory=register
tile scores shape=[1,BLOCK_K] dtype=fp32 memory=register
tile probs shape=[1,BLOCK_K] dtype=fp32 memory=register
tile acc shape=[1,BLOCK_D] dtype=fp32 memory=register

# O Operations
alloc acc <- zeros([1,BLOCK_D]) dtype=fp32
load q_tile <- q[token, head, 0:D]
load k_tile <- k[block_k*BLOCK_K:(block_k+1)*BLOCK_K, head, 0:D]
load v_tile <- v[block_k*BLOCK_K:(block_k+1)*BLOCK_K, head, 0:D]
compute scores = q_tile @ k_tile^T * scale
compute scores = where(k_idx <= token, scores, -inf)
compute scores = scores - max(scores)
compute probs = exp(scores)
compute probs = probs / sum(probs)
compute acc = acc + probs @ v_tile
store out[token, head, 0:D] <- acc

# C Control
parallel token over T
guard token < T
for block_k over KV_BLOCKS
guard block_k < KV_BLOCKS
end

# H Target Hints
target=triton_ascend
num_warps=1
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change: no buffer cache, allocator reuse, or stream/context ownership change"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse causal SDPA into a single Triton kernel that computes QK^T, applies the causal mask, softmax, and AV in one launch, eliminating the three TransposeAiCore copies, the standalone aclnnTriu mask, the OnesLike, and the inter-kernel sync waits","expected_causal_chain":["transpose and Triu and OnesLike and copy kernels disappear from the candidate scope","kernel_count_per_call decreases from 8.66 toward 1","device_us_per_call decreases","wall_time_ms decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease toward 1"},{"name":"device_us_per_call","expectation":"decrease"},{"name":"aclnnTranspose_kernel_presence","expectation":"absent"},{"name":"aclnnTriu_kernel_presence","expectation":"absent"}],"guardrails":["correctness:pass","output dtype and shape unchanged","causal semantics preserved"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. The three recorded failures (winner-tree selection, sort-32/sort-64 selection network, dynamic gather, cumsum compaction) all concern grouped top-k selection under an MLU590-H8 runtime and do not match the causal-attention fusion preconditions here (Ascend910B4, torch_npu 2.7.1.post4, triton 3.2.0, T=83 causal SDPA). No listed failure invalidates this fusion path.
- Consulted `prompts/coder_targets/triton_ascend.md`. All primitives required by the Sketch are Supported (`tl.load`, `tl.store`, `tl.arange`, `tl.dot` fp32, `tl.exp`, `tl.sum`, `tl.max`, `tl.where`, `tl.static_range`/`for`), with `num_warps=1` as a proven Constrained value. No Unknown primitive (block pointers, async copy, fast launcher, num_stages) is made normative here.
- The MLU sibling's `fast_libentry` host-launcher trick is deliberately NOT requested this round: it is Unknown on Ascend (no probe) and would force a host-scope change that is separately observable and out of scope for this kernel-fusion decision.

## Rationale and Evidence

The accepted report (`rounds/report_000.md`) classifies the baseline as mixed (device_ratio 0.360). The reference scope shows 8.66 kernels per call with dominant device work decomposed around a single fused `aclnnFlashAttentionScore` core (~24.87 us/call): three `aclnnFlashAttentionScore_TransposeAiCore_Transpose` invocations (~47.09 us/call), a standalone causal-mask `aclnnTriu_Triu_Triu` (~26.16 us/call), a final `aclnnInplaceCopy_TransposeAiCore_Transpose` (~13.53 us/call), an `aclnnInplaceOne_OnesLikeAiCore_OnesLike` (~4.88 us/call), plus `EVENT_WAIT_SQE` sync waits (~31.48 us/call) that separate these launches.

All of this decomposition overhead (~91 us/call of pure layout/mask/sync work) is inside the candidate's change boundary: it exists only because the reference materializes the causal mask and performs the pre/post layout transposes as separate library ops around `F.scaled_dot_product_attention`. A single Triton kernel that loads `[T,H,D]` tensors directly, computes `QK^T` with `tl.dot` in fp32, applies the causal mask via `tl.where` (no materialized upper triangle), performs an online-style softmax per query row, and accumulates `AV` with `tl.dot` removes every transpose, the Triu mask, the OnesLike, and the inter-kernel waits in one launch.

Expected effect: device time falls from ~148 us/call toward the single fused kernel (~24-50 us), and the inter-kernel `EVENT_WAIT_SQE` cost is removed. With host/harness time (~261 us) roughly fixed, the ~98 us device saving maps to a wall improvement well above the 5% adoption threshold (estimated ~15%, conservative lower bound comfortably >5%). This mirrors the sibling MLU backend's Round-1 result (22 kernels to 1 launch, 3.81x wall), adjusted for Ascend where the reference is already partially fused (8.66 kernels) so the expected gain is proportionally smaller but still decisive.

Falsifiable mechanism observables: `kernel_count_per_call` must drop toward 1, and the `aclnnTranspose`/`aclnnTriu` kernels must vanish from the candidate profiler scope. Correctness is preserved because the fused kernel computes the same causal SDPA (`scale=1/sqrt(64)=0.125`, lower-triangular mask) and stores the same `[83,512]` fp16 output under the loose `atol=1e-2, rtol=1e-2` tolerance.
