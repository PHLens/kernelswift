# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_fused_moe_001.py`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `821c40436ba4af5ed82029405060a4a55b5c3165c5da7d7c26ce4976136218f1`
- Candidate SHA256: `e42d811a7aee32f3fb34b19c00f88bf7922129faccc3d670dd07abc2df443287`
- Accepted reference SHA256: `a7fc0001db3ee9e636241954d2c071b62acee518b23f4c59c19efee886203a02`
- Base SHA256: `a0269ac15833098c11c63045318e829e404b09ca49ae8dce22b244e0d2894d2b`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `47e60b0db91c4c67e55f92cf79f5dddf591925620ec4db38704dfb42f0f185dd`
- verification_tier: authoritative
- screening_pairs: `not-run (candidate correctness passes and is clearly faster; proceeded directly to authoritative timing)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass (torch.allclose atol=1e-2, rtol=1e-2) | `PASS accuracy; v0=8.190205 ms, v1=0.612535 ms, speedup=13.371x` | pass | `auto_bench.py --v0_file base.py --v1_file triton_fused_moe_001.py --warmup 5 --repeat 10 --full-traceback` |
| output shape [83,128] fp16 unchanged | preserved | correctness PASS implies shape/dtype match (harness `compare_values` checks shape + dtype) | pass | harness `compare_values` |
| softmax+topk+renormalize routing preserved | preserved | routing code in `forward` is byte-identical to base.py (softmax → topk → renormalize → fp16 cast) | pass | `triton_fused_moe_001.py` forward lines 90-95 |
| weighted top-k reduce over exactly 2 experts | preserved | kernel loops `tl.static_range(0, K)` with K=top_k=2, accumulates `weight * out_k` | pass | `triton_fused_moe_001.py` kernel lines 39-66 |

Note: a `torch_npu` UserWarning is emitted from the **reference** path (`base.py:61 expert_out = torch.zeros_like(x_rep)`); it is benign and does not affect correctness.

## Screening Evidence

Not run. Candidate correctness passed and the first authoritative timing pair already
showed ~13x speedup, so no 10%-slower screen was possible; proceeded directly to
authoritative timing.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `three reference/candidate pairs (reference = base.py, candidate = triton_fused_moe_001.py)`
- reference_raw_samples_ms: `not-collected (harness reports median only)`
- candidate_raw_samples_ms: `not-collected (harness reports median only)`

| Pair | Reference wall ms | Candidate wall ms | Evidence |
|---:|---:|---:|---|
| 1 | 7.816390 | 0.588975 | `auto_bench.py --v0_file base.py --v1_file triton_fused_moe_001.py --warmup 50 --repeat 100` |
| 2 | 8.126030 | 0.547805 | same |
| 3 | 7.242815 | 0.569590 | same |

- reference_median_ms: `7.816390`
- candidate_median_ms: `0.569590`
- improvement_pct: `92.712876`

```text
improvement_pct = (7.816390 - 0.569590) / 7.816390 * 100 = 92.712876
```

The unrounded improvement (92.71%) far exceeds the 5% adoption threshold.
Correctness passed, so the terminal result is `accepted`.

Note on reference selection: the harness requires `--v0_file` to define `Model`,
so `baseline_adapter.py` (which defines `ModelNew`) cannot be used directly as the
`--v0_file`. Phase 0 established that `baseline_adapter.py` ≡ `base.py`
numerically and in wall time (7.158795 ms vs 7.159420 ms, 0.0087% apart). The
authoritative timing therefore uses `base.py` as the `Model`-defining reference,
which is the canonical accepted reference for wall-time comparison. The
`baseline_adapter.py` is profiled as the reference scope via
`--profile-reference-file`.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `fuse the per-expert feedforward (gate/up GEMM, SiLU gating, down GEMM) and weighted reduce into a single per-token Triton kernel using elementwise tl.sum outer-products, eliminating the mask/gather/scatter dispatch and the 16 per-expert MatMul launches`
- expected_causal_chain: `per-expert loop disappears → kernel_count 126→1 → device_us ~744→lower → wall_time decreases`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| kernel_count_per_call | decrease from 126 toward 1 | 126 → **12** | pass | candidate scope summary: kernel_count_per_call=12.0 (vs reference 126.0) |
| device_us_per_call | decrease from ~744 us/call | 746.6 → **97.4** | pass | candidate device_us_per_call=97.366 vs reference 746.641 |
| aclnnNonzeroV2_presence | absent from candidate scope | absent (0 occurrences) | pass | candidate top-kernels has no `aclnnNonzeroV2*` entries |
| aclnnIndexPutImpl_presence | absent from candidate scope | absent (0 occurrences) | pass | candidate top-kernels has no `aclnnIndexPutImpl*` entries |

Note: kernel_count did not reach ~1 because routing (softmax + topk + renormalize
+ casts) still runs in PyTorch inside `forward`. The candidate scope shows the
single fused `_fused_moe_per_token_kernel` (1/call) plus ~11 routing kernels
(2 TopK, 1 Softmax, 4 Cast, 1 ReduceSum, 1 GatherElements, 1 Div, 1 TopK Cast),
for 12 total. The per-expert loop kernels (16x aclnnMatmul, NonzeroV2, Index,
IndexPutImpl) are fully eliminated, exactly as the intervention predicted.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted` (Level 1)
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`,
  `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable (device time available via CANN msprof)`

Reference and candidate profiled in separate CANN msprof captures (via
`--profile-reference-file baseline_adapter.py`), summarized independently with
`summarize_cann_trace.py`. All totals normalized by `iterations=50`.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| reference (baseline_adapter.py) | 37332.06 | 746.641 | 6300 | 126.0 | 7.816390 | 0.095523 |
| candidate (triton_fused_moe_001.py) | 4868.30 | 97.366 | 600 | 12.0 | 0.569590 | 0.170941 |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)
```

### Reference Top Kernels (baseline_adapter.py)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnNonzeroV2_NonzeroAiCore_NonZero | 800 | 16.0 | 12285.02 | 245.700 |
| aclnnNonzeroV2_NonzeroAiCore_MemSet | 800 | 16.0 | 4582.80 | 91.656 |
| aclnnIndex_IndexAiCore_Index | 400 | 8.0 | 4342.36 | 86.847 |
| aclnnIndexPutImpl_IndexPutV2_IndexPutV2 | 400 | 8.0 | 2759.72 | 55.194 |
| aclnnMatmul_MatMulCommon_MatMulV2 | 800 | 16.0 | 2244.36 | 44.887 |
| aclnnTopk_TopkV2AiCore_TopKV2 | 100 | 2.0 | 2075.18 | 41.504 |
| aclnnAny_ReduceAny_ReduceAny | 400 | 8.0 | 1827.96 | 36.559 |
| aclnnMul_MulAiCore_Mul | 450 | 9.0 | 1038.58 | 20.772 |
| aclnnReduceSum_ReduceSumOpAiCore_ReduceSum | 100 | 2.0 | 957.78 | 19.156 |
| aclnnEqScalar_EqualAiCore_Equal | 400 | 8.0 | 787.98 | 15.760 |

### Candidate Top Kernels (triton_fused_moe_001.py)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnTopk_TopkV2AiCore_TopKV2 | 100 | 2.0 | 1957.92 | 39.158 |
| _fused_moe_per_token_kernel | 50 | 1.0 | 1004.68 | 20.094 |
| aclnnSoftmax_SoftmaxAiCore_SoftmaxV2 | 50 | 1.0 | 562.88 | 11.258 |
| aclnnInplaceCopy_CastAiCore_Cast | 200 | 4.0 | 545.46 | 10.909 |
| aclnnReduceSum_ReduceSumOpAiCore_ReduceSum | 50 | 1.0 | 354.04 | 7.081 |
| aclnnTopk_GatherElements_GatherElements | 50 | 1.0 | 311.14 | 6.223 |
| aclnnDiv_RealDivAiCore_RealDiv | 50 | 1.0 | 79.78 | 1.596 |
| aclnnTopk_CastAiCore_Cast | 50 | 1.0 | 52.40 | 1.048 |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial Round 1 verification | `e42d811a7aee32f3fb34b19c00f88bf7922129faccc3d670dd07abc2df443287` | `e42d811a7aee32f3fb34b19c00f88bf7922129faccc3d670dd07abc2df443287` | correctness PASS, accepted |

No repair was required.

## evidence_for_next_round

- The per-token Triton fusion removed the mask/gather/scatter and 16 MatMul
  launches: kernel_count 126 → 12, device_us 746.6 → 97.4, wall 7.82 → 0.57 ms
  (92.7% improvement). Confirmed.
- The candidate's device time is now dominated by **routing** kernels
  (aclnnTopk ~39 us, aclnnSoftmax ~11 us, aclnnInplaceCopy_Cast ~11 us,
  aclnnReduceSum ~7 us, aclnnTopk_GatherElements ~6 us), which still run in
  PyTorch. The fused Triton kernel itself is only ~20 us/call.
- candidate device_ratio rose to 0.171 (vs 0.096 for reference) because wall time
  dropped more than device time; a meaningful fraction of the remaining ~0.57 ms
  wall is now routing + host-side Python dispatch (softmax/topk/casts).
- Next bottleneck: routing (softmax + topk + renormalize + fp16 casts) is the
  dominant remaining device cost and the main remaining kernel count (~11 of 12
  kernels). Fusing or reducing the routing path is the natural next target, but
  Verifier records evidence only.

## Stop Recommendation

- recommendation: `continue`
- evidence: `large accepted improvement (92.7%); no stop condition met (not target-reached, streak 0, round 1 of 20).`

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/fused-moe-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/ascend/triton_fused_moe_001.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /workspace/kernelswift/.worktrees/fused-moe-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/ascend/triton_fused_moe_001.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/fused-moe-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/ascend/triton_fused_moe_001.py --warmup 50 --repeat 100 --profile --profile-reference-file kernels/track1-triton/fused_moe/ascend/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/fused_moe/ascend/log/fused_moe_round_001_forward_50iter.pt.trace.json
```

CANN profiler summarization:

```bash
cd /workspace/kernelswift/.worktrees/fused-moe-ascend
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py \
  kernels/track1-triton/fused_moe/ascend/log/profiling_data/reference_baseline_adapter/profiling_data/<ts>_ascend_pt \
  --iterations 50 --scope reference_baseline_adapter --wall-ms 7.816390
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py \
  kernels/track1-triton/fused_moe/ascend/log/profiling_data/candidate_triton_fused_moe_001/profiling_data/<ts>_ascend_pt \
  --iterations 50 --scope candidate_triton_fused_moe_001 --wall-ms 0.569590
```

Raw profiler trace: `kernels/track1-triton/fused_moe/ascend/log/fused_moe_round_001_forward_50iter.pt.trace.json`
