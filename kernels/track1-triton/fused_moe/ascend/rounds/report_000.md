# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py`
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `a7fc0001db3ee9e636241954d2c071b62acee518b23f4c59c19efee886203a02`
- Accepted reference SHA256: `a0269ac15833098c11c63045318e829e404b09ca49ae8dce22b244e0d2894d2b`
- Base SHA256: `a0269ac15833098c11c63045318e829e404b09ca49ae8dce22b244e0d2894d2b`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `47e60b0db91c4c67e55f92cf79f5dddf591925620ec4db38704dfb42f0f185dd`
- verification_tier: baseline
- screening_pairs: `not-run (Phase 0 baseline)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass (torch.allclose atol=1e-2, rtol=1e-2) | `PASS accuracy; v0=7.697280 ms, v1=7.290885 ms, speedup=1.056x` | pass | `auto_bench.py --v0_file .../base.py --v1_file .../baseline_adapter.py --warmup 5 --repeat 10 --full-traceback` |

Notes:
- The harness AST loader rewrites `device="cuda"` → `device="npu"` in both `get_inputs`; the `__main__` `.cuda()` call is not executed by the harness. Confirmed baseline actually runs on `npu:0`.
- A `torch_npu` UserWarning (`Cannot create tensor with internal format ...`) is emitted at `expert_out = torch.zeros_like(x_rep)`; it is a benign internal-format warning, not an error, and does not affect correctness.

## Screening Evidence

Not run. Phase 0 establishes the baseline; no screening pair is applicable.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `not-interleaved (Phase 0 baseline: single reference/candidate run)`
- reference_raw_samples_ms: `not-collected (harness reports median only)`
- candidate_raw_samples_ms: `not-collected (harness reports median only)`
- reference_median_ms: `7.158795`
- candidate_median_ms: `7.159420`
- improvement_pct: `-0.008731`

```text
improvement_pct = (7.158795 - 7.159420) / 7.158795 * 100 = -0.008731
```

The harness (`auto_bench.py`) computes and prints the unrounded median via
`statistics.median` but does not emit raw per-iteration samples to stdout.
Baseline wall time (candidate/baseline_adapter median) = **7.159420 ms**;
reference (base.py) median = 7.158795 ms. The two are statistically identical
(0.0087% difference), as expected because `baseline_adapter.py` is `base.py`
with only the top-level `Model` renamed to `ModelNew`.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `not-applicable: Phase 0`
- Hypothesis verdict: `inconclusive`

No round decision exists in Phase 0; there is no Evaluation Contract to mirror.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary` (Level 1)
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`,
  `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable (device time available via CANN msprof)`

Reference (`base.py`) and candidate (`baseline_adapter.py`) scopes were profiled
in separate CANN msprof captures (`ASCEND_WORK_PATH` per scope) and summarized
independently with `summarize_cann_trace.py` (reads `ai_core_op_summary.db`).
All totals below are normalized by `iterations=50`.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| reference (`base.py`) | 37197.38 | 743.948 | 6300 | 126.0 | 7.158795 | 0.103921 |
| candidate (`baseline_adapter.py`) | 37175.84 | 743.517 | 6300 | 126.0 | 7.159420 | 0.103852 |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)
```

### Reference Top Kernels (base.py)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnNonzeroV2_NonzeroAiCore_NonZero | 800 | 16.0 | 12233.98 | 244.680 |
| aclnnNonzeroV2_NonzeroAiCore_MemSet | 800 | 16.0 | 4604.62 | 92.092 |
| aclnnIndex_IndexAiCore_Index | 400 | 8.0 | 4108.94 | 82.179 |
| aclnnIndexPutImpl_IndexPutV2_IndexPutV2 | 400 | 8.0 | 2873.04 | 57.461 |
| aclnnMatmul_MatMulCommon_MatMulV2 | 800 | 16.0 | 2247.10 | 44.942 |
| aclnnTopk_TopkV2AiCore_TopKV2 | 100 | 2.0 | 2108.98 | 42.180 |
| aclnnAny_ReduceAny_ReduceAny | 400 | 8.0 | 1810.86 | 36.217 |
| aclnnMul_MulAiCore_Mul | 450 | 9.0 | 1141.38 | 22.828 |
| aclnnReduceSum_ReduceSumOpAiCore_ReduceSum | 100 | 2.0 | 984.80 | 19.696 |
| aclnnMul_SliceAiCore_Slice | 400 | 8.0 | 804.04 | 16.081 |

### Candidate Top Kernels (baseline_adapter.py)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnNonzeroV2_NonzeroAiCore_NonZero | 800 | 16.0 | 12238.78 | 244.776 |
| aclnnNonzeroV2_NonzeroAiCore_MemSet | 800 | 16.0 | 4620.94 | 92.419 |
| aclnnIndex_IndexAiCore_Index | 400 | 8.0 | 4093.14 | 81.863 |
| aclnnIndexPutImpl_IndexPutV2_IndexPutV2 | 400 | 8.0 | 2880.86 | 57.617 |
| aclnnMatmul_MatMulCommon_MatMulV2 | 800 | 16.0 | 2248.42 | 44.968 |
| aclnnTopk_TopkV2AiCore_TopKV2 | 100 | 2.0 | 2113.58 | 42.272 |
| aclnnAny_ReduceAny_ReduceAny | 400 | 8.0 | 1806.08 | 36.122 |
| aclnnMul_MulAiCore_Mul | 450 | 9.0 | 1146.12 | 22.922 |
| aclnnReduceSum_ReduceSumOpAiCore_ReduceSum | 100 | 2.0 | 988.28 | 19.766 |
| aclnnMul_SliceAiCore_Slice | 400 | 8.0 | 804.78 | 16.096 |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial Phase 0 baseline verification | not-applicable | `a7fc0001db3ee9e636241954d2c071b62acee518b23f4c59c19efee886203a02` | pass |

## evidence_for_next_round

- Baseline wall time is ~7.16 ms (median, 100 repeats, warmup 50), dominated by
  host-side Python + many small NPU kernels: **126 kernels per forward call**.
- The per-expert Python for-loop produces the dominant kernel families:
  `aclnnNonzeroV2` (NonZero + MemSet, ~337 us/call combined, used to build the
  `flat_ids == e` mask), `aclnnIndex` (~82 us/call) and `aclnnIndexPutImpl`
  (~57 us/call) for the `x_rep[mask]` gather and `expert_out[mask] = ...` scatter.
  Together these mask/gather/scatter families account for roughly half of
  device time (~460 us of ~744 us/call).
- 16 MatMul launches per call (`2*E=16`), one per expert w1/w2 GEMM, plus 2
  TopK and 2 ReduceSum per call.
- device_ratio is ~0.104 for both scopes: only ~10% of wall time is attributable
  to AI Core device kernels; the remaining ~90% is host launch/dispatch overhead
  and Python interpreter time — consistent with a large number of tiny kernels
  and host-side loop logic.
- Candidate (baseline_adapter) is byte-equivalent to reference (base) modulo the
  `Model`→`ModelNew` rename; no performance difference is expected or observed.

## Stop Recommendation

- recommendation: `continue`
- evidence: `Phase 0 baseline established; no stop condition met.`

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/fused-moe-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/ascend/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /workspace/kernelswift/.worktrees/fused-moe-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/ascend/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/fused-moe-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/ascend/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/fused_moe/ascend/log/fused_moe_baseline_forward_50iter.pt.trace.json
```

CANN profiler summarization:

```bash
cd /workspace/kernelswift/.worktrees/fused-moe-ascend
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py \
  kernels/track1-triton/fused_moe/ascend/log/profiling_data/baseline_base/profiling_data/<ts>_ascend_pt \
  --iterations 50 --scope baseline_base --wall-ms 7.158795
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py \
  kernels/track1-triton/fused_moe/ascend/log/profiling_data/candidate_baseline_adapter/profiling_data/<ts>_ascend_pt \
  --iterations 50 --scope candidate_baseline_adapter --wall-ms 7.159420
```

Raw profiler trace: `kernels/track1-triton/fused_moe/ascend/log/fused_moe_baseline_forward_50iter.pt.trace.json`
