# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py` (Phase 0 baseline establishment)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `7d4a79ae96328fc03a4489710f68b7f639ddea9cbd5c0f7bb45e1cec5472061b`
- Accepted reference SHA256: `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553`
- Base SHA256: `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `pending` (to be finalized by Orchestrator)
- verification_tier: baseline
- screening_pairs: `not-run (Phase 0 baseline, no candidate to screen)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | `torch.allclose(v0, v1, atol=1e-2, rtol=1e-2)` | PASS — outputs identical | pass | `PASS accuracy; v0=2.547680 ms, v1=2.565115 ms, speedup=0.993x` |

### Randomness / Correctness Gate Observation (KEY)

The operator's `forward` draws fresh random numbers on every call:
- `random_rotation_matrices` calls `torch.rand` 3 times (u1, u2, u3) for the quaternion.
- `centre_random_augmentation` calls `torch.randn` once for the translation `T`.
Output is therefore NON-DETERMINISTIC across calls in principle.

**What the harness actually does** (verified empirically):
1. `run_forward` (harness line 450) calls `set_seed(seed=42)` immediately before EACH forward invocation, and `time_forward` (line 470) reseeds before each timed call. So every v0 and v1 forward draws from a fresh, identical seeded RNG state.
2. `compare_values` (line 300) uses `torch.allclose(atol=1e-2, rtol=1e-2)` on fp32 tensors.
3. Because `base.py` and `baseline_adapter.py` are functionally identical (same RNG call order: 3x `torch.rand` then 1x `torch.randn`), the same seed yields identical random draws, hence identical outputs, and `allclose` passes trivially.

**Conclusion**: A faithful baseline IS established. Correctness passes because the harness seeds per-call with a fixed seed (`--seed 42`), so the two implementations draw the SAME random numbers. This does NOT depend on any shape-only comparison — it is a true value comparison that succeeds only because both files share identical RNG structure.

**Implication for future rounds**: any candidate (triton_operator_NNN.py) that reproduces the SAME RNG call order AND the same torch RNG consumption will produce identical values and pass `allclose`. A candidate that consumes a different number/order of RNG draws (e.g. a fused/reordered random generation) will draw DIFFERENT random numbers than base.py and FAIL the value comparison — even if mathematically correct — because `atol/rtol=1e-2` cannot absorb entirely different rotation matrices/translations (differences are O(1)). This is the central correctness hazard for this operator.

## Screening Evidence

`not-run: Phase 0 baseline establishment has no candidate to screen.`

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (harness runs v0 then v1; for baseline these are base.py vs baseline_adapter.py)
- reference_raw_samples_ms: `[harness reports median only; median=2.547680]`
- candidate_raw_samples_ms: `[harness reports median only; median=2.565115]`
- reference_median_ms: `2.547680`
- candidate_median_ms: `2.565115`
- improvement_pct: `-0.6846` (baseline_adapter is 0.68% slower; expected — functionally identical, within noise)

```text
improvement_pct = (2.547680 - 2.565115) / 2.547680 * 100 = -0.6846
```

Note: the harness prints only the median, not the raw per-sample list. `time_forward` collects 100 samples and returns `statistics.median`. Raw sample array is not surfaced by the harness CLI; median is the authoritative primary metric per project.md (`primary_metric: unrounded median wall_time_ms`).

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `not-applicable: Phase 0`
- Hypothesis verdict: `inconclusive` (no round decision exists yet)

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available` (CANN ai_core_op_summary.db)
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable (device_time_available=true on Ascend)`

Reference and candidate scopes were captured in SEPARATE CANN profiling_data directories (one per scope), each summarized independently with `summarize_cann_trace.py` (nanoseconds -> us, max across task_id).

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (base.py) | 14593.28 | 291.866 | 5500 | 110.0 | 2.547680 | 0.1146 |
| candidate (baseline_adapter.py) | 14575.06 | 291.501 | 5500 | 110.0 | 2.565115 | 0.1136 |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)
reference: 291.866 / 2547.680 = 0.1146
candidate: 291.501 / 2565.115 = 0.1136
```

### Accepted Reference Top Kernels (base.py scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnMul_StridedSliceAiCore_StridedSlice | 450 | 9.0 | 3198.30 | 63.966 |
| aclnnMul_MulAiCore_Mul | 1200 | 24.0 | 2511.72 | 50.234 |
| aclnnReduceSum_ReduceSumOpAiCore_ReduceSum | 100 | 2.0 | 1810.32 | 36.206 |
| aclnnMul_BroadcastToAiCore_BroadcastTo | 450 | 9.0 | 1681.22 | 33.624 |
| aclnnAdd_AddAiCore_Add | 650 | 13.0 | 1147.58 | 22.952 |
| aclnnMuls_MulAiCore_Mul | 700 | 14.0 | 803.92 | 16.078 |
| aclnnSub_SubAiCore_Sub | 200 | 4.0 | 691.82 | 13.836 |
| aclnnMul_SliceAiCore_Slice | 450 | 9.0 | 622.58 | 12.452 |
| PCIE_DMA_SQE | 200 | 4.0 | 480.74 | 9.615 |
| aclnnRsubs_SubAiCore_Sub | 250 | 5.0 | 301.54 | 6.031 |
| aclnnInplaceCopy_BroadcastToAiCore_BroadcastTo | 50 | 1.0 | 242.62 | 4.852 |
| aclnnStack_PackAiCore_Pack | 100 | 2.0 | 231.10 | 4.622 |
| aclnnSqrt_SqrtAiCore_Sqrt | 200 | 4.0 | 185.86 | 3.717 |
| aclnnCos_CosAiCore_Cos | 100 | 2.0 | 178.74 | 3.575 |
| aclnnSin_SinAiCore_Sin | 100 | 2.0 | 174.30 | 3.486 |
| aclnnInplaceUniform_DSARandomUniform_DSARandomUniform | 150 | 3.0 | 163.98 | 3.280 |
| aclnnAdds_AddAiCore_Add | 50 | 1.0 | 56.98 | 1.140 |
| aclnnInplaceNormal_DSARandomNormal_DSARandomNormal | 50 | 1.0 | 55.58 | 1.112 |
| aclnnDiv_RealDivAiCore_RealDiv | 50 | 1.0 | 54.38 | 1.088 |

### Candidate Top Kernels (baseline_adapter.py scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnMul_StridedSliceAiCore_StridedSlice | 450 | 9.0 | 3195.50 | 63.910 |
| aclnnMul_MulAiCore_Mul | 1200 | 24.0 | 2498.02 | 49.960 |
| aclnnReduceSum_ReduceSumOpAiCore_ReduceSum | 100 | 2.0 | 1822.90 | 36.458 |
| aclnnMul_BroadcastToAiCore_BroadcastTo | 450 | 9.0 | 1678.80 | 33.576 |
| aclnnAdd_AddAiCore_Add | 650 | 13.0 | 1150.82 | 23.016 |
| aclnnMuls_MulAiCore_Mul | 700 | 14.0 | 805.16 | 16.103 |
| aclnnSub_SubAiCore_Sub | 200 | 4.0 | 688.42 | 13.768 |
| aclnnMul_SliceAiCore_Slice | 450 | 9.0 | 624.50 | 12.490 |
| PCIE_DMA_SQE | 200 | 4.0 | 474.70 | 9.494 |
| aclnnRsubs_SubAiCore_Sub | 250 | 5.0 | 301.86 | 6.037 |
| aclnnInplaceCopy_BroadcastToAiCore_BroadcastTo | 50 | 1.0 | 234.94 | 4.699 |
| aclnnStack_PackAiCore_Pack | 100 | 2.0 | 230.84 | 4.617 |
| aclnnSqrt_SqrtAiCore_Sqrt | 200 | 4.0 | 185.36 | 3.707 |
| aclnnCos_CosAiCore_Cos | 100 | 2.0 | 179.52 | 3.590 |
| aclnnSin_SinAiCore_Sin | 100 | 2.0 | 174.62 | 3.492 |
| aclnnInplaceUniform_DSARandomUniform_DSARandomUniform | 150 | 3.0 | 162.90 | 3.258 |
| aclnnAdds_AddAiCore_Add | 50 | 1.0 | 56.68 | 1.134 |
| aclnnInplaceNormal_DSARandomNormal_DSARandomNormal | 50 | 1.0 | 55.32 | 1.106 |
| aclnnDiv_RealDivAiCore_RealDiv | 50 | 1.0 | 54.20 | 1.084 |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial Phase 0 baseline verification | not-applicable | 7d4a79ae96328fc03a4489710f68b7f639ddea9cbd5c0f7bb45e1cec5472061b | pass |

## evidence_for_next_round

- Baseline wall median: base.py `2.547680 ms`, baseline_adapter.py `2.565115 ms` (functionally identical, within noise).
- Baseline device time: `~291.9 us/call`, `110 kernels/call`, `device_ratio ~0.114` — device time is only ~11.4% of wall time, so the forward is heavily HOST/launch-bound (small tensor shapes: 256x3 in, 4x256x3 out; 110 kernel launches per call dominate).
- Dominant device kernels are tiny elementwise/stride ops (Mul_StridedSlice, Mul, ReduceSum, BroadcastTo, Add, Muls, Sub, Slice) plus RNG (`InplaceUniform` x3 for quaternion, `InplaceNormal` x1 for translation) and `Sin`/`Cos`/`Sqrt` for the quaternion->matrix conversion — all small, latency-bound launches.
- **Randomness hazard (blocking fact for future rounds)**: any candidate must reproduce the EXACT torch RNG consumption order (3x `torch.rand` then 1x `torch.randn`, same shapes) to pass the `allclose` gate under the harness's per-call `set_seed(42)`. A candidate that reorders, fuses, or changes the count of RNG draws will produce O(1)-different outputs and FAIL correctness, regardless of mathematical equivalence. This is the central correctness constraint for the optimization effort.
- Bottleneck hypothesis (observed, not prescribed): wall time >> device time, suggesting kernel-launch/host overhead dominates; but note the many tiny kernels are inherent to the elementwise math and RNG decomposition, not just launch overhead.

## Stop Recommendation

- recommendation: `continue`
- evidence: Phase 0 baseline established; optimization loop should proceed to Round 1 with full awareness of the RNG-order correctness constraint.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/centre-random-aug-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/centre_random_augmentation/base.py --v1_file kernels/track1-triton/centre_random_augmentation/ascend/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/centre-random-aug-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/centre_random_augmentation/base.py --v1_file kernels/track1-triton/centre_random_augmentation/ascend/baseline_adapter.py --profile --profile-reference-file kernels/track1-triton/centre_random_augmentation/ascend/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/centre_random_augmentation/ascend/log/round_000_forward_50iter.pt.trace.json
```

```bash
cd /workspace/kernelswift/.worktrees/centre-random-aug-ascend/kernels/track1-triton/centre_random_augmentation/ascend
python3 /workspace/kernelswift/skills/kernel-opt-loop/scripts/summarize_cann_trace.py log/profiling_data/reference_baseline_adapter/profiling_data --iterations 50 --scope reference_baseline_adapter
python3 /workspace/kernelswift/skills/kernel-opt-loop/scripts/summarize_cann_trace.py log/profiling_data/candidate_baseline_adapter/profiling_data --iterations 50 --scope candidate_baseline_adapter
```
