# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_centre_random_aug_001.py`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `23bb4a4e3b2830b7023216c5485b9fbf447ddf2f2ce62141697fbc21561cd31b`
- Candidate SHA256: `dcfeb039d3d8526d756775015560a22e1b0cd447c5c6dbd69ad12d3a3f0ee089`
- Accepted reference SHA256: `7d4a79ae96328fc03a4489710f68b7f639ddea9cbd5c0f7bb45e1cec5472061b`
- Base SHA256: `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `33ca785dc0b312e3d097f16bc2ea7de8f8d2dac779c04c2ac028a001f2b8aa4d`
- verification_tier: authoritative
- screening_pairs: `not-run`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | allclose(atol=1e-2,rtol=1e-2) | PASS (3/3 pairs) | pass | `PASS accuracy` x3 |
| output shape/dtype | [4,256,3] fp32 | PASS | pass | candidate line 180 |
| R/T RNG bitwise | 3x rand(4) + 1x randn(4,3) same order | PASS (R/T + Sin/Cos/Sqrt stay in torch) | pass | candidate lines 95-129,174-175 |
| center == mean(dim=-2) | matches base.py | PASS | pass | candidate lines 162-166 |
| ModelNew contract | n_sample=4, s_trans=1.0, centre_only=False | PASS | pass | candidate line 148 |

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (3 harness invocations)
- reference_raw_samples_ms (per-pair medians): `[2.463270, 2.490540, 2.457635]`
- candidate_raw_samples_ms (per-pair medians): `[2.066240, 2.010540, 2.023920]`
- reference_median_ms: `2.463270`
- candidate_median_ms: `2.023920`
- improvement_pct: `17.84`

```text
improvement_pct = (2.463270 - 2.023920) / 2.463270 * 100 = 17.84
```

17.84% >= 5.0% threshold -> accepted.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `fuse centering, 3x3 matvec, translation add, mask multiply into one Triton kernel over [4,256,3]`
- expected_causal_chain: `tiny elementwise kernels collapse -> kernel_count 110 toward ~20 -> host launch overhead falls -> wall time decreases`
- primary_metric: `wall_time`
- Hypothesis verdict: `partially-confirmed`

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| kernel_count_per_call | 110 -> <=25 | 110 -> 64 (decreased, NOT <=25) | fail | CANN 110.0 vs 64.0 |
| device_us_per_call | ~292us decrease | 294.97 -> 216.06 us (-26.8%) | pass | CANN summary |
| host_launch_count_per_call | decrease | 110 -> 64 launches/call | pass | kernel_count 5500 -> 3200 |
| wall_time | ~30% improvement | 17.84% (>= 5%) | pass | authoritative timing |

`partially-confirmed`: the fused kernel collapsed ~46 deterministic kernels into one, device_us_per_call fell 26.8%, wall improved 17.84%. But kernel_count_per_call = 64 (not <=25) because the R/T + quaternion Sin/Cos/Sqrt path and `contiguous()`/`empty`/host-transfer kernels (~63) remain in torch by design (RNG bitwise preservation).

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available`
- iterations: `50`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (baseline_adapter) | 14748.48 | 294.970 | 5500 | 110.0 | 2.463270 | 0.1197 |
| candidate (triton_001) | 10803.16 | 216.063 | 3200 | 64.0 | 2.023920 | 0.1067 |

```text
device_ratio = device_us_per_call / (median_ms * 1000)
reference: 0.1197, candidate: 0.1067
```

### Accepted Reference Top Kernels (baseline_adapter)

| Kernel | Count/call | Total us | Us/call |
|---|---:|---:|---:|
| aclnnMul_StridedSliceAiCore_StridedSlice | 9.0 | 3243.36 | 64.867 |
| aclnnMul_MulAiCore_Mul | 24.0 | 2545.66 | 50.913 |
| aclnnMul_BroadcastToAiCore_BroadcastTo | 9.0 | 1720.22 | 34.404 |
| aclnnReduceSum_ReduceSumOpAiCore_ReduceSum | 2.0 | 1718.44 | 34.369 |
| aclnnAdd_AddAiCore_Add | 13.0 | 1222.58 | 24.452 |
| aclnnMuls_MulAiCore_Mul | 14.0 | 809.46 | 16.189 |
| aclnnSub_SubAiCore_Sub | 4.0 | 725.44 | 14.509 |
| aclnnMul_SliceAiCore_Slice | 9.0 | 632.76 | 12.655 |
| PCIE_DMA_SQE | 4.0 | 467.24 | 9.345 |

### Candidate Top Kernels (triton_001)

| Kernel | Count/call | Total us | Us/call |
|---|---:|---:|---:|
| _centre_aug_linear_kernel | 1.0 | 6098.90 | 121.978 |
| aclnnMul_MulAiCore_Mul | 14.0 | 881.42 | 17.628 |
| aclnnReduceSum_ReduceSumOpAiCore_ReduceSum | 2.0 | 816.04 | 16.321 |
| aclnnMuls_MulAiCore_Mul | 14.0 | 766.84 | 15.337 |
| PCIE_DMA_SQE | 4.0 | 463.86 | 9.277 |
| aclnnAdd_AddAiCore_Add | 6.0 | 312.78 | 6.256 |
| aclnnRsubs_SubAiCore_Sub | 5.0 | 293.66 | 5.873 |
| aclnnSqrt_SqrtAiCore_Sqrt | 4.0 | 190.00 | 3.800 |

The fused `_centre_aug_linear_kernel` (1 launch/call) is now the largest device-time consumer (~122 us/call, 56% of candidate device total).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial authoritative verification | dcfeb039d3d8526d756775015560a22e1b0cd447c5c6dbd69ad12d3a3f0ee089 | unchanged | accepted |

No verifier-to-coder repair required.

## evidence_for_next_round

- Wall improved 17.84% (2.463270 -> 2.023920 ms), exceeding 5% threshold. H-001 partially-confirmed.
- Fused `_centre_aug_linear_kernel` is now the single largest device-time consumer (~122 us/call, 56% of device total), a single launch over [4,256,3]=1024 rows (BLOCK=256, 4 programs, num_warps=4) — potentially under-parallelized vs 20 cube/40 vector cores.
- kernel_count_per_call = 64 remains, dominated by torch R/T + quaternion Sin/Cos/Sqrt path plus `contiguous()`/`empty`/host-transfer launches — left in torch to preserve seeded RNG bitwise.
- Device ratio still ~0.107: wall time remains host-launch-bound; further reduction likely requires cutting the ~63 remaining torch launches WITHOUT breaking RNG draw order (the hard correctness constraint).

## Stop Recommendation

- recommendation: `continue`
- evidence: 17.84% wall improvement; device still ~10.7% of wall, host-bound overhead remains the bottleneck.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/centre-random-aug-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/centre_random_augmentation/base.py --v1_file kernels/track1-triton/centre_random_augmentation/ascend/triton_centre_random_aug_001.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/centre-random-aug-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/centre_random_augmentation/base.py --v1_file kernels/track1-triton/centre_random_augmentation/ascend/triton_centre_random_aug_001.py --profile --profile-reference-file kernels/track1-triton/centre_random_augmentation/ascend/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/centre_random_augmentation/ascend/log/round_001_forward_50iter.pt.trace.json
```
