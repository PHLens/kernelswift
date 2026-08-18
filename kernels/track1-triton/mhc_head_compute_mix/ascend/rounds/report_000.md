# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py` (Phase 0 reference)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `c3cc90deb93c69f092e876ddc0057ddf6c2d7ac28020f1b7fd1ed260bca72fee`
- Accepted reference SHA256: `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5`
- Base SHA256: `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `pending` (Phase 0 baseline establishment)
- verification_tier: baseline
- screening_pairs: `not-run`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | baseline_adapter output matches base within atol=1e-2, rtol=1e-2 | PASS accuracy | pass | `auto_bench.py ... --warmup 50 --repeat 100` exit 0 |

Conformance, correctness, and every declared guardrail must pass before adoption.
No additional guardrails are declared for Phase 0 (base.py defines no explicit
tolerance; harness default comparison applies).

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `reference/candidate (harness single-shot, baseline tier)`
- reference_raw_samples_ms: `[3.396440]` (harness reports median only; single-shot, no raw sample array at baseline tier)
- candidate_raw_samples_ms: `[3.400635]` (median only)
- reference_median_ms: `3.396440`
- candidate_median_ms: `3.400635`
- improvement_pct: `-0.1235` (candidate is marginally slower; expected at baseline — identical semantics)

```text
improvement_pct = (3.396440 - 3.400635) / 3.396440 * 100 = -0.1235%
```

Phase 0 baseline establishes the accepted reference wall time. The baseline
adapter is byte-equivalent in semantics to base.py (same Sinkhorn loop), so the
~0.12% wall delta is measurement noise, not an optimization signal.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `not-applicable: Phase 0`
- Hypothesis verdict: `inconclusive`

No round decision exists in Phase 0; there is no mechanism observable to mirror.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary` (Level 1)
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`,
  `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable` (device time available via CANN)

Reference and candidate scopes are collected and summarized independently. All
totals below are normalized by `iterations=50`.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference | 14059.240 | 281.185 | 6801 | 136.02 | 3.406305 | 0.0825 |
| candidate | 14030.060 | 280.601 | 6800 | 136.00 | 3.416190 | 0.0821 |

```text
device_ratio = device_us_per_call / (wall_ms * 1000)
reference: 281.185 / 3406.305 = 0.0825
candidate: 280.601 / 3416.190 = 0.0821
```

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnReduceSum_ReduceSumOpAiCore_ReduceSum | 2000 | 40.0 | 8452.500 | 169.050 |
| aclnnDiv_RealDivAiCore_RealDiv | 2000 | 40.0 | 2378.900 | 47.578 |
| aclnnAdds_AddAiCore_Add | 2050 | 41.0 | 1983.120 | 39.662 |
| aclnnMul_SliceAiCore_Slice | 150 | 3.0 | 412.660 | 8.253 |
| aclnnAmax_ReduceMaxAiCore_ReduceMax | 50 | 1.0 | 239.180 | 4.784 |
| aclnnAdd_AddAiCore_Add | 150 | 3.0 | 207.880 | 4.158 |
| aclnnMul_MulAiCore_Mul | 150 | 3.0 | 144.220 | 2.884 |
| aclnnSigmoid_SigmoidAiCore_Sigmoid | 100 | 2.0 | 89.800 | 1.796 |
| aclnnSub_SubAiCore_Sub | 50 | 1.0 | 60.960 | 1.219 |
| aclnnMuls_MulAiCore_Mul | 50 | 1.0 | 50.160 | 1.003 |
| aclnnExp_ExpAiCore_Exp | 50 | 1.0 | 39.840 | 0.797 |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnReduceSum_ReduceSumOpAiCore_ReduceSum | 2000 | 40.0 | 8431.020 | 168.620 |
| aclnnDiv_RealDivAiCore_RealDiv | 2000 | 40.0 | 2376.260 | 47.525 |
| aclnnAdds_AddAiCore_Add | 2050 | 41.0 | 1980.760 | 39.615 |
| aclnnMul_SliceAiCore_Slice | 150 | 3.0 | 411.060 | 8.221 |
| aclnnAmax_ReduceMaxAiCore_ReduceMax | 50 | 1.0 | 236.960 | 4.739 |
| aclnnAdd_AddAiCore_Add | 150 | 3.0 | 207.980 | 4.160 |
| aclnnMul_MulAiCore_Mul | 150 | 3.0 | 145.120 | 2.902 |
| aclnnSigmoid_SigmoidAiCore_Sigmoid | 100 | 2.0 | 89.660 | 1.793 |
| aclnnSub_SubAiCore_Sub | 50 | 1.0 | 61.380 | 1.228 |
| aclnnMuls_MulAiCore_Mul | 50 | 1.0 | 50.440 | 1.009 |
| aclnnExp_ExpAiCore_Exp | 50 | 1.0 | 39.420 | 0.788 |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial Phase 0 verification | not-applicable | `c3cc90deb93c69f092e876ddc0057ddf6c2d7ac28020f1b7fd1ed260bca72fee` | baseline established |

## evidence_for_next_round

- The operator is strongly host-bound: device_ratio ≈ 8.2% (device_us_per_call
  ≈ 281 us vs wall ≈ 3406 us). The remaining ~91% of wall time is host-side
  (launcher/alloc/sync/context), not device kernel time.
- The Sinkhorn loop (20 iterations) drives a very high kernel count:
  kernel_count_per_call = 136 (6801 kernels / 50 calls). The loop bodies are
  `ReduceSum` + `RealDiv` per iteration, accounting for 80 kernels/call (40
  ReduceSum + 40 Div) plus the fixed head (sigmoid, amax, exp, adds).
- Dominant device kernel is `aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`
  (169 us/call, 60% of device time), one per Sinkhorn column-sum step. Next is
  `aclnnDiv_RealDivAiCore_RealDiv` (47.6 us/call).
- The small per-kernel shapes (4x4 comb matrix, 2x8x4 gates) mean each kernel
  is tiny (ReduceSum ~4.2 us each on average), so the wall time is dominated by
  per-launch host overhead rather than kernel compute.
- Baseline reference wall median: 3.396440 ms. Baseline adapter wall median:
  3.400635 ms (semantically identical, noise-level delta).

## Stop Recommendation

- recommendation: `continue`
- evidence: Phase 0 baseline established; no optimization attempted yet. Host-bound
  profile (8.2% device ratio) with a 20-iteration Python loop is the clear first
  bottleneck target.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/mhc-head-compute-mix-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/ascend/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/ascend/baseline_adapter.py --profile --profile-reference-file kernels/track1-triton/mhc_head_compute_mix/ascend/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mhc_head_compute_mix/ascend/log/round_000_forward_50iter.pt.trace.json
```

```bash
python3 /workspace/kernelswift/skills/kernel-opt-loop/scripts/summarize_cann_trace.py <scope_ascend_pt_dir> --iterations 50 --wall-ms <scope_wall_ms>
```
