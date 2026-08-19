# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `../base.py` (Phase 0)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `98cf1e1e5f493ae4ae08c9391f00db83b3cd783a1b194d9c0d63851091133a5d`
- Accepted reference SHA256: `28d4d213638e5adc7f6bc52928c8f7596cd8446e3e0201beab364b5c3c7988fc`
- Base SHA256: `28d4d213638e5adc7f6bc52928c8f7596cd8446e3e0201beab364b5c3c7988fc`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `pending` (Orchestrator-owned)
- verification_tier: baseline
- screening_pairs: `not-run`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | v1 output matches v0 (atol=1e-2, rtol=1e-2) | `PASS accuracy` | pass | `auto_bench.py --warmup 50 --repeat 100` -> `PASS accuracy; v0=0.456720 ms, v1=0.433775 ms, speedup=1.053x` |
| output shape | tuple (grad_input_mix[2,1024,4], grad_mhc_scale[1], grad_mhc_base[4]) fp32 | matched; harness `compare_values` passed all three | pass | harness `compare_values` |
| dtype | fp32 throughout | matched | pass | harness allclose on fp32 |

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `harness single v0 then v1 (same process, same regime)`
- reference_raw_samples_ms: `[harness median only — see note]`
- candidate_raw_samples_ms: `[harness median only — see note]`
- reference_median_ms: `0.456720`
- candidate_median_ms: `0.433775`
- improvement_pct: `5.02`

Note: the harness reports unrounded medians, not raw sample arrays. `improvement_pct = (0.456720 - 0.433775) / 0.456720 * 100 = 5.02%`. This is Phase 0 baseline establishment: base.py and baseline_adapter.py are the same computation, so the ~5% gap is run-to-run variance, not a real improvement. No adoption decision is made in Phase 0.

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
```

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `not-applicable: Phase 0`
- Hypothesis verdict: `inconclusive` (no decision exists in Phase 0)

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable` (device time available)

Reference and candidate scopes are collected in separate CANN captures (distinct
`ASCEND_WORK_PATH`), so each `ai_core_op_summary.db` contains exactly one scope.
Totals normalized by `iterations = 50`.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (base.py) | 2058.10 | 41.162 | 500 | 10.0 | 0.456720 | 0.0901 |
| candidate (baseline_adapter.py) | 2059.14 | 41.183 | 500 | 10.0 | 0.433775 | 0.0949 |

```text
device_ratio = device_us_per_call / (wall_ms * 1000)
reference:  41.162 / (0.456720 * 1000) = 0.0901
candidate:  41.183 / (0.433775 * 1000) = 0.0949
```

### Accepted Reference Top Kernels (base.py)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnReduceSum_ReduceSumOpAiCore_ReduceSum | 100 | 2.0 | 1124.20 | 22.484 |
| aclnnMul_MulAiCore_Mul | 250 | 5.0 | 434.94 | 8.699 |
| aclnnAdd_AddAiCore_Add | 50 | 1.0 | 283.20 | 5.664 |
| aclnnRsubs_SubAiCore_Sub | 50 | 1.0 | 109.36 | 2.187 |
| aclnnSigmoid_SigmoidAiCore_Sigmoid | 50 | 1.0 | 106.40 | 2.128 |

### Candidate Top Kernels (baseline_adapter.py)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnReduceSum_ReduceSumOpAiCore_ReduceSum | 100 | 2.0 | 1125.44 | 22.509 |
| aclnnMul_MulAiCore_Mul | 250 | 5.0 | 434.28 | 8.686 |
| aclnnAdd_AddAiCore_Add | 50 | 1.0 | 283.68 | 5.674 |
| aclnnRsubs_SubAiCore_Sub | 50 | 1.0 | 108.76 | 2.175 |
| aclnnSigmoid_SigmoidAiCore_Sigmoid | 50 | 1.0 | 106.98 | 2.140 |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial Phase 0 verification | not-applicable | `98cf1e1e5f493ae4ae08c9391f00db83b3cd783a1b194d9c0d63851091133a5d` | baseline established |

## evidence_for_next_round

- Wall time ~434 us/call vs device time ~41 us/call -> `device_ratio ≈ 0.095` (9.5%), i.e. **host-bound**. ~91% of wall time is host-side (launcher, allocation, seed setup, harness device sync), not device kernel time.
- The forward decomposes into **10 kernels per call** (many small kernels): 2x ReduceSum, 5x Mul, 1x Add, 1x Rsubs (rsub for `1 - sigmoid`), 1x Sigmoid.
- The two reductions (`sum((0,1))` -> grad_mhc_base and `sum((0,1,2))` -> grad_mhc_scale) map to `aclnnReduceSum` (2/call), which together dominate device time at ~22.5 us/call (~55% of all device time).
- The elementwise sigmoid chain is unfused: 1 Sigmoid + 1 Rsubs + 5 Mul + 1 Add as separate kernels.
- Benchmark wall time is dominated by host-side overhead, not the reduction kernels — a triton fusion candidate would need to reduce kernel launch count (currently 10/call) to move wall time; device-side compute is already small (41 us).

## Stop Recommendation

- recommendation: `continue`
- evidence: `Phase 0 baseline established. No stop condition reached (target_mode is null; valid_no_improvement_limit not yet triggered).`

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/mhc-head-backward-ascend
/usr/local/python3.11.15/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix_backward/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix_backward/ascend/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/mhc-head-backward-ascend
/usr/local/python3.11.15/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix_backward/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix_backward/ascend/baseline_adapter.py --profile --profile-reference-file kernels/track1-triton/mhc_head_compute_mix_backward/ascend/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mhc_head_compute_mix_backward/ascend/log/round_000_forward_50iter.pt.trace.json
```

```bash
cd /workspace/kernelswift/.worktrees/mhc-head-backward-ascend/kernels/track1-triton/mhc_head_compute_mix_backward/ascend
/usr/local/python3.11.15/bin/python3 /workspace/kernelswift/skills/kernel-opt-loop/scripts/summarize_cann_trace.py log/profiling_data/reference_baseline_adapter/profiling_data --iterations 50 --wall-ms 0.456720
/usr/local/python3.11.15/bin/python3 /workspace/kernelswift/skills/kernel-opt-loop/scripts/summarize_cann_trace.py log/profiling_data/candidate_baseline_adapter/profiling_data --iterations 50 --wall-ms 0.433775
```
