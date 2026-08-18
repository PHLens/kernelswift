# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py for Phase 0`
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e`
- Accepted reference SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `pending` (owned by Orchestrator)
- verification_tier: baseline
- screening_pairs: `not-run`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | base.py vs baseline_adapter.py outputs match within harness default (atol=1e-2, rtol=1e-2) | `PASS accuracy; v0=0.320635 ms, v1=0.319655 ms, speedup=1.003x` | pass | `python3 auto_bench.py --v0_file .../base.py --v1_file .../baseline_adapter.py --warmup 50 --repeat 100` |

This is a faithful reproduction check: baseline_adapter.py renames `Model` -> `ModelNew` and is otherwise byte-for-byte semantically identical to base.py. Correctness passes (faithful reproduction confirmed).

## Screening Evidence

Screening is not applicable to Phase 0 baseline establishment; no candidate optimization exists yet.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `reference (base.py) then candidate (baseline_adapter.py)` — Phase 0 baseline; single comparison
- reference_median_ms: `0.320635`
- candidate_median_ms: `0.319655`
- improvement_pct: `0.3056`

```text
improvement_pct = (0.320635 - 0.319655) / 0.320635 * 100 = 0.3056
```

Baseline is within noise (0.31% difference), as expected for a faithful reproduction.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `wall_time`
- Hypothesis verdict: `inconclusive` (no decision/hypothesis exists in Phase 0)

## Profiler Evidence

- profiler_applicability: `required` (baseline device evidence)
- profiler_level: `summary`
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable`

Reference and candidate scopes are collected and summarized independently. Both scopes point to the same baseline_adapter.py logic (Phase 0 establishes the accepted reference via a self-comparison). All totals normalized by `iterations=50`.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| reference_baseline_adapter | 6007.62 | 120.1524 | 343 | 6.86 | 0.346075 | 0.3472 |
| candidate_baseline_adapter | 5387.72 | 107.7544 | 336 | 6.72 | 0.332835 | 0.3237 |

```text
device_ratio = device_us_per_call / (wall_ms * 1000)
```

Note: wall_ms used for device_ratio are the values observed during the profiler run itself (reference=0.346075 ms, candidate=0.332835 ms), which are the harness-reported timing for the same invocation. Device time (~108-120 us/call) is a small fraction of the ~0.33 ms wall time; the dominant cost is host-side launch/synchronization overhead, not device kernel time.

### Accepted Reference Top Kernels (reference_baseline_adapter)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnFlashAttentionScore_TransposeAiCore_Transpose | 150 | 3.0 | 2403.40 | 48.0680 |
| EVENT_WAIT_SQE | 49 | 0.98 | 1725.12 | 34.5024 |
| aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore | 50 | 1.0 | 1171.24 | 23.4248 |
| aclnnInplaceCopy_TransposeAiCore_Transpose | 50 | 1.0 | 706.98 | 14.1396 |
| EVENT_RECORD_SQE | 44 | 0.88 | 0.88 | 0.0176 |

### Candidate Top Kernels (candidate_baseline_adapter)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnFlashAttentionScore_TransposeAiCore_Transpose | 150 | 3.0 | 2395.56 | 47.9112 |
| aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore | 50 | 1.0 | 1179.38 | 23.5876 |
| EVENT_WAIT_SQE | 48 | 0.96 | 1114.16 | 22.2832 |
| aclnnInplaceCopy_TransposeAiCore_Transpose | 50 | 1.0 | 697.86 | 13.9572 |
| EVENT_RECORD_SQE | 38 | 0.76 | 0.76 | 0.0152 |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification (Phase 0 baseline) | `not-applicable` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | correctness pass, benchmark + profiler recorded |

## evidence_for_next_round

- Baseline correctness confirmed: baseline_adapter.py is a faithful reproduction of base.py (outputs match within harness tolerance).
- Baseline wall time: reference median 0.320635 ms, candidate median 0.319655 ms (within noise, 0.31% apart).
- Baseline device time: ~108-120 us/call of AI Core kernel time; device_ratio ~0.32-0.35, meaning the wall time is dominated by host-side launch/synchronization overhead, not device compute.
- Dominant device kernels: `aclnnFlashAttentionScore_TransposeAiCore_Transpose` (3/call, ~48 us/call), the FlashAttentionScore SDPA kernel (~23.5 us/call), and `aclnnInplaceCopy_TransposeAiCore_Transpose` (~14 us/call). The three transpose kernels per call correspond to the reshape/transpose around `scaled_dot_product_attention`.
- Current bottleneck: host-side overhead dominates (device_ratio < 0.35); among device kernels, the SDPA flash-attention kernel and its transpose-wrapping are the top contributors. The `EVENT_WAIT_SQE`/`EVENT_RECORD_SQE` events (synchronization) also contribute meaningful time in the reference capture (~34.5 us/call), indicating launch/sync overhead between the per-call transpose ops.

## Stop Recommendation

- recommendation: `continue`
- evidence: Phase 0 baseline is established; no target set, no no-improvement limit reached, round budget not exhausted.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/mm-encoder-attn-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/ascend/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/mm-encoder-attn-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/ascend/baseline_adapter.py --profile --profile-reference-file kernels/track1-triton/mm_encoder_attention/ascend/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mm_encoder_attention/ascend/log/round_000_forward_50iter.pt.trace.json
```

```bash
/usr/local/python3.11.15/bin/python3 /workspace/kernelswift/skills/kernel-opt-loop/scripts/summarize_cann_trace.py "kernels/track1-triton/mm_encoder_attention/ascend/log/profiling_data/reference_baseline_adapter/profiling_data" --iterations 50 --scope reference_baseline_adapter --wall-ms 0.346075

/usr/local/python3.11.15/bin/python3 /workspace/kernelswift/skills/kernel-opt-loop/scripts/summarize_cann_trace.py "kernels/track1-triton/mm_encoder_attention/ascend/log/profiling_data/candidate_baseline_adapter/profiling_data" --iterations 50 --scope candidate_baseline_adapter --wall-ms 0.332835
```
