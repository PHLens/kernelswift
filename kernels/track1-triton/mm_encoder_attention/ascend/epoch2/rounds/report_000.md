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
- Measurement fingerprint: `1b1822d7b74a8cd41411a27fcbc18a89cb50b1cfefb9fdac2585cdd520e9a79a`
- verification_tier: baseline
- screening_pairs: `not-run`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | base.py vs baseline_adapter.py outputs match within harness default (atol=1e-2, rtol=1e-2) | `PASS accuracy; v0=0.343000 ms, v1=0.335125 ms, speedup=1.023x` (warmup 5, repeat 10) | pass | `python3 auto_bench.py --v0_file .../base.py --v1_file .../epoch2/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback` |

This is a faithful reproduction check: baseline_adapter.py renames `Model` -> `ModelNew` and is otherwise byte-for-byte semantically identical to base.py. `base.py` bytes are unchanged after adapter generation.

## Screening Evidence

Screening is not applicable to Phase 0 baseline establishment; no candidate optimization exists yet.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `reference (base.py) then candidate (baseline_adapter.py)` — Phase 0 baseline; single comparison
- reference_median_ms: `0.349625`
- candidate_median_ms: `0.347800`
- improvement_pct: `0.5220`

```text
improvement_pct = (0.349625 - 0.347800) / 0.349625 * 100 = 0.5220
```

Baseline is within noise (0.52% difference), as expected for a faithful reproduction.

### Baseline drift versus epoch 1

The epoch-1 baseline recorded `reference_median_ms = 0.320635`. The current
measurement is `0.349625`, a drift of `+9.04%` under an identical measurement
fingerprint. Epoch-2 comparisons must use the current baseline; epoch-1 wall
numbers are not directly usable as the adoption reference even though the
fingerprint matches.

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

Reference and candidate scopes are collected and summarized independently. Both
scopes point to the same baseline_adapter.py logic (Phase 0 establishes the
accepted reference via a self-comparison). All totals normalized by
`iterations=50`.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| reference_baseline_adapter | 5808.48 | 116.1696 | 349 | 6.98 | 0.402620 | 0.2885 |
| candidate_baseline_adapter | 5206.32 | 104.1264 | 348 | 6.96 | 0.388015 | 0.2684 |

```text
device_ratio = device_us_per_call / (wall_ms * 1000)
```

Note: wall_ms used for device_ratio are the values observed during the profiler
run itself (reference=0.402620 ms, candidate=0.388015 ms). The profiler run adds
host overhead, so these wall values exceed the un-profiled benchmark medians
(0.349625 / 0.347800 ms). Using the benchmark medians instead gives device_ratio
of roughly 0.33, so device time is about one third of wall time either way.
**The dominant cost is host-side launch and synchronization, not device compute.**

### Reference Top Kernels (reference_baseline_adapter)

| Kernel | Count total | Count/call | Total us | Us/call | Share of device |
|---|---:|---:|---:|---:|---:|
| aclnnFlashAttentionScore_TransposeAiCore_Transpose | 150 | 3.0 | 2404.42 | 48.0884 | 41.4% |
| EVENT_WAIT_SQE | 49 | 0.98 | 1488.76 | 29.7752 | 25.6% |
| aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore | 50 | 1.0 | 1151.16 | 23.0232 | 19.8% |
| aclnnInplaceCopy_TransposeAiCore_Transpose | 50 | 1.0 | 763.14 | 15.2628 | 13.1% |
| EVENT_RECORD_SQE | 50 | 1.0 | 1.00 | 0.0200 | 0.0% |

### Candidate Top Kernels (candidate_baseline_adapter)

| Kernel | Count total | Count/call | Total us | Us/call | Share of device |
|---|---:|---:|---:|---:|---:|
| aclnnFlashAttentionScore_TransposeAiCore_Transpose | 150 | 3.0 | 2409.50 | 48.1900 | 46.3% |
| aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore | 50 | 1.0 | 1152.00 | 23.0400 | 22.1% |
| EVENT_WAIT_SQE | 50 | 1.0 | 888.50 | 17.7700 | 17.1% |
| aclnnInplaceCopy_TransposeAiCore_Transpose | 50 | 1.0 | 755.50 | 15.1100 | 14.5% |
| EVENT_RECORD_SQE | 48 | 0.96 | 1.00 | 0.0200 | 0.0% |

### Attribution

Two facts bound every epoch-2 hypothesis:

1. **Device time is ~29% of wall time.** Roughly 250 us/call of the ~349 us wall
   is host-side launch, dispatch, and synchronization. A candidate that only
   shrinks device compute cannot clear the 5% adoption threshold on wall time
   unless it also removes launches.
2. **Within device time, layout conversion costs more than attention.** The three
   `aclnnFlashAttentionScore_TransposeAiCore_Transpose` kernels cost 48.09 us/call
   and the `aclnnInplaceCopy_TransposeAiCore_Transpose` costs 15.26 us/call,
   together 63.35 us/call (54.5% of device), while the actual
   FlashAttentionScore kernel costs only 23.02 us/call (19.8%). The transposes are
   the reshape/transpose metadata ops around SDPA that the native backend
   materializes as contiguous copies.

This matches the cross-backend `device-win-wall-loss` pattern: an epoch-2
candidate must convert its device win into a wall win by collapsing launches and
eliminating materialized transposes, not by making the SDPA kernel itself faster.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification (Phase 0 baseline) | `not-applicable` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | correctness pass, benchmark + profiler recorded |

## evidence_for_next_round

- Baseline correctness confirmed: baseline_adapter.py is a faithful reproduction of base.py.
- Baseline wall: reference median `0.349625` ms, candidate median `0.347800` ms (0.52% apart, within noise).
- **Baseline drifted +9.04% versus epoch 1** (`0.320635` -> `0.349625`) under an identical measurement fingerprint; use the current numbers as the adoption reference.
- Device time is `104-116 us/call` at `device_ratio` 0.27-0.29 (~0.33 against benchmark medians), so about two thirds of wall time is host-side.
- Launch profile: `6.96-6.98` kernels/call — 3 transposes, 1 FlashAttentionScore, 1 inplace-copy transpose, plus `EVENT_WAIT_SQE`/`EVENT_RECORD_SQE` synchronization events.
- Device composition: layout conversion (`48.09 + 15.26 = 63.35 us/call`) exceeds the attention kernel itself (`23.02 us/call`).
- Bottleneck: host launch/synchronization overhead dominates wall time; the largest removable device cost is the materialized transpose/copy chain, not the SDPA math.
- Capability note (from the frozen profile snapshot): fp16 `tl.dot` is `constrained` and numerically correct on every probed tile including non-multiple-of-16 shapes, so `seq_len=83` needs no padding to 128; `num_warps` 1/2/4/8 and `num_stages` 1/2/3/4 are profile-legal.

## Stop Recommendation

- recommendation: `continue`
- evidence: Phase 0 baseline is established; no target set, no no-improvement limit reached, round budget not exhausted.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/baseline_adapter.py --profile --profile-reference-file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/round_000_forward_50iter.pt.trace.json
```

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py "kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/profiling_data/reference_baseline_adapter/profiling_data" --iterations 50 --scope reference_baseline_adapter --wall-ms 0.402620

python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py "kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/profiling_data/candidate_baseline_adapter/profiling_data" --iterations 50 --scope candidate_baseline_adapter --wall-ms 0.388015
```
