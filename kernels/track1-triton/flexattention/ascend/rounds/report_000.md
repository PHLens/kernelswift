# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `../base.py` (base.py for Phase 0)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `31c4e9acea7d94ddd97740dbd3d33e6b505cbc3a118ed891b28f9e1ac5c0696f`
- Accepted reference SHA256: `12f8a77b8f52b50d513800907b6b21ff9c98709647b306793241f6f8da3cb105`
- Base SHA256: `12f8a77b8f52b50d513800907b6b21ff9c98709647b306793241f6f8da3cb105`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `c1359d456700562802630e66368ce04856d871a993562ce1437e037af82581b8`
- verification_tier: baseline
- screening_pairs: `not-run: Phase 0 baseline`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | `PASS accuracy; v0=0.404295 ms, v1=0.407810 ms, speedup=0.991x` | pass | `python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback` |
| output contract | out `[83,512]` fp16, allclose(atol=1e-2, rtol=1e-2) | shapes/dtypes/values match under harness comparison | pass | correctness gate exit status 0, `Summary: 1 passed, 0 failed` |

Conformance, correctness, and every declared guardrail must pass before adoption.

## Screening Evidence

Screening follows correctness and uses exactly two ordered short interleaved
accepted-reference/candidate pairs. A correct candidate is `screened-out` only
when both pairs are at least 10% slower than the accepted reference. Any other
correct candidate proceeds to authoritative timing.

| Pair | Reference short wall ms | Candidate short wall ms | Candidate slower pct | Evidence |
|---:|---:|---:|---:|---|
| 1 | `<not-run>` | `<not-run>` | `<not-run>` | `not-run: Phase 0 baseline (no comparison target)` |
| 2 | `<not-run>` | `<not-run>` | `<not-run>` | `not-run: Phase 0 baseline (no comparison target)` |

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `baseline reference (base.py) / baseline adapter (ModelNew)` — Phase 0 baseline, not an adoption comparison
- reference_raw_samples_ms: `[median only reported by harness]`
- candidate_raw_samples_ms: `[median only reported by harness]`
- reference_median_ms: `0.409435`
- candidate_median_ms: `0.410860`
- improvement_pct: `-0.3481`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (0.409435 - 0.410860) / 0.409435 * 100 = -0.3481
```

The unrounded improvement controls the 5% adoption threshold. Profiler time does
not replace this benchmark result. Only authoritative timing can yield `accepted` or `no-improvement`.

Note: The harness `time_forward` reports only the `statistics.median` of the
repeat samples (per `project.md` primary_metric `unrounded median wall_time_ms`).
For the Phase 0 baseline, the baseline adapter (`ModelNew`) is byte-equivalent
semantics to the reference (`Model`); the two medians are expected to coincide
within noise. Baseline wall time adopted for the campaign is the reference
(base.py) median `0.409435 ms`.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `not-applicable: Phase 0`
- Hypothesis verdict: `inconclusive`

No round decision exists in Phase 0; there are no mechanism observables to
mirror.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available` (CANN msprof `ai_core_op_summary.db`)
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`,
  `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable` (device time available, not GCU runtime launch fallback)

Reference and candidate scopes are collected and summarized independently. All
totals below are normalized by `iterations` before they are compared.
Profiler evidence is required for baseline and accepted candidates, and is not
run for `screened-out` candidates.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (`baseline_base`) | 7400.94 | 148.0188 | 433 | 8.66 | 0.410860 | 0.360266 |
| candidate (`candidate_baseline_adapter`) | 6869.72 | 137.3944 | 442 | 8.84 | 0.410860 | 0.334407 |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)
```

### Accepted Reference Top Kernels (`baseline_base`)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnFlashAttentionScore_TransposeAiCore_Transpose | 150 | 3.0 | 2354.34 | 47.0868 |
| EVENT_WAIT_SQE | 50 | 1.0 | 1573.94 | 31.4788 |
| aclnnTriu_Triu_Triu | 50 | 1.0 | 1308.14 | 26.1628 |
| aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore | 50 | 1.0 | 1243.48 | 24.8696 |
| aclnnInplaceCopy_TransposeAiCore_Transpose | 50 | 1.0 | 676.58 | 13.5316 |
| aclnnInplaceOne_OnesLikeAiCore_OnesLike | 50 | 1.0 | 243.80 | 4.8760 |
| EVENT_RECORD_SQE | 33 | 0.66 | 0.66 | 0.0132 |

### Candidate Top Kernels (`candidate_baseline_adapter`)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnFlashAttentionScore_TransposeAiCore_Transpose | 150 | 3.0 | 2357.72 | 47.1544 |
| aclnnTriu_Triu_Triu | 50 | 1.0 | 1313.82 | 26.2764 |
| aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore | 50 | 1.0 | 1242.02 | 24.8404 |
| EVENT_WAIT_SQE | 50 | 1.0 | 1034.52 | 20.6904 |
| aclnnInplaceCopy_TransposeAiCore_Transpose | 50 | 1.0 | 677.06 | 13.5412 |
| aclnnInplaceOne_OnesLikeAiCore_OnesLike | 50 | 1.0 | 243.74 | 4.8748 |
| EVENT_RECORD_SQE | 42 | 0.84 | 0.84 | 0.0168 |

Note: `EVENT_WAIT_SQE` / `EVENT_RECORD_SQE` are synchronisation/event tasks, not
compute kernels; their apparent duration reflects wait time rather than AI Core
work. The dominant compute is the FlashAttentionScore op (transpose + fused
attention) plus the causal mask `aclnnTriu` and the final transpose copies.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial baseline verification | not-applicable | `31c4e9acea7d94ddd97740dbd3d33e6b505cbc3a118ed891b28f9e1ac5c0696f` | baseline established (correctness pass) |

At most one Verifier-to-Coder repair is allowed in the same round.

## evidence_for_next_round

- Baseline wall time (reference `base.py`, warmup 50 / repeat 100) median: `0.409435 ms`.
- Baseline device time (reference scope `baseline_base`): `148.0188 us/call`, `8.66` kernels/call.
- Dominant compute kernels per forward call: FlashAttentionScore transpose + fused attention (~47 us/call across 3 transpose invocations), causal `aclnnTriu` (~26.2 us/call), and FlashAttentionScore core (~24.9 us/call).
- `EVENT_WAIT_SQE` accounts for ~31.5 us/call (reference) / ~20.7 us/call (candidate) of apparent device wait time — a potential host/launch synchronisation cost observable for future rounds.
- The causal mask is materialized via a standalone `aclnnTriu_Triu_Triu` op and the pre/post transposes are separate `TransposeAiCore` kernels, indicating decomposition opportunities (fusing the causal mask and transposes into the attention kernel).

Record evidence only; do not select the next optimization.

## Stop Recommendation

- recommendation: `continue`
- evidence: baseline established at `0.409435 ms` wall / `148.0188 us` device per call; campaign has not yet attempted any optimization round.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/flexattention/ascend/log/flexattention_baseline_forward_50iter.pt.trace.json
```

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/flexattention/ascend/log/profiling_data/baseline_base/profiling_data --iterations 50 --scope baseline_base --wall-ms 0.410860
```

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/flexattention/ascend/log/profiling_data/candidate_baseline_adapter/profiling_data --iterations 50 --scope candidate_baseline_adapter --wall-ms 0.410860
```
