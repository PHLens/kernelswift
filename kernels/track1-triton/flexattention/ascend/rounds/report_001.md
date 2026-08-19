# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_flexattention_001.py`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `91cae0bcb4eb0792e59be2c359b21dde2cc038a2d11e25f01e36bb20784bf379`
- Candidate SHA256: `53e87eff27457f6268040c64979f99dcf30a809effc562caec3db951b141d4a3`
- Accepted reference SHA256: `31c4e9acea7d94ddd97740dbd3d33e6b505cbc3a118ed891b28f9e1ac5c0696f`
- Base SHA256: `12f8a77b8f52b50d513800907b6b21ff9c98709647b306793241f6f8da3cb105`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `c1359d456700562802630e66368ce04856d871a993562ce1437e037af82581b8`
- verification_tier: authoritative
- screening_pairs: `not-run: candidate proceeded directly to authoritative timing (correctness pass, expected improvement)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | `PASS accuracy; v0=0.403910 ms, v1=0.329930 ms, speedup=1.224x` | pass | `python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/triton_flexattention_001.py --warmup 5 --repeat 10 --full-traceback` |
| output dtype and shape unchanged | out `[83,512]` fp16 | harness `compare_values` shape/dtype check passed; output `[83,512]` fp16 | pass | correctness gate exit 0, `Summary: 1 passed, 0 failed` |
| causal semantics preserved | lower-triangular mask, scale=1/sqrt(head_size) | allclose(atol=1e-2, rtol=1e-2) passed against reference | pass | correctness gate exit 0 |

Conformance, correctness, and every declared guardrail must pass before adoption.

## Screening Evidence

Screening follows correctness and uses exactly two ordered short interleaved
accepted-reference/candidate pairs. A correct candidate is `screened-out` only
when both pairs are at least 10% slower than the accepted reference. Any other
correct candidate proceeds to authoritative timing.

| Pair | Reference short wall ms | Candidate short wall ms | Candidate slower pct | Evidence |
|---:|---:|---:|---:|---|
| 1 | `<not-run>` | `<not-run>` | `<not-run>` | `not-run: proceeded to authoritative timing` |
| 2 | `<not-run>` | `<not-run>` | `<not-run>` | `not-run: proceeded to authoritative timing` |

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate`
- reference_raw_samples_ms: `[0.386255, 0.405655, 0.411600]`
- candidate_raw_samples_ms: `[0.330810, 0.326940, 0.337995]`
- reference_median_ms: `0.405655`
- candidate_median_ms: `0.330810`
- improvement_pct: `18.450407365865082`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (0.405655 - 0.330810) / 0.405655 * 100 = 18.4504
```

The unrounded improvement controls the 5% adoption threshold. Profiler time does
not replace this benchmark result. Only authoritative timing can yield `accepted` or `no-improvement`.

Note: The harness `time_forward` reports only `statistics.median` per run. Three
independent authoritative runs were executed in one Verifier turn, each
measuring the accepted reference (`base.py`, semantically identical to
`baseline_adapter.py` with `Model`→`ModelNew`) and the candidate in the same
process/environment. The three per-run medians above are the raw samples; the
overall unrounded medians are the median of those three samples. Improvement
18.45% exceeds the 5% adoption threshold.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| kernel_count_per_call | decrease toward 1 | reference `8.72` → candidate `1.00` | pass | `summarize_cann_trace.py` per-scope summaries |
| device_us_per_call | decrease | reference `145.4256` → candidate `54.04` us/call | pass | `summarize_cann_trace.py` per-scope summaries |
| aclnnTranspose_kernel_presence | absent | absent from candidate scope (only `_causal_attn_kernel` present) | pass | candidate scope kernel list |
| aclnnTriu_kernel_presence | absent | absent from candidate scope | pass | candidate scope kernel list |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `fuse causal SDPA into a single Triton kernel that computes QK^T, applies the causal mask, softmax, and AV in one launch, eliminating the three TransposeAiCore copies, the standalone aclnnTriu mask, the OnesLike, and the inter-kernel sync waits`
- expected_causal_chain: `transpose and Triu and OnesLike and copy kernels disappear from the candidate scope → kernel_count_per_call decreases from 8.66 toward 1 → device_us_per_call decreases → wall_time_ms decreases`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

Every declared observable passed: the transpose/Triu/OnesLike/copy kernels and
`EVENT_WAIT_SQE` all vanished, `kernel_count_per_call` dropped 8.72 → 1.0,
`device_us_per_call` dropped 145.43 → 54.04 us, and wall time improved 18.45%.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
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
| accepted_reference (`baseline_base`) | 7271.28 | 145.4256 | 436 | 8.72 | 0.405655 | 0.358496 |
| candidate (`candidate_triton_flexattention_001`) | 2702.00 | 54.04 | 50 | 1.00 | 0.330810 | 0.163357 |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)
```

### Accepted Reference Top Kernels (`baseline_base`)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnFlashAttentionScore_TransposeAiCore_Transpose | 150 | 3.0 | 2333.72 | 46.6744 |
| EVENT_WAIT_SQE | 49 | 0.98 | 1456.80 | 29.1360 |
| aclnnTriu_Triu_Triu | 50 | 1.0 | 1293.24 | 25.8648 |
| aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore | 50 | 1.0 | 1247.60 | 24.9520 |
| aclnnInplaceCopy_TransposeAiCore_Transpose | 50 | 1.0 | 687.98 | 13.7596 |
| aclnnInplaceOne_OnesLikeAiCore_OnesLike | 50 | 1.0 | 251.20 | 5.0240 |
| EVENT_RECORD_SQE | 37 | 0.74 | 0.74 | 0.0148 |

### Candidate Top Kernels (`candidate_triton_flexattention_001`)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _causal_attn_kernel | 50 | 1.0 | 2702.00 | 54.04 |

The candidate scope contains exactly one Triton kernel (`_causal_attn_kernel`)
per forward call. All decomposed library kernels (`aclnnTranspose`/`aclnnTriu`/
`aclnnInplaceCopy`/`aclnnInplaceOne`) and the `EVENT_WAIT_SQE`/`EVENT_RECORD_SQE`
sync tasks are gone, confirming the single-launch fusion.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable | `53e87eff27457f6268040c64979f99dcf30a809effc562caec3db951b141d4a3` | accepted (correctness pass, improvement 18.45%) |

At most one Verifier-to-Coder repair is allowed in the same round.

## evidence_for_next_round

- Round 1 fused kernel is accepted: single `_causal_attn_kernel` launch, `1.0` kernels/call, `54.04 us/call` device time, `0.330810 ms` wall time.
- The candidate device time (54.04 us) is now dominated by the fused Triton kernel itself (no more decomposition/sync overhead). Wall time (0.330810 ms) still far exceeds device time (0.054 ms), so host/launch overhead (~0.28 ms) is now the dominant wall-time component (device_ratio 0.163).
- Kernel uses `tl.sum` rank-1 reductions with `num_warps=1` and a 1D `T*H` grid (664 programs), `BLOCK_K=128`, `BLOCK_D=64`; each program loads a full `[128,64]` K and V block. The single fused kernel is ~54 us, which still leaves headroom versus the theoretical single `aclnnFlashAttentionScore` core (~25 us) from the reference.
- The `EVENT_WAIT_SQE`/`EVENT_RECORD_SQE` sync tasks observed in the reference are absent in the candidate scope, confirming inter-kernel sync elimination.

Record evidence only; do not select the next optimization.

## Stop Recommendation

- recommendation: `continue`
- evidence: improvement 18.45% (> 5% threshold), hypothesis confirmed; `performance_miss_streak` will reset on acceptance, campaign has room under `max_rounds=20`.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/triton_flexattention_001.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/triton_flexattention_001.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/triton_flexattention_001.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/flexattention/ascend/log/round_001_forward_50iter.pt.trace.json
```

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/flexattention/ascend/log/profiling_data/baseline_base/profiling_data/c8843a4fa93a_118069_20260818094923005_ascend_pt --iterations 50 --scope baseline_base --wall-ms 0.405655
```

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/flexattention/ascend/log/profiling_data/candidate_triton_flexattention_001/profiling_data --iterations 50 --scope candidate_triton_flexattention_001 --wall-ms 0.330810
```
