# Report 002

Result: accepted

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md`
- Candidate: `triton_grouped_topk_002.py`
- Accepted reference: `triton_grouped_topk_001.py`
- Accepted reference report: `rounds/report_001.md`
- Decision SHA256: `a3b8aebf92a887ec07def2f9a3f804726db620b37f9b6e9f7bb7bbaba6aebf78`
- Candidate SHA256: `9315412c0dd7e2c56a6ce16924c74d7fbf0c4872edab454dd51ce2f62e91413f`
- Accepted reference SHA256: `b7b47d1fec7eaed59eba784dd3300393df12bdc94cab164b9e9d238afb39357a`
- Base SHA256: `12f3324896d6b72bd4eac556839db53fb7045b2965b7f8caf2734551daad0f58`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `d2dc2d5a61930039371da06149b3156c4911a136c6c5df859f50d68ea0e3b871`
- verification_tier: `authoritative`
- screening_pairs: `not-run; formal ordered pairs completed after correctness`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | Candidate matched reference outputs with `atol=1e-2, rtol=1e-2`; harness reported `PASS accuracy` both at the correctness gate and inside the profiling run. | pass | auto_bench `PASS accuracy` |
| output shape | `[83,8]` weights and ids | Candidate returned `[83,8]` for both outputs. | pass | auto_bench compare_values |
| output dtype | weights fp32, ids int32 | Candidate returned fp32 weights and int32 ids. | pass | auto_bench compare_values |
| target/device | preserve NPU execution | Candidate ran on `npu:0` with direct Triton-Ascend launch; no stream or device context created. | pass | compile smoke and formal benchmark |
| launcher conformance | direct launch is the selected Ascend path | Direct Triton launch compiled and ran; kernel, grid, `num_warps=1`, and all `tl.*` primitives unchanged from round 001. | pass | candidate smoke and diff-vs-001 |
| Host Plan conformance | instance-level cache, full cache-key gating, invalidation on key change | `ModelNew.forward` reuses cached `_weights`/`_ids` only when `(tokens, topk, fp32, int32, device)` matches `_output_cache_key`; otherwise allocates fresh buffers on `gating_output.device` and replaces the cache. No global or class-level cache. | pass | code inspection |

## Screening Evidence

Screening was not run as a separate two-pair stage; the formal ordered
reference/candidate benchmark after correctness was used for the adoption
decision (same convention as round 001). Every authoritative pair improved,
so the `screened-out` criterion (both pairs >= 10% slower) is trivially false.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `reference, candidate` ordered pair; unchanged harness command (`--v1_file` only differs)
- pairs: 3
- reference_raw_samples_ms: `[0.326705, 0.333440, 0.320765]`
- candidate_raw_samples_ms: `[0.267220, 0.275295, 0.262465]`
- reference_median_ms: `0.326705`
- candidate_median_ms: `0.267220`
- improvement_pct: `18.207557276442053`

```text
improvement_pct = (0.326705 - 0.267220) / 0.326705 * 100
                = 18.207557276442053
```

The unrounded median clears the 5% adoption threshold.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| wall_time | improve >= 5.0% | `0.326705` ms reference to `0.267220` ms candidate, +18.21% | pass | interleaved 3-pair medians above |
| output_allocations_per_call | decrease from 2.0 toward 0.0 | Steady-state forward performs 0 fresh `torch.empty` allocations (cache-key hit reuses `_weights`/`_ids`); only the first forward allocates. | pass | code inspection of `ModelNew.forward` |
| host_us_per_call | decrease | Host time (`wall_us - device_us_per_call`) drops from `291.743` to `232.086` us/call, -20.45%. | pass | profiler device_us_per_call combined with wall medians |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-002`
- intervention: `reuse the two output tensors across compatible forwards on the ModelNew instance instead of allocating two fresh torch.empty tensors per call`
- expected_causal_chain: `two per-call torch.empty output allocations disappear -> host-side allocation overhead per call decreases -> benchmark wall time decreases`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available` (CANN msprof per-scope captures via torch_npu.profiler)
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`

Reference (round 001 accepted) and candidate (round 002) were captured in
separate CANN msprof runs (one `ASCEND_WORK_PATH` per scope), so each
`ai_core_op_summary.db` belongs to exactly one scope. Device time is
statistically flat across scopes (34.96 vs 35.13 us/call, +0.5% noise), as
expected for a host-only allocation-reuse change; the wall improvement is
entirely host-side.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (`reference_triton_grouped_topk_001`) | 1748.08 | 34.962 | 50 | 1.0 | 0.326705 | 0.107 |
| candidate (`candidate_triton_grouped_topk_002`) | 1756.70 | 35.134 | 50 | 1.0 | 0.267220 | 0.131 |

```text
device_ratio = device_us_per_call / (wall_ms * 1000)
host_us_per_call = wall_us - device_us_per_call
reference host: 326.705 - 34.962 = 291.743 us/call
candidate host: 267.220 - 35.134 = 232.086 us/call
```

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _grouped_topk_kernel | 50 | 1.0 | 1748.08 | 34.962 |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _grouped_topk_kernel | 50 | 1.0 | 1756.70 | 35.134 |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable | `9315412c0dd7e2c56a6ce16924c74d7fbf0c4872edab454dd51ce2f62e91413f` | correctness, timing, and profile completed |

No Verifier-to-Coder repair was required.

## evidence_for_next_round

- Device time is unchanged by the host-only change: 34.96 -> 35.13 us/call on
  the single fused `_grouped_topk_kernel` (1.0 kernel/call), confirming the
  intervention did not touch device work.
- Wall time improves from 0.326705 ms to 0.267220 ms (+18.21%), clearing the
  5% threshold and confirming the causal chain: two per-call `torch.empty`
  allocations removed -> host time drops from 291.743 to 232.086 us/call
  (-20.45%) -> wall time decreases.
- Device ratio rose from 0.107 to 0.131 because wall shrank while device time
  stayed flat; the candidate remains host-bound (device is ~13.1% of wall).
- Remaining bottleneck is host dispatch: ~232 us/call of host time around a
  ~35 us device kernel. The allocation-reuse mechanism is now exhausted
  (steady-state allocation count is 0); remaining host cost is the Triton
  launch/dispatch path.

## Global Stop Observation

- global_stop_observation: `not-observed` (no `target-reached`, no
  `valid-no-improvement-limit`, no `round-budget-exhausted`, no
  `user-intervention`). Orchestrator decides the transition.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/groupedtopk-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_002.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /workspace/kernelswift/.worktrees/groupedtopk-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_001.py --warmup 50 --repeat 100
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_002.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/groupedtopk-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_002.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-reference-file kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_001.py --profile-output kernels/track1-triton/groupedtopk/ascend/log/groupedtopk_round_002_forward_50iter.pt.trace.json
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/groupedtopk/ascend/log/profiling_data/reference_triton_grouped_topk_001/profiling_data --iterations 50 --scope reference_triton_grouped_topk_001 --wall-ms 0.326705
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/groupedtopk/ascend/log/profiling_data/candidate_triton_grouped_topk_002/profiling_data --iterations 50 --scope candidate_triton_grouped_topk_002 --wall-ms 0.267220
```
