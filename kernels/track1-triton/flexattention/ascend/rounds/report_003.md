# Report 003

Result: no-improvement

## Identity

- Round: `003`
- Decision: `rounds/decision_003.md`
- Candidate: `triton_flexattention_003.py`
- Accepted reference: `triton_flexattention_002.py`
- Accepted reference report: `rounds/report_002.md`
- Decision SHA256: `c2d0d068f7595bed4aec4e2497b9b390ae875f67dcbcf9de551b448383991b37`
- Candidate SHA256: `4faadac6cd0e3bb5d1faeaddafd899f0fd64c275632d2635f1612bf182686546`
- Accepted reference SHA256: `b0fe058c5b5336978e89933e8f0fed0d5a0449aede33c6b1b7a97c9c319c100f`
- Base SHA256: `12f8a77b8f52b50d513800907b6b21ff9c98709647b306793241f6f8da3cb105`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `c1359d456700562802630e66368ce04856d871a993562ce1437e037af82581b8`
- verification_tier: authoritative
- screening_pairs: `not-run: candidate proceeded directly to authoritative timing (correctness pass, expected improvement)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | `PASS accuracy; v0=0.418195 ms, v1=0.327605 ms, speedup=1.277x` | pass | `python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/triton_flexattention_003.py --warmup 5 --repeat 10 --full-traceback` |
| output dtype and shape unchanged | out `[83,512]` fp16 | harness `compare_values` shape/dtype check passed | pass | correctness gate exit 0, `Summary: 1 passed, 0 failed` |
| causal semantics preserved | lower-triangular mask, scale=1/sqrt(head_size) | allclose(atol=1e-2, rtol=1e-2) passed | pass | correctness gate exit 0 |
| kernel_count_per_call unchanged at 1 | still a single fused kernel | candidate `1.00` kernels/call | pass | `summarize_cann_trace.py` candidate scope |
| output buffer cache behavior unchanged | host cache retained (round-002) | output buffer cache code byte-identical to round 002 | pass | source diff (kernel body changed; `_get_output_buffer`/`self._out_cache` unchanged) |

Conformance, correctness, and every declared guardrail must pass before adoption.

Partial token block note: T=83 with BLOCK_M=16 gives `ceil(83/16)=6` token blocks;
the final block (tokens 80..82) is partial with only 3 valid tokens, guarded by
the 2D `q_mask = m_off[:,None] < T` on both load and store. Correctness passed,
confirming the partial-block guard is correct.

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
- order: `interleaved accepted-reference/candidate` (three pairs; both measured as v1 against immutable base.py v0)
- reference_raw_samples_ms: `[0.292920, 0.296535, 0.299435]`
- candidate_raw_samples_ms: `[0.321280, 0.317370, 0.336525]`
- reference_median_ms: `0.296535`
- candidate_median_ms: `0.321280`
- improvement_pct: `-8.344714789147998`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (0.296535 - 0.321280) / 0.296535 * 100 = -8.3447
```

The unrounded improvement controls the 5% adoption threshold. Profiler time does
not replace this benchmark result. Only authoritative timing can yield `accepted` or `no-improvement`.

Measurement method note (same as Round 2): the harness's `--v0_file` must define
`Model` (only `base.py` does); the accepted reference `triton_flexattention_002.py`
defines `ModelNew`. Both reference and candidate were measured as `--v1_file`
against the identical immutable `base.py` v0, three times each, with identical
flags. The reference median `0.296535` is consistent with the Round 2 accepted
wall `0.281900` (within run-to-run noise).

The candidate is **slower** than the accepted reference: improvement_pct is
negative (-8.34%). This is a terminal `no-improvement` regardless of the device
mechanism, because benchmark wall time controls adoption.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| device_us_per_call | decrease | `54.4268` (reference) → `24.0532` (candidate) us/call, -55.8% | pass | `summarize_cann_trace.py` per-scope summaries |
| kernel_count_per_call | remain 1 (still a single fused kernel) | reference `1.00` → candidate `1.00` | pass | `summarize_cann_trace.py` per-scope summaries |
| output_allocations_per_call | remain 0 (host cache unchanged) | host cache byte-identical to round 002; no additional steady-state output allocation | pass | source diff (round 002 cache retained) |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-003`
- intervention: `replace the elementwise tl.sum rank-1 reductions for QK^T and AV with tl.dot matrix multiplies on a multi-token-per-program layout (BLOCK_M=16 tokens per program), routing both matmuls through the Ascend Cube (BMM) hardware unit instead of the vector path`
- expected_causal_chain: `QK^T and AV matmuls run on the Cube matrix unit instead of elementwise vector reductions → device_us_per_call decreases from ~54 us toward the ~25 us fused core floor → wall_time_ms decreases`
- primary_metric: `wall_time`
- Hypothesis verdict: `partially-confirmed`

The device-side causal chain held exactly as predicted: `tl.dot` on BLOCK_M=16
routed QK^T/AV through the Cube unit and cut device time 54.43 → 24.05 us/call
(56% reduction, reaching the ~25 us `aclnnFlashAttentionScore` core floor). But
the final link — `wall_time_ms decreases` — was **falsified**: wall time
regressed -8.34% because host-side cost rose by ~55 us/call, more than offsetting
the ~30 us/call device saving. The primary metric (wall_time) did not improve, so
the hypothesis is only partially confirmed.

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
| accepted_reference (`reference_triton_flexattention_002`) | 2721.34 | 54.4268 | 50 | 1.00 | 0.296535 | 0.183543 |
| candidate (`candidate_triton_flexattention_003`) | 1202.66 | 24.0532 | 50 | 1.00 | 0.321280 | 0.074867 |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)
```

Host-time derivation (wall minus device per call, same measurement regime):

- accepted_reference host_us_per_call ≈ 296.535 - 54.427 = 242.108 us
- candidate host_us_per_call ≈ 321.280 - 24.053 = 297.227 us
- host increase ≈ 55.12 us/call (device saving 30.37 us/call is outweighed)

### Accepted Reference Top Kernels (`reference_triton_flexattention_002`)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _causal_attn_kernel | 50 | 1.0 | 2721.34 | 54.4268 |

### Candidate Top Kernels (`candidate_triton_flexattention_003`)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _causal_attn_kernel | 50 | 1.0 | 1202.66 | 24.0532 |

Both scopes contain exactly one `_causal_attn_kernel` per forward call. The
device time halved (54.43 → 24.05 us) confirming the Cube/BMM routing worked, but
the wall time regressed because host-side launch/dispatch/synchronization cost
grew by ~55 us/call. The Level0 chrome trace shows only negligible host op
enqueue (~0.03 us/call of `empty_tensor`/`aten::empty`/`aten::view`/`aten::reshape`,
plus `_causal_attn_kernel` host launch median 0.014 us); the ~55 us host increase
is therefore not attributable to visible op enqueue but to the Triton launch /
Cube-unit dispatch / stream-synchronization path for the `tl.dot` kernel, which
the Level0 trace does not attribute. The `HostToDevice` events in the candidate
trace all have zero duration, ruling out data transfer as the cause.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable | `4faadac6cd0e3bb5d1faeaddafd899f0fd64c275632d2635f1612bf182686546` | no-improvement (correctness pass, improvement -8.34%) |

At most one Verifier-to-Coder repair is allowed in the same round.

## evidence_for_next_round

- Round 3 device-side `tl.dot` change is rejected: device time halved (54.43 → 24.05 us/call, hitting the ~25 us floor) but wall time regressed -8.34% (0.296535 → 0.321280 ms) because host-side cost rose ~55 us/call, outweighing the ~30 us device saving.
- The device kernel is now at its structural floor (~24 us, matching the reference `aclnnFlashAttentionScore` core from report_000). Further device-side optimization has little headroom.
- The campaign is now firmly host-bound: candidate host time ~297 us/call dominates wall (device_ratio 0.075). The host path is the Triton kernel launch/dispatch + the harness `sync_devices()` per iteration. The `tl.dot` Cube kernel increased this host cost by ~55 us relative to the `tl.sum` vector kernel — a counterintuitive, attribution-worthy host-side cost that is NOT visible in Level0 op enqueue or HostToDevice events.
- Reverting to the round-002 kernel (single-token `tl.sum`, device ~54 us but host ~242 us) yields better wall time than the `tl.dot` multi-token kernel (device ~24 us but host ~297 us). The tradeoff between device speed and host launch cost is negative on this runtime.

Record evidence only; do not select the next optimization.

## Stop Recommendation

- recommendation: `continue`
- evidence: `no-improvement` increments `performance_miss_streak` to 1 (below `valid_no_improvement_limit=3`); campaign has room under `max_rounds=20`. The device mechanism is understood (Cube routing works but host launch cost dominates); a host-side reversal or launch-path intervention remains an open, falsifiable direction.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/triton_flexattention_003.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/triton_flexattention_002.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/triton_flexattention_003.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/triton_flexattention_003.py --profile --profile-reference-file kernels/track1-triton/flexattention/ascend/triton_flexattention_002.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/flexattention/ascend/log/round_003_forward_50iter.pt.trace.json
```

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/flexattention/ascend/log/profiling_data/reference_triton_flexattention_002/profiling_data --iterations 50 --scope reference_triton_flexattention_002 --wall-ms 0.296535
```

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/flexattention/ascend/log/profiling_data/candidate_triton_flexattention_003/profiling_data --iterations 50 --scope candidate_triton_flexattention_003 --wall-ms 0.321280
```
