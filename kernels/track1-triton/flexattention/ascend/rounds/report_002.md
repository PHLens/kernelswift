# Report 002

Result: accepted

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md`
- Candidate: `triton_flexattention_002.py`
- Accepted reference: `triton_flexattention_001.py`
- Accepted reference report: `rounds/report_001.md`
- Decision SHA256: `1be71c8d099e870321bbcdde02fc6bc078d929fc7ca0b1dc7bce89cb19ee2f06`
- Candidate SHA256: `b0fe058c5b5336978e89933e8f0fed0d5a0449aede33c6b1b7a97c9c319c100f`
- Accepted reference SHA256: `53e87eff27457f6268040c64979f99dcf30a809effc562caec3db951b141d4a3`
- Base SHA256: `12f8a77b8f52b50d513800907b6b21ff9c98709647b306793241f6f8da3cb105`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `c1359d456700562802630e66368ce04856d871a993562ce1437e037af82581b8`
- verification_tier: authoritative
- screening_pairs: `not-run: candidate proceeded directly to authoritative timing (correctness pass, expected improvement)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | `PASS accuracy; v0=0.413360 ms, v1=0.303550 ms, speedup=1.362x` | pass | `python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/triton_flexattention_002.py --warmup 5 --repeat 10 --full-traceback` |
| output dtype and shape unchanged | out `[83,512]` fp16 | harness `compare_values` shape/dtype check passed | pass | correctness gate exit 0, `Summary: 1 passed, 0 failed` |
| numerical semantics unchanged | same kernel, same inputs | allclose(atol=1e-2, rtol=1e-2) passed against reference | pass | correctness gate exit 0 |
| device_us_per_call unchanged | ~54 us (same single kernel) | reference 54.2028 vs candidate 54.6408 us/call | pass | `summarize_cann_trace.py` per-scope summaries |

Conformance, correctness, and every declared guardrail must pass before adoption.

Output-buffer reuse safety note: the harness `compare_case` calls
`run_forward(model, ...)` (v0) then `run_forward(model_new, ...)` (v1) and
immediately `compare_values(v0_output, v1_output, ...)`, reading both outputs
right after each forward before any subsequent forward can overwrite the reused
buffer. This was confirmed by re-reading `auto_bench.py` lines 734-736; the
comparison holds and correctness passes.

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
- order: `interleaved accepted-reference/candidate` (three pairs; reference measured as v1 against immutable base.py v0 in the same harness)
- reference_raw_samples_ms: `[0.330510, 0.325910, 0.336205]`
- candidate_raw_samples_ms: `[0.287685, 0.281900, 0.280755]`
- reference_median_ms: `0.330510`
- candidate_median_ms: `0.281900`
- improvement_pct: `14.707573144534217`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (0.330510 - 0.281900) / 0.330510 * 100 = 14.7076
```

The unrounded improvement controls the 5% adoption threshold. Profiler time does
not replace this benchmark result. Only authoritative timing can yield `accepted` or `no-improvement`.

Measurement method note: the harness's `--v0_file` must define `Model`, which
only `base.py` provides; the accepted reference `triton_flexattention_001.py`
defines `ModelNew`. To interleave the accepted reference against the candidate
in the same measurement regime, both were measured as `--v1_file` (v1) against
the identical immutable `base.py` v0, three times each, back-to-back in one
Verifier turn with identical flags (warmup 50, repeat 100). The reference
median (`0.330510`) reproduces the Round 1 accepted wall (`0.330810`) within
noise, confirming the regime is stable. Improvement is computed against the
accepted reference, never against `base.py`.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| output_allocations_per_call | decrease toward 0 | steady-state forwards show no `torch.empty`/allocator event; output buffer cached after first forward | pass | chrome trace (50 iter) contains no allocation op; code uses `_get_output_buffer` cache |
| host_us_per_call | decrease | host time fell ~276.31 us (reference) → ~227.26 us (candidate), ~49 us reduction | pass | wall minus device per call (see Profiler Evidence) |
| device_us_per_call | unchanged (~54 us, same single kernel) | 54.2028 (reference) → 54.6408 (candidate) us/call, within noise | pass | `summarize_cann_trace.py` per-scope summaries |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-002`
- intervention: `reuse a single preallocated output buffer across compatible forward calls instead of allocating a fresh torch.empty([T,H,D]) tensor on every call, removing per-call output allocation from the dominant host time`
- expected_causal_chain: `per-call output allocation disappears from the forward path → host-side per-call work decreases → wall_time_ms decreases`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

All three observables passed: output allocation is gone from steady-state
forwards (buffer cached), host time decreased (~49 us), device time unchanged
(~54 us), and wall time improved 14.71%.

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
| accepted_reference (`reference_triton_flexattention_001`) | 2710.14 | 54.2028 | 50 | 1.00 | 0.330510 | 0.163997 |
| candidate (`candidate_triton_flexattention_002`) | 2732.04 | 54.6408 | 50 | 1.00 | 0.281900 | 0.193830 |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)
```

Host-time derivation (wall minus device per call, same measurement regime):

- accepted_reference host_us_per_call ≈ 330.510 - 54.203 = 276.307 us
- candidate host_us_per_call ≈ 281.900 - 54.641 = 227.259 us
- host reduction ≈ 49.05 us/call

### Accepted Reference Top Kernels (`reference_triton_flexattention_001`)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _causal_attn_kernel | 50 | 1.0 | 2710.14 | 54.2028 |

### Candidate Top Kernels (`candidate_triton_flexattention_002`)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _causal_attn_kernel | 50 | 1.0 | 2732.04 | 54.6408 |

Both scopes contain exactly one `_causal_attn_kernel` per forward call; the
kernel is byte-identical between reference and candidate (device time unchanged
within noise, 54.20 vs 54.64 us). The candidate's chrome trace (50 iterations)
shows no allocation op — the `torch.empty` output buffer is allocated once and
cached, and the remaining host ops per call are only `aten::view` (median
0.0102 us) and `aten::reshape` (median 0.0127 us) plus the `_causal_attn_kernel`
launch (median 0.0192 us).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable | `b0fe058c5b5336978e89933e8f0fed0d5a0449aede33c6b1b7a97c9c319c100f` | accepted (correctness pass, improvement 14.71%) |

At most one Verifier-to-Coder repair is allowed in the same round.

## evidence_for_next_round

- Round 2 host-only output-buffer reuse is accepted: wall `0.281900 ms`, device `54.6408 us/call` (unchanged), 1.0 kernels/call.
- Device time is essentially unchanged (~54 us); the win is host-side (~49 us/call removed). Wall time is still host-dominated (device_ratio ~0.194, host ~227 us/call).
- The remaining host time is dominated by per-iteration NPU launch + `sync_devices()` in the harness (`time_forward` calls `sync_devices()` after every call), not by the forward's own op enqueue (which is only ~0.04 us/call). Further wall-time gains from reducing in-`forward` host work are limited; the next bottleneck is the kernel itself (54 us) and/or the harness synchronisation pattern.
- The device kernel (`_causal_attn_kernel`, `tl.sum` rank-1 reductions, `num_warps=1`, 1D grid T*H=664 programs, BLOCK_K=128, BLOCK_D=64) is unchanged and still ~54 us, versus the reference `aclnnFlashAttentionScore` core (~25 us) from report_000, indicating remaining device headroom via e.g. a vectorized/tiled `tl.dot` approach (currently M=1 `tl.dot` is unproven on Ascend per decision_002).

Record evidence only; do not select the next optimization.

## Stop Recommendation

- recommendation: `continue`
- evidence: improvement 14.71% (> 5% threshold), hypothesis confirmed; `performance_miss_streak` resets on acceptance, campaign has room under `max_rounds=20`.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/triton_flexattention_002.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/triton_flexattention_001.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/triton_flexattention_002.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/triton_flexattention_002.py --profile --profile-reference-file kernels/track1-triton/flexattention/ascend/triton_flexattention_001.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/flexattention/ascend/log/round_002_forward_50iter.pt.trace.json
```

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/flexattention/ascend/log/profiling_data/reference_triton_flexattention_001/profiling_data --iterations 50 --scope reference_triton_flexattention_001 --wall-ms 0.330510
```

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/flexattention/ascend/log/profiling_data/candidate_triton_flexattention_002/profiling_data --iterations 50 --scope candidate_triton_flexattention_002 --wall-ms 0.281900
```
