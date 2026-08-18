# Report 001

Result: no-improvement

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_rotary_001.py`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `0d279802517154dbbcd5e5ebb60c3e3d8219bd3132cc422251192d9826831360`
- Candidate SHA256: `74a960f54b4519f43948d4c2e374d9d93edbf76645ab7adc0d63595f5a4802b5`
- Accepted reference SHA256: `9fc87abbe0e6268f06c969e94f5400abea51cdf315276a4ac5cef5bd0ad8a26f`
- Base SHA256: `99829754f4cdc4bfd2808e051de549f0791414241e7fdbad7a1b8294a15be475`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `a1ee09ca54ab2210943bd030a6649c57d96b09d4c1beed863f4a98681ae425f2`
- verification_tier: authoritative
- screening_pairs: not-run (correctness passed; candidate proceeded directly to authoritative timing)

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass at atol=1e-2, rtol=1e-2, equal_nan=True | `PASS accuracy` | pass | `auto_bench.py --warmup 5 --repeat 10 --full-traceback` |
| output tuple structure and shape | tuple (cos, sin), each `[4,32,128]` fp32 | harness `compare_values` recursed tuple and matched shapes/dtype | pass | correctness gate passed |
| state_dict keys {inv_freq, position_angles} | unchanged keys | both registered buffers present in `__init__` | pass | candidate source `__init__` |

## Screening Evidence

Not run: correctness passed and candidate was not expected to be screened out (it was not >=10% slower on a short pair basis but proceeded directly to authoritative timing per the verifier contract; the single correctness pass already ran a short `--warmup 5 --repeat 10` timing that showed ~0.13x speedup, i.e. a large slowdown, so authoritative timing was run to quantify it).

| Pair | Reference short wall ms | Candidate short wall ms | Candidate slower pct | Evidence |
|---:|---:|---:|---:|---|
| 1 | 0.691403 | 5.292693 | -665.7% | `auto_bench.py --warmup 5 --repeat 10` |

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate`
- reference_raw_samples_ms: harness emits median only (v0=base.py); authoritative median `0.474164` (base.py) vs accepted-reference `baseline_adapter.py` median `0.464657` from report_000
- candidate_raw_samples_ms: harness emits median only (v1); authoritative median `5.162427`
- reference_median_ms: `0.464657` (accepted reference `baseline_adapter.py`, from report_000)
- candidate_median_ms: `5.162427`
- improvement_pct: `-1010.99`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
             = (0.464657 - 5.162427) / 0.464657 * 100
             = -1010.99
```

The candidate is ~11x slower than the accepted reference, far from the >=5% adoption threshold.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| runtime_launch_count_per_call | decrease | 13 -> 1 (baseline 13.0, candidate 1.0) | pass | `summarize_trace.py --scope baseline_base` / `--scope candidate_triton_rotary_001` |
| runtime_launch_us_per_call | decrease | 139.38 -> 9.70 us | pass | trace runtime_launch_total_us normalized by 50 iters |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `fuse the forward elementwise/view ops (arange, mul/div, repeat_interleave, broadcast, cat, angle multiply, cos, sin) into a single Triton elementwise kernel that writes both cos and sin output buffers in one launch`
- expected_causal_chain: `13 eager topsLaunchKernel launches collapse to 1 fused Triton launch -> runtime_launch_us_per_call decreases -> intermediate eager tensors disappear -> wall time decreases`
- primary_metric: `wall_time`
- Hypothesis verdict: `falsified`

The launch-collapse mechanism was confirmed (13 -> 1 launches, launch overhead 139.38 -> 9.70 us/call), but the primary wall-time outcome was falsified: wall time regressed from 0.464657 ms to 5.162427 ms. The single fused Triton kernel itself is ~5.15 ms on device — far slower than the eager chain's combined device+launch time (~0.465 ms). The launch savings (~129.7 us) are dwarfed by the kernel's own device execution cost.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `unavailable: GCU trace exposes runtime launch events but no cat=kernel device durations`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `runtime_launch_total_us`, `runtime_launch_us_per_call`, `runtime_launch_count_total`, `runtime_launch_count_per_call`, `runtime_launches` when applicable

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (baseline_base) | unavailable | unavailable | unavailable | unavailable | 0.474164 | unavailable |
| candidate | unavailable | unavailable | unavailable | unavailable | 5.162427 | unavailable |

Device time is unavailable on the GCU exporter; only runtime launch events are recorded. Runtime launch evidence (the only device-side proxy) is summarized below.

### Accepted Reference (baseline_base) Runtime Launches

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| topsLaunchKernel | 650 | 13.0 | 6968.77 | 139.38 |

### Candidate (triton_rotary_001) Runtime Launches

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| topsModuleLaunchKernel | 50 | 1.0 | 485.14 | 9.70 |

The launch count collapsed 13 -> 1 per call and launch overhead dropped ~129.7 us/call as designed, but the candidate's wall time is dominated by the single Triton kernel's device execution (~5.15 ms), which far exceeds the entire eager chain.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable | `74a960f54b4519f43948d4c2e374d9d93edbf76645ab7adc0d63595f5a4802b5` | correctness pass; no-improvement (wall regression) |

## evidence_for_next_round

- Launch fusion succeeded mechanically: `runtime_launch_count_per_call` dropped from 13.0 to 1.0 and `runtime_launch_us_per_call` dropped from 139.38 us to 9.70 us, exactly matching the decision's expected causal chain.
- The primary metric was falsified: candidate wall time 5.162427 ms vs accepted reference 0.464657 ms (improvement_pct = -1010.99). The single fused Triton kernel's device execution cost (~5.15 ms) far exceeds the eager chain's total (~0.465 ms).
- The candidate uses `num_warps=1` with `block = next_power_of_2(16384) = 16384`, i.e. a single program processing all 16384 output elements serially within one thread-block; the resulting device execution is extremely slow on GCU (multi_processor_count=2). The launch-overhead win (~129.7 us) is dwarfed by the kernel body cost.
- Device kernel time is not exposed by the GCU exporter (`device_time_available: false`), so the regression is attributed via wall-time delta and runtime-launch evidence only.

## Stop Recommendation

- recommendation: `continue`
- evidence: correctness and the launch-collapse mechanism are confirmed; only the wall-time primary metric regressed. The bottleneck has shifted from host launch overhead to the fused kernel's own device execution cost. A revised candidate (e.g. a grid that exposes more parallelism / avoids a single serial block, or a different launch configuration) remains viable against the >=5% wall target.

## Exact Reproduction Commands

```bash
cd /root/kernelswift/.worktrees/rotary-embedding-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/s60/triton_rotary_001.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/kernelswift/.worktrees/rotary-embedding-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/s60/triton_rotary_001.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift/.worktrees/rotary-embedding-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/s60/triton_rotary_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/music_flamingo_rotary_embedding/s60/log/rotary_round_001_forward_50iter.pt.trace.json
```
