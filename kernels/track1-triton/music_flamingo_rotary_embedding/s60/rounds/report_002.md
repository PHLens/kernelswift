# Report 002

Result: no-improvement

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md`
- Candidate: `triton_rotary_002.py`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `300d29bbd59c051d77c2b06f7041a64c440b5f68b9d11c4b6756b33b7e8e28fd`
- Candidate SHA256: `a1c5c38a4ecd0a038ebbcd9e6f04b0b5a18e437aea8acabd4104a4e5d579ad9d`
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

Not run: correctness passed and the candidate proceeded directly to authoritative timing per the verifier contract.

| Pair | Reference short wall ms | Candidate short wall ms | Candidate slower pct | Evidence |
|---:|---:|---:|---:|---|
| 1 | 0.626188 | 0.551894 | +13.5% faster (short warmup) | `auto_bench.py --warmup 5 --repeat 10` |

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate`
- reference_raw_samples_ms: harness emits median only (v0=base.py); base.py median `0.478748` (this round); accepted-reference `baseline_adapter.py` median `0.464657` (report_000)
- candidate_raw_samples_ms: harness emits median only (v1); authoritative median `0.525050`
- reference_median_ms: `0.464657` (accepted reference `baseline_adapter.py`, from report_000)
- candidate_median_ms: `0.525050`
- improvement_pct: `-13.00`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
             = (0.464657 - 0.525050) / 0.464657 * 100
             = -13.00
```

The grid-parallelism repair collapsed device time from ~5.15 ms (Round 1) to ~0.525 ms, a ~10x device improvement, but the fused kernel still does not beat the eager baseline (0.464657 ms). improvement_pct is negative (-13.00%), below the >=5% adoption threshold.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| runtime_launch_count_per_call | remain | 1.0 (candidate) vs 13.0 (baseline); launch stayed at 1 | pass | `summarize_trace.py --scope candidate_triton_rotary_002` |
| runtime_launch_us_per_call | remain | 8.51 us (candidate) vs 105.44 us (baseline) | pass | trace runtime_launch_total_us normalized by 50 iters |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-002`
- intervention: `partition the 16384-element fused elementwise kernel across a grid of many programs (BLOCK=128, grid=128) instead of a single program with BLOCK=16384, keeping num_warps=1 and one launch that writes both cos and sin`
- expected_causal_chain: `many programs run concurrently on the device instead of one serial warp -> the fused kernel's device execution time collapses toward eager elementwise time -> launch stays at 1 and launch overhead stays low -> wall time decreases`
- primary_metric: `wall_time`
- Hypothesis verdict: `partially-confirmed`

The launch mechanism was confirmed: launch count remained at 1 (13 -> 1 vs baseline), and the grid repair collapsed device execution from ~5.15 ms (Round 1) to ~0.525 ms — the device-bound cause named in decision_002 was correctly diagnosed and fixed. However the primary wall-time metric was not met: candidate 0.525050 ms vs accepted reference 0.464657 ms (-13.00%), still below the >=5% threshold. The fused single-launch kernel now approaches but does not beat the eager chain.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `unavailable: GCU trace exposes runtime launch events but no cat=kernel device durations`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `runtime_launch_total_us`, `runtime_launch_us_per_call`, `runtime_launch_count_total`, `runtime_launch_count_per_call`, `runtime_launches` when applicable

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (baseline_base) | unavailable | unavailable | unavailable | unavailable | 0.478748 | unavailable |
| candidate | unavailable | unavailable | unavailable | unavailable | 0.525050 | unavailable |

Device time is unavailable on the GCU exporter; only runtime launch events are recorded.

### Accepted Reference (baseline_base) Runtime Launches

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| topsLaunchKernel | 650 | 13.0 | 5272.10 | 105.44 |

### Candidate (triton_rotary_002) Runtime Launches

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| topsModuleLaunchKernel | 50 | 1.0 | 425.45 | 8.51 |

The launch count stayed at 1 per call (13 -> 1 vs baseline) and launch overhead stayed low (~8.51 us/call vs 105.44 us/call baseline), confirming the "remain" expectation for both observables. The wall-time gap is now dominated by the fused kernel's residual device execution, not launch overhead.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable | `a1c5c38a4ecd0a038ebbcd9e6f04b0b5a18e437aea8acabd4104a4e5d579ad9d` | correctness pass; no-improvement (wall -13.00%) |

## evidence_for_next_round

- The grid-parallelism repair (BLOCK=128, grid=(128,), num_warps=1) collapsed the fused kernel's device execution from ~5.15 ms (Round 1, single serial program BLOCK=16384) to ~0.525 ms — a ~10x improvement that confirms the Round 1 regression was a launch-configuration defect, not an intrinsic fusion cost.
- The launch-collapse mechanism is fully retained: `runtime_launch_count_per_call` = 1.0 and `runtime_launch_us_per_call` = 8.51 us, versus baseline 13.0 launches and 105.44 us/call.
- Despite the 10x device improvement, the fused kernel (0.525050 ms) still does not beat the eager baseline (0.464657 ms), a residual -13.00% wall gap. The bottleneck is now the fused kernel's remaining device execution cost relative to the highly-optimized eager elementwise ops.
- `tl.arange(0, 128)` extent and `tl.cos`/`tl.sin` are now PROVEN available on this GCU runtime; no capability-miss.
- Device kernel time remains unavailable on the GCU exporter (`device_time_available: false`), so attribution relies on wall-time delta and runtime-launch evidence only.

## Stop Recommendation

- recommendation: `continue`
- evidence: correctness and both launch observables pass, and the device-bound defect from Round 1 was fixed (~10x device improvement). The remaining -13.00% wall gap is small; the fused kernel now nearly matches the eager baseline. A further micro-optimization (e.g. tuning BLOCK/warp count, vectorized loads, or eliminating redundant register traffic) may close the remaining ~60 us gap against the 0.464657 ms target. The primary wall-time metric is still unmet, so this round is no-improvement, not accepted.

## Exact Reproduction Commands

```bash
cd /root/kernelswift/.worktrees/rotary-embedding-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/s60/triton_rotary_002.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/kernelswift/.worktrees/rotary-embedding-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/s60/triton_rotary_002.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift/.worktrees/rotary-embedding-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/s60/triton_rotary_002.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/music_flamingo_rotary_embedding/s60/log/rotary_round_002_forward_50iter.pt.trace.json
```
