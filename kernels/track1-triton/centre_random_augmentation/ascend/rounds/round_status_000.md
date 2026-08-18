# Round Status 000

- phase: `verifying` (complete)
- round: `000`
- result: `baseline`
- last_safe_step: `all Phase 0 evidence collected`

## Completed Commands

| Step | Command | Status |
|---|---|---|
| 1 | Read role contract + inputs | done |
| 2 | correctness + benchmark (`--warmup 50 --repeat 100`) | done — PASS, v0=2.547680 ms, v1=2.565115 ms |
| 3 | CANN profiler (`--profile-mode forward --profile-iterations 50`) | done — 2 scopes, device_time_available=true |
| 4 | summarize_cann_trace (reference + candidate scopes) | done |

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| base.py | `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553` |
| baseline_adapter.py | `7d4a79ae96328fc03a4489710f68b7f639ddea9cbd5c0f7bb45e1cec5472061b` |
| harness auto_bench.py | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` |

## Randomness / Correctness Observation (CONFIRMED)

Harness seeds per-call (`set_seed(42)` in `run_forward` line 450 and `time_forward` line 470). `compare_values` uses `torch.allclose(atol=1e-2, rtol=1e-2)`. base.py and baseline_adapter.py share identical RNG call order, so identical seed -> identical random draws -> identical outputs -> correctness PASSES. Faithful baseline established.

## Raw Samples

- reference_median_ms: `2.547680`
- candidate_median_ms: `2.565115`
- device_us_per_call (reference scope): `291.866`
- device_us_per_call (candidate scope): `291.501`
- kernel_count_per_call: `110.0`
- device_ratio: `~0.114`

## Next Safe Action

`none — Phase 0 complete. Hand off report_000.md to Orchestrator for baseline acceptance.`
