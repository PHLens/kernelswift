# Round Status 000

- phase: verifying (Phase 0 baseline) — COMPLETE
- current_round: 000
- verification_tier: baseline
- result: baseline

## Completed Commands

1. correctness + benchmark:
   `python3 auto_bench.py --v0_file .../base.py --v1_file .../baseline_adapter.py --warmup 50 --repeat 100`
   -> PASS accuracy; v0=3.396440 ms, v1=3.400635 ms, speedup=0.999x
2. profiler (CANN, forward, warmup 20 / iterations 50):
   -> reference scope + candidate scope captured under log/profiling_data/
3. summarize reference scope -> device_us_per_call=281.185, kernel_count_per_call=136.02, device_ratio=0.0825
4. summarize candidate scope -> device_us_per_call=280.601, kernel_count_per_call=136.00, device_ratio=0.0821

## Artifact Hashes

- base.py: `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5`
- baseline_adapter.py: `c3cc90deb93c69f092e876ddc0057ddf6c2d7ac28020f1b7fd1ed260bca72fee`
- auto_bench.py: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`

## Raw Samples

- reference_median_ms: 3.396440
- candidate_median_ms: 3.400635
- improvement_pct: -0.1235

## Next Safe Action

- none (Phase 0 complete; report_000.md written). Hand off to Orchestrator for
  state transition and phase advancement.
