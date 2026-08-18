# Round Status 001

- phase: `verifying` (complete)
- round: `001`
- result: `accepted`
- last_safe_step: `authoritative timing (3 pairs) + CANN profiler complete`

## Completed Commands

| Step | Command | Status |
|---|---|---|
| 1 | Read decision_001.md, coder_result_001.md, candidate, team-state | done |
| 2 | Correctness + pair 1 (v0=base.py, v1=candidate, warmup 50/repeat 100) | done — PASS, 1.192x |
| 3 | Pair 2 | done — PASS, 1.239x |
| 4 | Pair 3 | done — PASS, 1.214x |
| 5 | CANN profiler (candidate + baseline_adapter reference, 50 iter) | done |
| 6 | summarize_cann_trace (both scopes) | done |

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| accepted reference (baseline_adapter.py) | `7d4a79ae96328fc03a4489710f68b7f639ddea9cbd5c0f7bb45e1cec5472061b` |
| candidate (triton_centre_random_aug_001.py) | `dcfeb039d3d8526d756775015560a22e1b0cd447c5c6dbd69ad12d3a3f0ee089` |
| decision_001.md | `23bb4a4e3b2830b7023216c5485b9fbf447ddf2f2ce62141697fbc21561cd31b` |

## Raw Samples

- reference per-pair medians: `2.463270, 2.490540, 2.457635` -> median `2.463270`
- candidate per-pair medians: `2.066240, 2.010540, 2.023920` -> median `2.023920`
- improvement_pct: `17.84` (>= 5% threshold)
- device_us_per_call: reference `294.970`, candidate `216.063` (-26.8%)
- kernel_count_per_call: reference `110.0`, candidate `64.0`

## Next Safe Action

`none — round 001 verification complete. Hand off report_001.md (result: accepted) to Orchestrator.`
