# Round 000 Status — Phase 0 Baseline Establishment

## Phase
- phase: `verifying` — COMPLETE (Phase 0 baseline established)

## Completed Commands
1. Correctness + benchmark (warmup 50 / repeat 100): `PASS accuracy; v0=3.212215 ms, v1=3.206765 ms, speedup=1.002x`
2. Profiler (forward, 20 warmup / 50 iter): PASS; reference + candidate CANN scopes captured and summarized.

## Artifact Hashes
| Artifact | SHA-256 |
|---|---|
| `../base.py` | `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3` |
| `baseline_adapter.py` | `a4f0aa8ac2d59c57059223b1710d20718af1b0f892cd7c373174e531c927133e` |

## Raw Samples
- Correctness: PASS (atol=1e-2, rtol=1e-2)
- Benchmark median wall: v0=3.212215 ms, v1=3.206765 ms
- Profiler reference: device_us_per_call=3094.95, kernel_count_per_call=6.0, device_ratio=0.9635
- Profiler candidate: device_us_per_call=3082.85, kernel_count_per_call=6.0, device_ratio=0.9614

## Next Safe Action
- Phase 0 complete. Await Orchestrator to apply baseline state transition.
