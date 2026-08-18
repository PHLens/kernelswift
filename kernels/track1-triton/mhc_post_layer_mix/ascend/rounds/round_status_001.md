# Round 001 Status — Kernel Fusion Candidate

## Phase
- phase: `verifying` — COMPLETE

## Classification
- result: `accepted` (improvement_pct ≈ 264%, well above 5% threshold)

## Completed Commands
1. Correctness + timing pair 1: `PASS accuracy; v0=3.194895 ms, v1=0.876125 ms, speedup=3.647x`
2. Timing pair 2: `PASS accuracy; v0=3.227110 ms, v1=0.887760 ms, speedup=3.635x`
3. Timing pair 3: `PASS accuracy; v0=3.197965 ms, v1=0.879715 ms, speedup=3.635x`
4. Profiler (forward, 20 warmup / 50 iter): reference + candidate CANN scopes summarized.

## Artifact Hashes
| Artifact | SHA-256 |
|---|---|
| base `../base.py` | `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3` |
| accepted reference `baseline_adapter.py` | `a4f0aa8ac2d59c57059223b1710d20718af1b0f892cd7c373174e531c927133e` |
| candidate `candidate_001.py` | `b74e407348d424c9265ddf831b245cda90297a48bdbaa576fa7e6b57b5d121f9` |

## Raw Samples
- v0 medians: 3.194895 / 3.227110 / 3.197965 ms
- v1 medians: 0.876125 / 0.887760 / 0.879715 ms
- Profiler reference: device_us_per_call=3093.30, kernel_count_per_call=6.0
- Profiler candidate: device_us_per_call=619.76, kernel_count_per_call=1.0

## Next Safe Action
- Verifier classification is `accepted`. Await Orchestrator to apply the state
  transition (update last_accepted_kernel / canonical pointers / counters).
