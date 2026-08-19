# Round 002 Status

## Phase

- verifying -> COMPLETE (classification: `no-improvement`)

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `../base.py` (immutable reference) | `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341` |
| `triton_rotary_001.py` (Round 1 accepted) | `51a9a33b82f550abfd80400bb0748b74fd181d0f3c4fd4b5d70b4ca1f5d6984e` |
| `triton_rotary_002.py` (candidate) | `98e86e48f00a25254561ad5bf8ef91824c87e5bb9dfa0279f221633d910b07ab` |
| harness `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` |

## Commands (completed)

- [x] correctness: `--warmup 5 --repeat 10` -> `PASS accuracy`
- [x] authoritative timing: interleaved accepted-reference (triton_rotary_001) vs candidate (triton_rotary_002), 3 pairs each, warmup 50 / repeat 100
- [x] profiler evidence: `--profile --profile-mode forward --profile-warmup 20 --profile-iterations 50`

## Result

- accepted-reference wall median (triton_rotary_001): 0.327830 ms
- candidate wall median (triton_rotary_002): 0.330345 ms
- improvement_pct: -0.77% (marginally slower; < 5.0 threshold)
- device_us_per_call: 48.27 -> 12.116 us (4x reduction, mechanism confirmed)
- kernel_count_per_call: 1.0 (unchanged)
- classification: **no-improvement**
- hypothesis verdict: partially-confirmed (device_us_per_call decreased 4x; wall_time did not decrease)

## Next Safe Action

- Report complete. Orchestrator to record `no-improvement` (increments performance_miss_streak), keep canonical accepted reference as triton_rotary_001.py.
