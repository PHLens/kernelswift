# Round 001 Status

## Phase

- verifying -> COMPLETE (classification: `accepted`)

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `../base.py` (accepted reference) | `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341` |
| `baseline_adapter.py` | `40c1a8bfbd9a0e957f21ae8ac686aa4c378a28299fd1f053d1e35b5fa8c443e0` |
| `triton_rotary_001.py` (candidate) | `51a9a33b82f550abfd80400bb0748b74fd181d0f3c4fd4b5d70b4ca1f5d6984e` |
| harness `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` |

## Commands (completed)

- [x] correctness: `--warmup 5 --repeat 10` -> `PASS accuracy` (v0=0.585220, v1=0.333110, 1.757x)
- [x] authoritative timing (3 pairs, warmup 50 / repeat 100):
  - pair 1: v0=0.632515, v1=0.340050 (1.860x)
  - pair 2: v0=0.604445, v1=0.333955 (1.810x)
  - pair 3: v0=0.622330, v1=0.331230 (1.879x)
- [x] profiler evidence: `--profile --profile-mode forward --profile-warmup 20 --profile-iterations 50`
  - reference `baseline_base`: 14 kernels/call, 48.54 us/call
  - candidate `candidate_triton_rotary_001`: 1 kernel/call, 48.27 us/call

## Result

- reference median: 0.622330 ms
- candidate median: 0.333955 ms
- improvement_pct: 46.33% (>= 5.0 threshold)
- classification: **accepted**
- hypothesis verdict: partially-confirmed (kernel_count 14→1 confirmed; device_us_per_call did NOT decrease — unchanged ~48us, as expected for host-bound fragmentation fix)

## Next Safe Action

- Report complete. Orchestrator to apply `accepted` transition and update canonical pointers/counters.
