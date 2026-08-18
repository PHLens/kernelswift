# Round 000 Status

## Phase

- Phase 0 baseline verification — COMPLETE

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `../base.py` | `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341` |
| `baseline_adapter.py` | `40c1a8bfbd9a0e957f21ae8ac686aa4c378a28299fd1f053d1e35b5fa8c443e0` |
| harness `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` |

## Commands (completed)

- [x] correctness: `python3 auto_bench.py --v0_file .../base.py --v1_file .../baseline_adapter.py --warmup 5 --repeat 10 --full-traceback` -> `PASS accuracy` (v0=0.601575 ms, v1=0.608120 ms)
- [x] baseline benchmark: `--warmup 50 --repeat 100` -> v0=0.581820 ms, v1=0.581270 ms, speedup=1.001x
- [x] profiler evidence: `--profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output .../log/music_rotary_baseline_forward_50iter.pt.trace.json`
  - CANN scopes: `baseline_base` and `candidate_baseline_adapter`

## Raw results

- baseline wall_time_ms (reference median): 0.581820
- candidate wall_time_ms (median): 0.581270
- device_us_per_call: 47.7836 (baseline), 47.6552 (candidate)
- kernel_count_per_call: 14.0 (both)

## Next Safe Action

- Phase 0 complete; report written. Orchestrator to record transition to next phase.
