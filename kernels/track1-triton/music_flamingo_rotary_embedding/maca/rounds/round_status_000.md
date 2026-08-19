# Round Status 000

- round: `000`
- phase: `measuring`
- result: `baseline`
- updated_at: `2026-08-18T22:20:00Z`

## Verification Timeline

| Step | Status | Completed commands | Notes |
|---|---|---|---|
| 1. Correctness | done | `auto_bench.py --warmup 5 --repeat 10 --full-traceback` | exit 0, `PASS accuracy` |
| 2. Authoritative wall timing (x3) | done | `auto_bench.py --warmup 50 --repeat 100` (3 runs) | medians recorded |
| 3. Profiler (forward) | done | `auto_bench.py --profile --profile-mode forward ...` | trace written; known C500 overlap issue filtered |
| 4. Trace summary | done | `summarize_trace.py` (both scopes, filtered trace) | device us/call + kernel counts recorded |

## Raw Samples

### Correctness

- exit code: `0`
- output: `PASS accuracy; v0=0.204476 ms, v1=0.198539 ms, speedup=1.030x`

### Authoritative wall timing (warmup 50, repeat 100)

| Run | v0_ms | v1_ms | speedup |
|---:|---:|---:|---:|
| 1 | 0.191406 | 0.190557 | 1.004x |
| 2 | 0.192640 | 0.190242 | 1.013x |
| 3 | 0.191099 | 0.191967 | 0.995x |

- reference_median_ms = `0.191406`
- candidate_median_ms = `0.190557`

### Profiler summary

- baseline_base (v0): device_total_us `2559.98046875`, device_us_per_call `51.199609375`, kernel_count_total `550`, kernel_count_per_call `11.0`
- candidate_baseline_adapter (v1): device_total_us `2547.44482421875`, device_us_per_call `50.948896484375`, kernel_count_total `550`, kernel_count_per_call `11.0`

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `../base.py` | `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341` |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` |
| `baseline_adapter.py` | `40c1a8bfbd9a0e957f21ae8ac686aa4c378a28299fd1f053d1e35b5fa8c443e0` |

## Next Safe Action

Round 000 is a Phase 0 baseline: no decision/hypothesis, no adoption comparison.
Verifier has completed all measurement steps. Orchestrator may record the
baseline result and advance the canonical pointers (`last_accepted_kernel` ->
`baseline_adapter.py`) as governed by the workflow, then dispatch Round 001 design.
