# Round Status 001

- round: `001`
- phase: `measuring`
- result: `accepted`
- updated_at: `2026-08-18T22:30:00Z`

## Verification Timeline

| Step | Status | Completed commands | Notes |
|---|---|---|---|
| 1. Correctness | done | `auto_bench.py --warmup 5 --repeat 10 --full-traceback` | exit 0, `PASS accuracy` |
| 2. Authoritative wall timing (x3) | done | `auto_bench.py --warmup 50 --repeat 100` (3 runs) | medians recorded |
| 3. Profiler (forward) | done | `auto_bench.py --profile --profile-mode forward ...` | trace written; C500 overlap issue filtered |
| 4. Trace summary | done | `summarize_trace.py` (both scopes, filtered trace) | device us/call + kernel counts recorded |

## Raw Samples

### Correctness

- exit code: `0`
- output: `PASS accuracy; v0=0.195046 ms, v1=0.082840 ms, speedup=2.354x`

### Authoritative wall timing (warmup 50, repeat 100)

| Run | v0_ms | v1_ms | speedup |
|---:|---:|---:|---:|
| 1 | 0.180151 | 0.079770 | 2.258x |
| 2 | 0.180017 | 0.080036 | 2.249x |
| 3 | 0.184270 | 0.080713 | 2.283x |

- reference_median_ms = `0.180151`
- candidate_median_ms = `0.080036`
- improvement_pct = `55.5727`

### Profiler summary

- baseline_base (v0): device_us_per_call `50.692626953125`, kernel_count_per_call `11.0`
- candidate_triton_rotary_001 (v1): device_us_per_call `16.901123046875`, kernel_count_per_call `1.0` (single `_rotary_embed_fused_kernel`)

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `../base.py` | `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341` |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` |
| `baseline_adapter.py` | `40c1a8bfbd9a0e957f21ae8ac686aa4c378a28299fd1f053d1e35b5fa8c443e0` |
| `triton_rotary_001.py` | `dec9aa12bc50886503831c48b82767e6a76ecd29d3a5c29cb41185d6ef633c39` |
| `decision_001.md` | `6e5741d2ccabe1883520625bfdb5a8e6e7f334b9ea995de5069943246342eceb` |

## Next Safe Action

Round 001 classified `accepted` (correctness pass + 55.57% wall improvement).
Verifier does not advance canonical pointers; Orchestrator will record the
accepted result, advance `last_accepted_kernel` -> `triton_rotary_001.py` and
`last_accepted_report` -> `rounds/report_001.md`, and dispatch Round 002 design.
