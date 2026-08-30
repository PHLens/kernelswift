# Round Status 000

- phase: `verifying`
- round: `000`
- result: `baseline`
- verification_tier: `baseline`
- started_at: `2026-08-19T12:30:00Z` (project_started_at)

## Frozen Artifact Hashes (before measurement)

| Artifact | SHA256 | Matches project.md |
|---|---|---|
| `base.py` | `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553` | pass |
| `baseline_adapter.py` | `012754740961f6ec10d515563e51cd07eeaf35caefe33731d5c1e9a88387fe9b` | pass |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | pass |

Runtime fingerprint: torch 2.7.1, triton 3.1.0, `Iluvatar BI-V150`, capability (7,1) — matches project.md.

## Completed Commands

| Step | Command | Return code | Result |
|---|---|---|---|
| correctness | `--warmup 50 --repeat 100 --full-traceback` | `0` | PASS accuracy; v0=1.084037 ms, v1=1.077019 ms, speedup=1.007x |
| wall sample 1 | `--warmup 50 --repeat 100` | `0` | v0=1.070444 ms, v1=1.070630 ms |
| wall sample 2 | `--warmup 50 --repeat 100` | `0` | v0=1.073250 ms, v1=1.072018 ms |
| wall sample 3 | `--warmup 50 --repeat 100` | `0` | v0=1.078100 ms, v1=1.076946 ms |
| profiler | `--profile --profile-mode forward --profile-warmup 20 --profile-iterations 50` | `0` | trace written |
| summarize `baseline_base` | `summarize_trace.py --iterations 50` | `0` | 420.68 us/call, 78.8 kernels/call |
| summarize `candidate_baseline_adapter` | `summarize_trace.py --iterations 50` | `0` | 423.57 us/call, 79.02 kernels/call |

## Raw Samples

- reference_raw_samples_ms (v0): `[1.070444, 1.073250, 1.078100]`
- candidate_raw_samples_ms (v1): `[1.070630, 1.072018, 1.076946]`
- reference_median_ms: `1.073250`
- candidate_median_ms: `1.072018`

## Trace

- path: `kernels/track1-triton/centre_random_augmentation/bi150/log/round_000_forward_50iter.pt.trace.json`
- SHA256: `397ecd670561b94933fc2cde22561992078b324c19252cc473c2863835cd8739`

## Next Safe Action

- Verification complete; result `baseline`. Orchestrator owns canonical pointer update and workflow transition. Verifier does not update `last_accepted_kernel`.
