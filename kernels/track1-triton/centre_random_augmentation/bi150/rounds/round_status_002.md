# Round Status 002

- phase: `verifying`
- round: `002`
- result: `accepted`
- verification_tier: `authoritative`
- candidate: `triton_centre_random_augmentation_002.py`

## Frozen Artifact Hashes (before measurement)

| Artifact | SHA256 | Matches expectation |
|---|---|---|
| candidate `triton_centre_random_augmentation_002.py` | `efac6ee782e859701bb14aca04b7f56516a575a5f74507958e1930a95005a530` | pass (matches coder_result_002 / team-lead) |
| decision `rounds/decision_002.md` | `2290e37b81072b794ca5735dddba52ed19805c943a8e7109b598e5fd1f65af8e` | pass (matches coder_result_002) |
| canonical reference `triton_centre_random_augmentation_001.py` | `4e33276ec28f3695aa08462aa6cb796a160aca47dad889168a7cdd8aa8e16036` | pass |
| base `base.py` | `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553` | pass |
| harness `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | pass |

Runtime fingerprint: torch 2.7.1, triton 3.1.0, `Iluvatar BI-V150`, capability (7,1).

## Completed Commands

| Step | Command | Return code | Result |
|---|---|---|---|
| correctness | `--v0_file base.py --v1_file candidate_002 --warmup 50 --repeat 100 --full-traceback` | `0` | PASS accuracy; v0=1.014323 ms, v1=0.243855 ms, speedup=4.160x |
| independent numerical probe | base.Model vs candidate_002.ModelNew via AST loader | `0` | allclose=True; max_abs_diff=4.77e-07; shape [4,256,3] fp32 both |
| wall pair 1 | `--v0_file wrapper(001) --v1_file candidate_002 --warmup 50 --repeat 100` | `0` | v0=0.711623 ms, v1=0.239284 ms |
| wall pair 2 | same | `0` | v0=0.724253 ms, v1=0.244788 ms |
| wall pair 3 | same | `0` | v0=0.711154 ms, v1=0.237824 ms |
| profiler | `--profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-reference-file 001.py` | `0` | trace written |
| summarize reference(001) | manual (summarize_trace rejects overlapping Triton device-side scope event) | `0` (manual) | 238.19 us/call, 54.92 kernels/call |
| summarize candidate(002) | manual (same overlapping-scope reason) | `0` (manual) | 29.24 us/call, 5.52 kernels/call |

## Raw Samples (authoritative, vs canonical 001 wrapper)

- reference_raw_samples_ms (v0 = 001): `[0.711623, 0.724253, 0.711154]`
- candidate_raw_samples_ms (v1 = 002): `[0.239284, 0.244788, 0.237824]`
- reference_median_ms: `0.711623`
- candidate_median_ms: `0.239284`
- improvement_pct: `66.37489232360393`

## Trace

- path: `kernels/track1-triton/centre_random_augmentation/bi150/log/round_002_forward_50iter.pt.trace.json`
- SHA256: `7e70eec09eb4e9240f99644726c80e851b4bb9c9d21e8691bf4b9ec0321d368c`

## Next Safe Action

- Verification complete; result `accepted`. Orchestrator owns canonical pointer update (`last_accepted_kernel` → `triton_centre_random_augmentation_002.py`), counters, and workflow transition. Verifier does not update `last_accepted_kernel`.
