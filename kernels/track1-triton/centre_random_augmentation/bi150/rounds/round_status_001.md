# Round Status 001

- phase: `verifying`
- round: `001`
- result: `accepted`
- verification_tier: `authoritative`
- candidate: `triton_centre_random_augmentation_001.py`

## Frozen Artifact Hashes (before measurement)

| Artifact | SHA256 | Matches expectation |
|---|---|---|
| candidate `triton_centre_random_augmentation_001.py` | `4e33276ec28f3695aa08462aa6cb796a160aca47dad889168a7cdd8aa8e16036` | pass (matches coder_result_001 / team-lead) |
| decision `rounds/decision_001.md` | `ad2f891ebb8929b7c8b290388081573f25dbb78dc39ab04585cf258e99a1156b` | pass (matches coder_result_001) |
| accepted reference `baseline_adapter.py` | `012754740961f6ec10d515563e51cd07eeaf35caefe33731d5c1e9a88387fe9b` | pass |
| base `base.py` | `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553` | pass |
| harness `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | pass |

Runtime fingerprint: torch 2.7.1, triton 3.1.0, `Iluvatar BI-V150`, capability (7,1).

## Completed Commands

| Step | Command | Return code | Result |
|---|---|---|---|
| correctness | `--v0_file base.py --v1_file candidate --warmup 50 --repeat 100 --full-traceback` | `0` | PASS accuracy; v0=1.012969 ms, v1=0.709612 ms, speedup=1.427x |
| independent numerical probe | base.Model vs candidate.ModelNew via AST loader | `0` | allclose=True; max_abs_diff=4.77e-07; shape [4,256,3] fp32 both |
| wall pair 1 | `--v0_file wrapper --v1_file candidate --warmup 50 --repeat 100` | `0` | v0=1.023173 ms, v1=0.726311 ms |
| wall pair 2 | same | `0` | v0=1.029014 ms, v1=0.712600 ms |
| wall pair 3 | same | `0` | v0=1.006014 ms, v1=0.707727 ms |
| profiler | `--profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-reference-file baseline_adapter.py` | `0` | trace written |
| summarize `reference_baseline_adapter` | `summarize_trace.py --iterations 50` | `0` | 411.05 us/call, 78.94 kernels/call |
| summarize candidate scope | manual (summarize_trace rejects overlapping Triton device-side scope event) | `0` (manual) | 237.95 us/call, 54.8 kernels/call |

## Raw Samples (authoritative, vs baseline_adapter wrapper)

- reference_raw_samples_ms (v0): `[1.023173, 1.029014, 1.006014]`
- candidate_raw_samples_ms (v1): `[0.726311, 0.712600, 0.707727]`
- reference_median_ms: `1.023173`
- candidate_median_ms: `0.712600`
- improvement_pct: `30.35390886976103`

## Trace

- path: `kernels/track1-triton/centre_random_augmentation/bi150/log/round_001_forward_50iter.pt.trace.json`
- SHA256: `aa1da42ee52a9475f59ef575f251ff078f8c56f1a7a97385f3d13428884ca932`

## Next Safe Action

- Verification complete; result `accepted`. Orchestrator owns canonical pointer update (`last_accepted_kernel` → `triton_centre_random_augmentation_001.py`), counters, and workflow transition. Verifier does not update `last_accepted_kernel`.
