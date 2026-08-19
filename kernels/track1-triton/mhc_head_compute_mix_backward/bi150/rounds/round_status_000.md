# Round Status 000

- phase: `verifying` (Phase 0 baseline verification)
- round: `000`
- result: `baseline`
- started_at: `2026-08-19T16:00:00Z`

## Frozen Artifact Hashes (verified before measurement)

| File | SHA256 | Match |
|---|---|---|
| `base.py` | `28d4d213638e5adc7f6bc52928c8f7596cd8446e3e0201beab364b5c3c7988fc` | pass |
| `baseline_adapter.py` | `98cf1e1e5f493ae4ae08c9391f00db83b3cd783a1b194d9c0d63851091133a5d` | pass |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | pass |

## Command Status

| Step | Command | Return code | Outcome |
|---|---|---|---|
| frozen SHA256 verify | `sha256sum ...` | `0` | all match project.md |
| correctness 50/100 | `auto_bench.py --warmup 50 --repeat 100 --full-traceback` | `0` | `PASS accuracy; v0=0.351913 ms, v1=0.352298 ms, speedup=0.999x` |
| wall sample 1 | 50/100 | `0` | v0=0.352471 ms, v1=0.352724 ms |
| wall sample 2 | 50/100 | `0` | v0=0.351449 ms, v1=0.351174 ms |
| wall sample 3 | 50/100 | `0` | v0=0.348174 ms, v1=0.348107 ms |
| forward profiler 20/50 | `--profile --profile-mode forward` | `0` | trace written |
| summarize `baseline_base` | `summarize_trace.py` | `0` | device 185.599 us/call, 9.74 kernels/call |
| summarize `candidate_baseline_adapter` | `summarize_trace.py` | `0` | device 191.698 us/call, 9.82 kernels/call |

## Raw Samples

- reference_raw_samples_ms: `[0.352471, 0.351449, 0.348174]`
- reference_median_ms: `0.351449`
- candidate_raw_samples_ms: `[0.352724, 0.351174, 0.348107]`
- candidate_median_ms: `0.351174`

## Trace

- path: `log/round_000_forward_50iter.pt.trace.json`
- SHA256: `6b0b555f903e0f61fc23ba90bf14cb9f64fa855d95f55c5e837e58217f54cb97`

## Next Safe Action

Verification complete. Await Orchestrator transition.
