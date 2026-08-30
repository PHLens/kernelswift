# Round Status 000

- round: `000`
- phase: `verifying` (Phase 0 baseline verification)
- result: `baseline`
- verification_tier: `baseline`
- status: `complete`

## Progress

| Step | Status | Detail |
|---|---|---|
| frozen-file SHA256 (before) | done | base/adapter/harness all match project.md |
| runtime fingerprint | done | torch 2.7.1, triton 3.1.0, Iluvatar BI-V150 (7,1) |
| correctness 50/100 | done | `PASS accuracy; v0=1.519272 ms, v1=1.515536 ms, speedup=1.002x`; exit 0 |
| wall sample 1 | done | v0=1.517299 ms, v1=1.518374 ms; exit 0 |
| wall sample 2 | done | v0=1.515616 ms, v1=1.519256 ms; exit 0 |
| wall sample 3 | done | v0=1.523859 ms, v1=1.533459 ms; exit 0 |
| forward profiler 20/50 | done | `log/round_000_forward_50iter.pt.trace.json`; exit 0 |
| summarize baseline_base | done | device_us_per_call=926.395 us, 132.88 kernels/call; exit 0 |
| summarize candidate_baseline_adapter | done | device_us_per_call=926.369 us, 132.88 kernels/call; exit 0 |
| report written | done | `rounds/report_000.md` |
| frozen-file SHA256 (after) | pending | not re-run; files not edited |

## Raw Samples

- v0 (reference `base.py`) raw wall samples ms: `[1.517299, 1.515616, 1.523859]`
- v0 unrounded median: `1.517299`
- v1 (candidate `baseline_adapter.py`) raw wall samples ms: `[1.518374, 1.519256, 1.533459]`
- v1 unrounded median: `1.518374`

## Artifact Hashes

- base_sha256: `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5`
- baseline_adapter_sha256: `ceebdc6185de4c980156a7833073678a0964fb7ccb5edf74b42be6156652eaed`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- trace_sha256: `ca2fa8c940184de974e6dc326c4a4cd7f0a0e9322826395aa28ab42ef825b083`

## Next Safe Action

Phase 0 baseline is complete. Orchestrator owns canonical pointer update and workflow transition to round 001.
