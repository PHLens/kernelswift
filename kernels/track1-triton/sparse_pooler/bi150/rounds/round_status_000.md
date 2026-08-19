# Round 000 Status

Phase: `verifying` (Phase 0 baseline establishment)
Result: `baseline`

## Progress

| Step | Status | Return code | Evidence |
|---|---|---|---|
| frozen-file SHA256 verification | done | `0` | base `46106baa...`, adapter `359f4c80...`, harness `71fb3ad0...` all match project.md |
| correctness 50/100 | done | `0` | `PASS accuracy; v0=1.070367 ms, v1=1.071914 ms, speedup=0.999x` |
| wall sample 1, 50/100 | done | `0` | v0=1.069584, v1=1.068099 |
| wall sample 2, 50/100 | done | `0` | v0=1.070644, v1=1.068803 |
| wall sample 3, 50/100 | done | `0` | v0=1.070492, v1=1.064938 |
| forward profiler 20/50 | done | `0` | `log/round_000_forward_50iter.pt.trace.json` |
| summarize baseline_base | done | `0` | 743.06 us/call, 11.92 kernels/call |
| summarize candidate_baseline_adapter | done | `0` | 743.82 us/call, 12.0 kernels/call |

## Raw Samples

- reference_raw_samples_ms: `[1.069584, 1.070644, 1.070492]`
- candidate_raw_samples_ms: `[1.068099, 1.068803, 1.064938]`
- reference_median_ms: `1.070492`
- candidate_median_ms: `1.068099`

## Artifact Hashes

- trace SHA256: `74a3604cb6fade42a2ecfb4dc6de409f8329c5e363e47457140e952cc81e995a`

## Next Safe Action

Report complete; awaiting Orchestrator to record terminal baseline transition.
