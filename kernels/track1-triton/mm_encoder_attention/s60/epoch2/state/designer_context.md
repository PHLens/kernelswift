# Designer Context

- role_contract_sha256: `(to be recorded on first dispatch)`
- context_epoch: `0`
- last_completed_round: `000 (baseline)`
- accepted_kernel: `baseline_adapter.py` @ `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e`
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `(none yet — Phase 0 complete)`
- open_hypotheses: `(none — awaiting round-001 dispatch)`
- artifact_read_hashes: `(recorded on first dispatch)`

## Current Bottleneck

- Phase 0 baseline established. Base SDPA -> vendor `_scaled_dot_product_flash_attention`, 2 launches/call, wall 0.227194 ms. GCU device-duration unavailable; runtime-launch evidence only (topsLaunchKernel 21.99us/call). Bottleneck classification pending round-001 Verifier Level 1 evidence.

## Campaign Physics Map (initial)

| Line | Value | Source |
|---|---|---|
| wall_base | 0.227194 ms (v0) / 0.228385 ms (v1 identity) | report_000 |
| Base SDPA | vendor flash attention, 2 launches/call | report_000 census |
| runtime_launch | 21.99 us/call (topsLaunchKernel x2) | report_000 |
| tl.dot | constrained: M/N/K mult-of-16 (T=83 pad to 96) | profile_snapshot/triton_gcu.yaml |
| num_warps | 1/2/4/8 all legal | profile_snapshot/triton_gcu.yaml |

## Ranked Plausible Families (initial)

1. Direct Triton MHA with tl.dot (padding T=83→96) + num_warps tuning.
2. Graph-replay composition (BI150 sibling precedent) if launcher tax dominates.

## Recent Three-round Evidence

- (none yet)

## Open Hypotheses or Checks

- (none yet)

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| (recorded on first dispatch) | | |
