# Verifier Context State

- role_contract_sha256: `<to-be-computed>`
- context_epoch: 2
- last_completed_round: `001`
- accepted_kernel: `triton_mha_001.py` (fused Triton MHA, accepted under epoch-2 deliverable policy)
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `Round 001: correctness parity PASS (allclose 1e-2); fused _mha_fwd_kernel 1.0/call replaces 2 mcFlashAttn kernels; wall median v0=0.110801 ms, v1=0.164166 ms (~48% slower, expected); device us/call candidate=79.70 vs baseline=15.08; candidate emits 4 extra transpose12_copy_64 kernels/call from .contiguous()`
- open_hypotheses: `Round 1 deliverable (Triton MHA kernel) satisfied via correctness parity; performance secondary under epoch-2 policy`
- artifact_read_hashes: `populated`

## Current Bottleneck

- Candidate device time ~79.7 us/call, dominated by `_mha_fwd_kernel` (~67.1 us/call) — ~4.5x the baseline flash-attention device time (~15.1 us/call). The candidate also emits 4 `transpose12_copy_64` copy kernels/call (~12.6 us/call) from `.contiguous()` materialization of q/k/v in the benchmark path.

## Recent Three-round Evidence

- Round 001 (deliverable): correctness parity PASS (allclose 1e-2); wall median v0=0.110801 ms, v1=0.164166 ms; improvement_pct -48.16%; device us/call candidate=79.70 (5.0 kernels/call), baseline=15.08 (2.0 kernels/call). Hypothesis verdict `partially-confirmed`.

## Open Hypotheses or Checks

- If a subsequent performance round targets the Triton kernel: the `.contiguous()` transpose-copy materialization (4 extra kernels/call) and the single-warp `_mha_fwd_kernel` serial scan over 83 key positions are the primary device-time contributors to address.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | 001 |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | 001 |
| `baseline_adapter.py` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | 001 |
| `triton_mha_001.py` (candidate) | `9fac12aa0298a970c208dbc6af7a602da4f34e43d44921b62aad571ca662c00b` | 001 |
