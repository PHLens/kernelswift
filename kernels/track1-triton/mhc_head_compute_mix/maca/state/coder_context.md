# Coder Context State

- role_contract_sha256: `<computed-at-materialization>`
- context_epoch: 2
- last_completed_round: `001`
- accepted_kernel: `baseline_adapter.py`
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `Round 000 baseline (wall 1.515187 ms, device 534.685 us/call, 133 kernels/call, host-bound). Round 001 candidate-ready: fused Triton kernel, 1 launch/call, smoke speedup 12.641x.`
- open_hypotheses: `H-001 (sinkhorn-loop-fusion) awaiting Verifier measurement.`
- artifact_read_hashes: `see table below`

## Current Bottleneck

- `host-bound: device_ratio ~0.35 (device ~535 us out of ~1515 us wall); 133 kernels/call.`

## Recent Three-round Evidence

- `000, baseline, report_000.md, Phase 0 adapter`
- `001, candidate-ready, coder_result_001.md, sinkhorn-loop-fusion`

## Open Hypotheses or Checks

- `H-001: fused single-kernel Sinkhorn reduces wall time well beyond 5% threshold.`

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` | `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5` | 001 |
| `baseline_adapter.py` | `c3cc90deb93c69f092e876ddc0057ddf6c2d7ac28020f1b7fd1ed260bca72fee` | 001 |
| `rounds/decision_001.md` | `<decision_001.md>` | 001 |
| `project.md` | `<project.md>` | 001 |
| `team-state.md` | `<team-state.md>` | 001 |
| `references/invariants.md` | `<invariants.md>` | 001 |
| `triton_mhcc_001.py` | `f29b71c87712aa3f674c2ec6e448bf1026a81986ecaa070645e691d66e969c39` | 001 |
