# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: 1
- last_completed_round: `001`
- accepted_kernel: `baseline_adapter.py`
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `Round 001 candidate-ready: fused host-bound rotary-embedding chain (11 launches -> 1 Triton-MACA elementwise kernel, num_warps=1, direct launch); smoke PASS accuracy (2.479x).`
- open_hypotheses: `H-001 host-bound fusion awaiting Verifier authoritative measurement; expected kernel_count/call 11 -> 1 and wall improvement >= 5%.`
- artifact_read_hashes: `see table below`

## Current Bottleneck

- `Host-bound: 11 small PyTorch elementwise kernels per forward call, device_ratio ~0.267 (~73% wall is host launch overhead).`

## Recent Three-round Evidence

- `Round 000 baseline (report_000.md): wall 0.190557 ms, 11 kernels/call, device 50.95 us/call.`

## Open Hypotheses or Checks

- `H-001: fuse full rotary-embedding chain into one Triton-MACA kernel; candidate triton_rotary_001.py written; await Verifier.`

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` | `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341` | 001 |
| `baseline_adapter.py` | `40c1a8bfbd9a0e957f21ae8ac686aa4c378a28299fd1f053d1e35b5fa8c443e0` | 001 |
| `rounds/decision_001.md` | `6e5741d2ccabe1883520625bfdb5a8e6e7f334b9ea995de5069943246342eceb` | 001 |
| `project.md` | (read) | 001 |
| `team-state.md` | (read) | 001 |
| `references/invariants.md` | `22b53f5f900c8062c445f35be52414b4abba99f8e4893a4dfab996eb1cd8d29c` | 001 |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | 001 |
