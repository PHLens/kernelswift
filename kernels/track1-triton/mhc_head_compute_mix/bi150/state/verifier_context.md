# Verifier Context State

- role_contract_sha256: `<not-computed>`
- context_epoch: `2`
- last_completed_round: `001`
- accepted_kernel: `kernels/track1-triton/mhc_head_compute_mix/bi150/baseline_adapter.py`
- accepted_report: `kernels/track1-triton/mhc_head_compute_mix/bi150/rounds/report_000.md`
- recent_three_round_evidence: `round 001 accepted: wall 1.433128 -> 0.183889 ms (87.17%), kernel 132.88 -> 1.0/call, device 924.79 -> 12.996 us/call, device_ratio 0.0755 (now host-bound)`
- open_hypotheses: `round 001 fusion accepted; remaining bottleneck is host-side per-call overhead (device_ratio ~0.075)`
- artifact_read_hashes: `base 4c5167f6..., adapter ceebdc61..., candidate a98b1b12..., harness 3d4fa4ee..., trace 961335d1...`

## Current Bottleneck

- After round 001 fusion, the operator is host-bound: device_ratio dropped to
  ~0.0755, so ~92.5% of wall time is host/launch/synchronization overhead
  (largely the harness `set_seed` + `sync_devices` per-iteration floor and the
  single kernel launch), not device math. The fused `_mhc_head_compute_mix_kernel`
  is ~12.996 us/call; further device-side fusion has limited headroom.

## Recent Three-round Evidence

- round 000, result `baseline`, evidence `rounds/report_000.md`, change family
  `baseline establishment (mechanical class rename)`.
- round 001, result `accepted`, evidence `rounds/report_001.md`, change family
  `kernel-fusion (Sinkhorn 20-round + head-compute elementwise into single Triton kernel)`.

## Open Hypotheses or Checks

- Candidate device_ratio is ~0.0755 (host-bound). A future round targeting the
  remaining host overhead (launcher/stream/synchronization reduction) would need
  a new validated decision; device-side fusion is largely exhausted.
- `tl.static_range(19)` unroll confirmed infeasible on CoreX/Triton 3.1.0 (>300s
  compile); dynamic `tl.range(19)` is the proven semantic-equivalent fallback.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `kernels/track1-triton/mhc_head_compute_mix/base.py` | `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5` | `001` |
| `kernels/track1-triton/mhc_head_compute_mix/bi150/baseline_adapter.py` | `ceebdc6185de4c980156a7833073678a0964fb7ccb5edf74b42be6156652eaed` | `001` |
| `kernels/track1-triton/mhc_head_compute_mix/bi150/triton_mhc_head_compute_mix_001.py` | `a98b1b12593d858ca29c787afa939a3ae0061df4ec6b51aa9a0fe7fa43c6b473` | `001` |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | `001` |
| `kernels/track1-triton/mhc_head_compute_mix/bi150/log/round_001_forward_50iter.pt.trace.json` | `961335d11c644fa987f3c32f1d1be9e0f170b633070eea4f6b357afa83492b94` | `001` |
