# Coder Context State

- role_contract_sha256: `<not-computed>`
- context_epoch: 1
- last_completed_round: `002`
- accepted_kernel: `null`
- accepted_report: `null`
- recent_three_round_evidence: `Round 002 candidate-ready: remove-transpose-copy (kernel loads q/k/v from original [bsz,seq,hidden] layout via stride, dropping 4 transpose12_copy_64 kernels; wall ~0.135 ms vs round 001 ~0.175 ms).`
- open_hypotheses: `await Verifier authoritative runtime evidence for triton_mha_002.py (expect kernel count 5->1, device ~79.7->67.1 us/call).`
- artifact_read_hashes: see table below

## Current Bottleneck

- `Round 002 candidate produced; awaiting Verifier measurement. Dominant device cost remains the fused _mha_fwd_kernel (~67 us/call) with manual tl.sum dot; deeper single-pass softmax / tl.dot are deferred (tl.dot Unknown).`

## Recent Three-round Evidence

- `Round 002: remove-transpose-copy, result candidate-ready, evidence rounds/coder_result_002.md, change family remove-transpose-copy.`
- `Round 001: fused-mha-kernel, result candidate-ready, evidence rounds/coder_result_001.md, change family fused-mha-kernel.`

## Open Hypotheses or Checks

- `Verify candidate kernel count per call = 1.0 (transpose12_copy_64 4.0 -> 0.0, _mha_fwd_kernel stays 1.0).`
- `Verify device_us_per_call drops from ~79.7 to ~67.1 us/call.`

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | 000 |
| `baseline_adapter.py` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | 001 |
| `triton_mha_001.py` | `9fac12aa0298a970c208dbc6af7a602da4f34e43d44921b62aad571ca662c00b` | 002 |
| `rounds/decision_002.md` | `be804f497dcb6070e1a07d290b43c6c8acc65e3007d88657985026aa5640ac7e` | 002 |
| `triton_mha_002.py` | `29e6b192bf778f0264fb7657c9a33b97819c406896a2ad86e1daf22f3c9ff0a1` | 002 |
