# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: 3
- last_completed_round: 001
- accepted_kernel: `triton_centre_random_aug_001.py`
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `<round 000 baseline: host-bound, device_ratio 0.114, 110 kernels/call, wall ~2.548 ms> | <round 001 accepted +17.84%: elementwise-launch-fusion, 110->64 kernels, device 294.97->216.06 us, wall 2.463270->2.023920 ms, device_ratio 0.107>`
- open_hypotheses: `<none — round 002 aborted at host-bound-floor>`
- artifact_read_hashes: `<see table below>`

## Current Bottleneck

- Verifier-backed (report_001): host-bound floor reached. wall 2.023920 ms; device_us_per_call 216.06 us; device_ratio ~0.107 (<20%). The fused `_centre_aug_linear_kernel` is 56% of device (~122 us/call) but only ~6% of wall. kernel_count_per_call = 64, dominated by torch R/T + quaternion Sin/Cos/Sqrt path (bitwise-RNG constraint) plus `contiguous()`/`empty`/host-transfer launches.

## Recent Three-round Evidence

- Round 000 (baseline): host-bound; 110 tiny kernels; device_ratio 0.114; wall ~2.548 ms; randomness gate confirmed (per-call set_seed(42) + allclose(1e-2,1e-2)).
- Round 001 (accepted): elementwise-launch-fusion fused centering + 3x3 matvec + translation + mask into one Triton kernel; kernel_count 110 -> 64, device -26.8%, wall +17.84%. H-001 partially-confirmed (did not reach <=25 because RNG + quaternion transcendentals stay in torch).

## Open Hypotheses or Checks

- None. Round 002 decision is `abort` (change_family `no-change`, stop_reason `host-bound-floor`): no falsifiable >=5% wall intervention remains. Device is 10.7% of wall (sub-threshold for further tuning); the only remaining launch reduction (~63 torch launches) is blocked by the bitwise-RNG-order correctness invariant. Matches the campaign-wide host-bound-floor termination of all 8 prior sibling operators.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` | `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553` | 000 |
| `baseline_adapter.py` | `7d4a79ae96328fc03a4489710f68b7f639ddea9cbd5c0f7bb45e1cec5472061b` | 000 |
| `triton_centre_random_aug_001.py` | `dcfeb039d3d8526d756775015560a22e1b0cd447c5c6dbd69ad12d3a3f0ee089` | 002 |
| `project.md` | `<orchestrator-owned>` | 000 |
| `team-state.md` | `<orchestrator-owned>` | 002 |
