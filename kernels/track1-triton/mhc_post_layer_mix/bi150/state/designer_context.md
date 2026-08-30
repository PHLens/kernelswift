# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: `2`
- last_completed_round: `001`
- accepted_kernel: `triton_mhc_post_layer_mix_001.py`
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `Round 000 baseline (wall 8.189 ms, device 7323.8 us, ratio 0.894); Round 001 elementwise-tail fusion accepted (wall 6.427 ms, +20.09%, device 6122.5 us, kernel_count 5.66->2.96)`
- open_hypotheses: `H-001 confirmed. Remaining GEMM gemm_tcu_h (5183 us/call, ~85% device) is memory-bound M=4/K=4 narrow batched GEMM; tl.dot tile (16/32) mismatch => no >=5% win; Round 002 abort.`
- artifact_read_hashes: `base.py`, `baseline_adapter.py`, `auto_bench.py`, `report_000.md`, `report_001.md` (see table)

## Current Bottleneck

- Verifier-backed: TCU batched GEMM `gemm_tcu_h` (`5183.49 us/call`, ≈85% of
  remaining device) for `[4,4]@[4,1280]` (M=4, K=4 contraction, N=1280). This is
  a memory-bound narrow batched GEMM; remaining device is the fused tail
  (`496.18 us`) and `residual.float()` cast (`442.86 us`). device_ratio ≈ 0.95.

## Recent Three-round Evidence

- Round 000 (Phase 0): `baseline` — wall `8.189047 ms`, device `7323.847 us`,
  `5.48 kernels/call`, ratio `0.894`.
- Round 001: `accepted` — elementwise-tail fusion (change_family
  `elementwise-fusion`), wall `8.043548 -> 6.427432 ms` (+20.09%), device
  `7516.836 -> 6122.542 us`, kernel_count `5.66 -> 2.96`. H-001 confirmed.

## Open Hypotheses or Checks

- H-001: confirmed (Round 001 accepted).
- GEMM rewrite via tl.dot: rejected in Round 002 (abort). tl.dot is Supported
  only for 16x16/32x32 tiles; M=4/K=4 padding wastes ~16x compute on a
  memory-bound GEMM and does not reduce ~250 MB traffic; expected to regress.
- residual.float() cast elimination: depends on unverified cublasLt bf16->fp32
  promotion equivalence; changes explicit reference semantics; at best ~6.9%,
  not a robust 5% win. Not pursued.
- No remaining falsifiable >=5% intervention identified.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `kernels/track1-triton/mhc_post_layer_mix/base.py` | `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3` | `0` |
| `kernels/track1-triton/mhc_post_layer_mix/bi150/baseline_adapter.py` | `66a3a2c31863d18c725a52ab57fd1b9f89fe655dd7bab7cb4da158b8130b5d07` | `0` |
| `kernels/track1-triton/mhc_post_layer_mix/bi150/triton_mhc_post_layer_mix_001.py` | `08a9d59f17ffa80224943b19bdcce390d908ca8ba15bf2e06ae469f45787d9fb` | `2` |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | `0` |
| `skills/kernel-opt-loop/prompts/designer.md` | `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef` | `1` |
| `rounds/report_000.md` | `<read at round 001>` | `1` |
| `rounds/report_001.md` | `<read at round 002>` | `2` |
