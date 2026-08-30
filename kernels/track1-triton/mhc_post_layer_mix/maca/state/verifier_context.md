# Verifier Context State

- role_contract_sha256: `<not-computed>`
- context_epoch: 1
- last_completed_round: `001`
- accepted_kernel: `maca/triton_mhc_001.py` (Orchestrator owns canonical pointer)
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `R000 baseline (wall 7.64ms, tf32 GEMM 6071us). R001 accepted: fused K=4 Triton kernel -> wall 0.241ms, device 168.56us/call, 1.0 kernel/call, tf32 GEMM eliminated, +96.84%.`
- open_hypotheses: `Round 002: candidate is now partially host/launch-bound (device_ratio 0.699); remaining 168.56us device in single fused kernel.`
- artifact_read_hashes: `base.py e392…ebf3; auto_bench.py 3d4f…5bf2; baseline_adapter.py 2c0c…4e6; triton_mhc_001.py e54e…b944`

## Current Bottleneck

- `After R001: single fused kernel at 168.56 us/call (was 7560.89 us). candidate device_ratio dropped to 0.699, so ~30% of the 0.241ms wall is now host/launch overhead rather than device work.`

## Recent Three-round Evidence

- `R000 (baseline): wall 7.635598/7.636740 ms (v0/v1); device 7559 us/call; tf32 GEMM mcblas__Mck_tf32gemm 6071 us/call (~80%).`
- `R001 (accepted): fused K=4 kernel `_mhc_post_layer_mix_fused_kernel`; wall 0.241083 ms; device 168.56 us/call; 1.0 kernel/call; tf32 GEMM gone; +96.84% wall.`

## Open Hypotheses or Checks

- `R002 candidate direction: launch-overhead reduction / vectorization of the single fused kernel; tf32 GEMM bottleneck is already removed.`

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` | `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3` | 001 |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | 001 |
| `baseline_adapter.py` | `2c0c121e2e4504e791fee3675ac1cd54d1322059fc537555578a91cabd2a24e6` | 001 |
| `triton_mhc_001.py` | `e54e5b2e553449134eb3b6679d6ed759e30fd2dd42499f9a21716ae57216b944` | 001 |
