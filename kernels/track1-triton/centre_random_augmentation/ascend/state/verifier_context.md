# Verifier Context State

- role_contract_sha256: `f9d06fdf3ddbb18944568412f7d86d88266245f8dfa974a2ab3cf282f37bbd27`
- context_epoch: 2
- last_completed_round: `001`
- accepted_kernel: `triton_centre_random_aug_001.py` (Orchestrator-owned canonical pointer; verifier reports only)
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `001: accepted 17.84% wall (2.463->2.024ms), H-001 partially-confirmed, device 294.97->216.06us, kernel_count 110->64; 000: baseline 2.55ms, device 291.9us, kernel_count 110`
- open_hypotheses: `host-bound launch overhead remains; kernel_count still 64; RNG-order correctness constraint persists for all future candidates`
- artifact_read_hashes: `see table below`

## Current Bottleneck

- `Wall time remains host-launch-bound (device_ratio ~0.107 after fusion). kernel_count_per_call=64 (not <=25); remaining torch R/T + quaternion Sin/Cos/Sqrt path + contiguous/empty/host-transfer launches dominate launch count.`

## Recent Three-round Evidence

- `001 (accepted): fuse deterministic linear tail into one Triton kernel; wall 17.84% faster; device -26.8%; kernel_count 110->64. H-001 partially-confirmed.`
- `000 (baseline): correctness PASS; base.py 2.547680ms vs baseline_adapter 2.565115ms; device 291.9us/call, kernel_count 110.`

## Open Hypotheses or Checks

- `RNG-order hazard: future candidates must preserve 3x torch.rand(4) + 1x torch.randn(4,3) draw order bitwise, else allclose gate fails (O(1) divergence).`
- `Fused kernel under-parallelized (BLOCK=256, 4 programs, num_warps=4 over 1024 rows) vs 20 cube/40 vector cores.`

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` | `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553` | 001 |
| `baseline_adapter.py` | `7d4a79ae96328fc03a4489710f68b7f639ddea9cbd5c0f7bb45e1cec5472061b` | 001 |
| `triton_centre_random_aug_001.py` | `dcfeb039d3d8526d756775015560a22e1b0cd447c5c6dbd69ad12d3a3f0ee089` | 001 |
| `project.md` | `<orchestrator-owned>` | 001 |
