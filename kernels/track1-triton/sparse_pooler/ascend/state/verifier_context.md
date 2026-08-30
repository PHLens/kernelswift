# Verifier Context State

- role_contract_sha256: `f9d06fdf3ddbb18944568412f7d86d88266245f8dfa974a2ab3cf282f37bbd27`
- context_epoch: 3
- last_completed_round: 002
- accepted_kernel: `triton_sparse_pooler_001.py`
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `<round 000 baseline; round 001 accepted +33.78%; round 002 no-improvement +2.75%>`
- open_hypotheses: `<matmul fusion / host launch-dispatch overhead as next bottleneck>`
- artifact_read_hashes: `<see table>`

## Current Bottleneck

- Accepted canonical wall `0.618775 ms` (triton_sparse_pooler_001), device `~194-203 us/call`, 5 kernels/call.
- Dominant device bottleneck: `aclnnAddmm_MatMulCommon_MatMulV2` x2 (~135-147 us/call, ~74-76% of device) — MLM-head dense + decoder matmuls.
- device_ratio ≈ 0.30 → ~70% of wall is host-side, but NOT output allocation (Round 002 proved allocation-reuse gives only +2.75%); remaining host time is launch/dispatch + harness-fixed (sync_devices, seed).

## Recent Three-round Evidence

- Round 000 (Phase 0 baseline): wall 0.935560 ms; device 374.81 us/call; 14 kernels/call.
- Round 001 (accepted +33.78%): fused relu+log1p+max pool → one kernel; wall 0.618775 ms; device 202.86 us/call; 5 kernels/call.
- Round 002 (no-improvement +2.75%): output-buffer reuse (host-only); wall 0.619190 ms (sub-threshold); kernel/device unchanged (5 kernels, byte-identical kernel). Output allocation is near-free (NPU caching allocator), falsifying H-002.

## Open Hypotheses or Checks

- Matmul fusion (deferred in decision_001; MLU + flexattention-ascend both showed tl.dot regression) — next device bottleneck candidate, requires its own evidence.
- Host launch/dispatch overhead (~70% of wall) — but output allocation already ruled out; remaining host time is largely harness-fixed, not compressible.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` | `2b740bba37a87a7bcb022af36537486179538feed5dada3f3c1d5e32cd3f6c36` | 000 |
| `baseline_adapter.py` | `94d00f1a5d26f453fd5078fd9d50dfcddbb0c11d20a145d223544e59234add0f` | 001 |
| `triton_sparse_pooler_001.py` | `dc2a8b6582cf9d6fef3e044081426762b88833e056ce8d7f04086e0d92f429e0` | 002 |
| `triton_sparse_pooler_002.py` | `a7338d89a1f5a30843e84d3f533ac151245d6547453ddc5a2dcff66f93cb7957` | 002 |
| `auto_bench.py` (harness) | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 000 |
| `decision_002.md` | `8beb3c34e9eb88aac1722ee9a99117a7c05453e2f08ed5487ccd49a5004b003f` | 002 |
