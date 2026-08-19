# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: 3
- last_completed_round: "002"
- accepted_kernel: `triton_sparse_pooler_001.py`
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `Round 000 baseline (wall 0.935560 ms, device 374.81 us/call). Round 001 kernel-fusion accepted (+33.78%): wall 0.618775 ms, device ~202 us/call (5 kernels). Round 002 allocation-reuse no-improvement (+2.75%): NPU caching allocator makes output alloc near-free.`
- open_hypotheses: `<none: campaign aborted at Round 003 — no falsifiable >=5% intervention remains>`
- artifact_read_hashes: `<see Artifact Read Hashes table>`

## Current Bottleneck

- Mixed (device_ratio ~0.30). Remaining device = aclnnAddmm x2 (~135-147 us, ~76% device, MLM head matmuls, library-optimal; tl.dot fusion has double negative evidence). Remaining host (~70% wall) = fixed launch/dispatch + harness sync (output alloc ruled out in R2).

## Recent Three-round Evidence

- Round 000 (baseline): wall 0.935560 ms, device 374.81 us/call, 14 kernels.
- Round 001 (kernel-fusion, accepted +33.78%): wall 0.618775 ms, device ~202 us/call, 5 kernels.
- Round 002 (allocation-reuse, no-improvement +2.75%): output alloc near-free on NPU caching allocator.

## Open Hypotheses or Checks

- `<none>`. Round 003 decision is `abort`: every lever exhausted (matmul=tl.dot double-negative; tile-tuning falsified MLU R2; fast_libentry Unknown; output alloc falsified R2; remaining host fixed). Final cumulative result: +33.78% (canonical triton_sparse_pooler_001.py).

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` | `2b740bba37a87a7bcb022af36537486179538feed5dada3f3c1d5e32cd3f6c36` | 000 |
| `baseline_adapter.py` | `94d00f1a5d26f453fd5078fd9d50dfcddbb0c11d20a145d223544e59234add0f` | 001 |
| `triton_sparse_pooler_001.py` | `<read at round 002>` | 002 |
| `rounds/report_001.md` | `<read at round 002>` | 002 |
| `rounds/report_002.md` | `<read at round 003>` | 003 |
| `skills/kernel-opt-loop/prompts/designer.md` | `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef` | 003 |
| `../../../../../auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 003 |
