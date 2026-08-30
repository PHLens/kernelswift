# Verifier Context State

- role_contract_sha256: `<to-fill>`
- context_epoch: `1`
- last_completed_round: `001`
- accepted_kernel: `kernels/track1-triton/mhc_post_layer_mix/bi150/triton_mhc_post_layer_mix_001.py`
- accepted_report: `kernels/track1-triton/mhc_post_layer_mix/bi150/rounds/report_001.md`
- recent_three_round_evidence: `<Round 001 accepted: elementwise-fusion, wall 8.043548->6.427432 ms (20.09%), kernel_count 5.66->2.96>`
- open_hypotheses: `<GEMM (term2) rewrite remains the dominant bottleneck; tl.dot Unknown on triton_cuda>`
- artifact_read_hashes: `base.py`, `baseline_adapter.py`, `triton_mhc_post_layer_mix_001.py`, `auto_bench.py`, `decision_001.md` (see table)

## Current Bottleneck

- Round 001 fused the post-GEMM tail; wall `8.043548 → 6.427432 ms` (20.09%),
  device `7516.836 → 6122.542 us/call` (−18.5%), kernel_count `5.66 → 2.96`.
- Remaining dominant cost: unchanged TCU batched GEMM `gemm_tcu_h`
  (`~5183.49 us/call`, ≈85% of remaining device time), plus `residual.float()`
  cast (`~442.86 us/call`) and the fused tail (`~496.18 us/call`).

## Recent Three-round Evidence

- Round 000 (Phase 0): baseline `8.189047 ms`; einsum → TCU batched GEMM +
  4 elementwise/cast kernels (5.48/call).
- Round 001: elementwise-fusion accepted; kernel_count 2.96/call; H-001 confirmed.

## Open Hypotheses or Checks

- GEMM (`[4,4]@[4,1280]`, contraction dim 4) rewrite via `tl.dot` is the next
  candidate bottleneck but is a capability risk (Unknown on triton_cuda profile).
- Note: candidate Triton direct launch produced nested same-name
  `record_function` markers, making the unmodified summarizer report
  `overlapping scope events` for the candidate scope; future candidates with
  Triton launches may need scope-marker awareness.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `kernels/track1-triton/mhc_post_layer_mix/base.py` | `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3` | `1` |
| `kernels/track1-triton/mhc_post_layer_mix/bi150/baseline_adapter.py` | `66a3a2c31863d18c725a52ab57fd1b9f89fe655dd7bab7cb4da158b8130b5d07` | `1` |
| `kernels/track1-triton/mhc_post_layer_mix/bi150/triton_mhc_post_layer_mix_001.py` | `08a9d59f17ffa80224943b19bdcce390d908ca8ba15bf2e06ae469f45787d9fb` | `1` |
| `kernels/track1-triton/mhc_post_layer_mix/bi150/rounds/decision_001.md` | `335389df2498f37fb9f2c5c7ebc10986ab4edf555d939525413900e0e885ecfc` | `1` |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | `1` |
