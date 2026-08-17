# Coder Result 002

Result: `candidate-ready`

- round: `002`
- source_canonical: `triton_grouped_topk_001.py`
- source_canonical_sha256: `f42ff6b47b28996199bbe9b8df0a181db2834be99473453f3eea35df51df693e`
- reference_adapter: `reference_triton_grouped_topk_001.py`
- reference_adapter_sha256: `800ec0080e66589f6dfcf3a71ee79f08e01be68f145b4cb3c6c6b50dd7c03027`
- reference_adapter_contract: `byte-equivalent to the canonical source except ModelNew -> Model for the unchanged harness v0 loader`
- decision: `rounds/decision_002.md`
- decision_sha256: `8d56aaf1e9ca91f59a439e3ace0bba74d0234b7f02f4a3712f592100884f0805`
- candidate: `triton_grouped_topk_002.py`
- candidate_sha256: `90d7b09569d1d155c8e44e1626f2c0f3b3f41e0919a8a9e5b76719e874b17ce3`
- selected_profile: `triton_gcu`
- runtime_fingerprint: `project.md#runtime-fingerprint`
- measurement_fingerprint: `3942e25aebbe7690a55cf27768a3bc3fd552cc8106f6bd2dd7416cea2d274bf3`

## Implementation Conformance

- The accepted Triton kernel body, grid, constexprs, direct launch, and
  `num_warps=1` are unchanged from Round 001.
- Output pooling is private to each `ModelNew` instance and keyed by GCU device,
  current GCU stream id, token count, topk, dtypes, and contiguous output strides.
- Pool metadata is protected by a model-local `threading.Lock`; concurrent calls
  reserve distinct entries and never serialize by waiting for a live entry.
- GCU `_storage_Use_Count` and Python reference counts are used only as a
  conservative reuse check. If stream identity or storage count is unavailable,
  the candidate allocates fresh outputs instead of reusing storage.
- No explicit synchronization, device-context switch, stream switch, cross-stream
  wait, or extra Triton launch was added.

## Guardrail Evidence

| Check | Result | Evidence |
|---|---|---|
| sequential compatible calls reuse storage | PASS | S60 lifecycle command: `sequential_reuse True`, pool size 1 |
| retained returned outputs stay stable | PASS | S60 lifecycle command: `retained_output_distinct True`, `retained_stable True` |
| retained view alias stays stable | PASS | S60 lifecycle command: `alias_distinct True`, `alias_stable True` |
| concurrent same-instance calls use distinct storage | PASS | S60 lifecycle command: `concurrent_distinct True`, pool size 2 |
| correctness against accepted reference | PASS | S60 auto_bench smoke and three formal paired runs |

## Attempt Ledger

| Attempt | Command / change | Exit status | Defect | Candidate before SHA256 | Candidate after SHA256 |
|---:|---|---:|---|---|---|
| 1 | Initial lease thresholds used `sys.getrefcount(...) > 3` | 0 | lifecycle check exposed retained-output and concurrent storage reuse | `36abc6656221a61414b85a6eb838efbd9519205aba2814f1a8eff9138229ff58` | `36abc6656221a61414b85a6eb838efbd9519205aba2814f1a8eff9138229ff58` |
| 2 | Calibrated S60 pool reference baseline and changed threshold to `> 2` | 0 | none; all lifecycle guardrails passed | `36abc6656221a61414b85a6eb838efbd9519205aba2814f1a8eff9138229ff58` | `90d7b09569d1d155c8e44e1626f2c0f3b3f41e0919a8a9e5b76719e874b17ce3` |
| 3 | `python3 -m py_compile s60/groupedtopk/triton_grouped_topk_002.py` | 0 | none | `90d7b09569d1d155c8e44e1626f2c0f3b3f41e0919a8a9e5b76719e874b17ce3` | `90d7b09569d1d155c8e44e1626f2c0f3b3f41e0919a8a9e5b76719e874b17ce3` |

No unresolved semantic or runtime defect remains for the recorded S60
fingerprint.
