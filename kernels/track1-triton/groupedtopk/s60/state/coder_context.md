# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: `2`
- last_completed_round: `002`
- accepted_kernel: `triton_grouped_topk_002.py`
- accepted_report: `rounds/report_002.md`
- recent_three_round_evidence: `Round 002 accepted; safe per-instance output pool passed retained-output, alias, and concurrency checks; wall median 0.301983 ms -> 0.274740 ms.`
- open_hypotheses: `Host metadata specialization is next, but it is only a source-backed hypothesis. Kernel dataflow changes require matched GCU device evidence or a microbenchmark.`
- artifact_read_hashes: `decision_002.md, candidate source, reference adapter, coder_result_002.md, project.md, and team-state.md recorded in the Round 002 ledger.`

## Current Bottleneck

- The accepted candidate has one direct Triton-GCU launch per call and safe
  output storage reuse. GCU device duration remains unavailable from the
  recorded profiler exporter.

## Recent Three-round Evidence

- Round 001, accepted, `rounds/report_001.md`, `kernel-fusion`: wall median
  `0.449626 -> 0.273881 ms`, runtime launches `12.0 -> 1.0` per call.
- Round 002, accepted, `rounds/report_002.md`, `allocation-reuse`: wall median
  `0.301983 -> 0.274740 ms`, runtime launches remain `1.0` per call; all
  lifecycle guardrails pass.

## Open Hypotheses or Checks

- Host metadata specialization must preserve shape/device/stream semantics and
  be validated with paired wall timing before adoption.
- Any future GCU kernel change must keep device-time claims unavailable unless a
  matched exporter provides attributable device durations.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `rounds/decision_002.md` | `8d56aaf1e9ca91f59a439e3ace0bba74d0234b7f02f4a3712f592100884f0805` | 002 |
| `reference_triton_grouped_topk_001.py` | `800ec0080e66589f6dfcf3a71ee79f08e01be68f145b4cb3c6c6b50dd7c03027` | 002 |
| `triton_grouped_topk_002.py` | `90d7b09569d1d155c8e44e1626f2c0f3b3f41e0919a8a9e5b76719e874b17ce3` | 002 |
| `rounds/coder_result_002.md` | 7d694c3 | 002 |
