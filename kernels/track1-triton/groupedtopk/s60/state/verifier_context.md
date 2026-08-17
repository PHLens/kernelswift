# Verifier Context State

- role_contract_sha256: `6bc3277527523950cab2c9295b07e6a5dc66030618c3b7e648c69a095a48f0a2`
- context_epoch: `2`
- last_completed_round: `002`
- accepted_kernel: `triton_grouped_topk_002.py`
- accepted_report: `rounds/report_002.md`
- recent_three_round_evidence: `Round 001 and Round 002 accepted. Round 002 wall median 0.301983 ms -> 0.274740 ms; correctness and all lifecycle guardrails PASS; runtime launches remain 1.0/call.`
- open_hypotheses: `GCU device duration is unavailable in PrivateUse1 trace; runtime launch evidence remains diagnostic. Next host metadata hypothesis is unmeasured until paired verification.`
- artifact_read_hashes: `Round 002 trace, summary outputs, lifecycle command, formal samples, and report are recorded.`

## Current Bottleneck

- The GCU profiler exporter exposes `gcu_runtime` launch events but no
  `cat=kernel` device-duration events. The accepted candidate has one
  `topsModuleLaunchKernel` event per call in the recorded scope.

## Recent Three-round Evidence

- Round 000, baseline, `rounds/report_000.md`: 12 runtime launches/call; device
  duration unavailable.
- Round 001, accepted, `rounds/report_001.md`: 1 runtime launch/call, 10.409482
  runtime-launch us/call, wall median 0.273881 ms.
- Round 002, accepted, `rounds/report_002.md`: 1 runtime launch/call, 10.782412109375
  runtime-launch us/call, wall median 0.274740 ms against 0.301983 ms reference.

## Open Hypotheses or Checks

- Keep `device_time_available=false` for this exporter until a matched TOPS/
  TOPSPTI device-duration path is established.
- Preserve separate reference and candidate scopes; use wall time as adoption
  metric and runtime launches only as causal diagnostics.
- For future host caches, require output lifetime, alias, device, stream, and
  concurrency guardrails before timing adoption.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `log/groupedtopk_round_002_forward_50iter.pt.trace.json` | `c01d6f966b94e5476d77ad008df9c6d2e5072702d7ffbcce17a9f9c348f68a62` | 002 |
| `rounds/report_002.md` | 7d694c3 | 002 |
| `rounds/coder_result_002.md` | 7d694c3 | 002 |
