# Verifier Context State

- role_contract_sha256: `6bc3277527523950cab2c9295b07e6a5dc66030618c3b7e648c69a095a48f0a2`
- context_epoch: `2`
- last_completed_round: `003`
- accepted_kernel: `triton_grouped_topk_003.py`
- accepted_report: `rounds/report_003.md`
- recent_three_round_evidence: `Round 003 accepted. Wall median 0.292588 ms -> 0.273673 ms; correctness, metadata hit/miss/invalidation, lifecycle, and concurrency PASS; runtime launches remain 1.0/call.`
- open_hypotheses: `GCU device duration is unavailable in PrivateUse1 trace; runtime launch evidence remains diagnostic. Launcher/context specialization is unmeasured until paired verification.`
- artifact_read_hashes: `Round 003 trace, summary outputs, metadata/lifecycle command, formal samples, and report are recorded.`

## Current Bottleneck

- The GCU profiler exporter exposes `gcu_runtime` launch events but no
  `cat=kernel` device-duration events. The accepted candidate has one
  `topsModuleLaunchKernel` event per call in the recorded scope.

## Recent Three-round Evidence

- Round 001, accepted, `rounds/report_001.md`: wall improvement
  `39.08693002628853%`; runtime launches `12.0 -> 1.0` per call.
- Round 002, accepted, `rounds/report_002.md`: wall improvement
  `9.02136875254568%`; runtime launches remain `1.0/call`; lifecycle PASS.
- Round 003, accepted, `rounds/report_003.md`: wall improvement
  `6.464721724746064%`; runtime launches remain `1.0/call`; metadata and lifecycle PASS.

## Open Hypotheses or Checks

- Keep `device_time_available=false` for this exporter until a matched TOPS/
  TOPSPTI device-duration path is established.
- Preserve separate reference and candidate scopes; use wall time as adoption
  metric and runtime launches only as causal diagnostics.
- Any launcher/context optimization must prove no stream/device/synchronization
  change and retain all output-pool guardrails.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `log/groupedtopk_round_003_forward_50iter.pt.trace.json` | `dc1b1ac9b8dbf8d21b52804dc540c116dd48a3560c347886ca3780ccd8c4af34` | 003 |
| `rounds/report_003.md` | 71861c3 | 003 |
| `rounds/coder_result_003.md` | 71861c3 | 003 |
