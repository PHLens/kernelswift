# Verifier Context State

- role_contract_sha256: `6bc3277527523950cab2c9295b07e6a5dc66030618c3b7e648c69a095a48f0a2`
- context_epoch: `1`
- last_completed_round: `001`
- accepted_kernel: `triton_grouped_topk_001.py`
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `Round 000 baseline and Round 001 accepted. Wall median 0.449626 ms -> 0.273881 ms; correctness PASS; GCU runtime launches 12.0 -> 1.0 per call.`
- open_hypotheses: `GCU device duration is unavailable in PrivateUse1 trace; runtime launch evidence is retained. Future reports must keep separate scopes and avoid device_ratio claims.`
- artifact_read_hashes: `baseline and Round 001 traces, summary outputs, commands, and reports are recorded.`

## Current Bottleneck

- The GCU profiler exporter exposes `gcu_runtime` launch events but no
  `cat=kernel` device-duration events. The accepted candidate has one
  `topsModuleLaunchKernel` event per call in the recorded scope.

## Recent Three-round Evidence

- Round 000, baseline, `rounds/report_000.md`: 12 runtime launches/call; device
  duration unavailable.
- Round 001, accepted, `rounds/report_001.md`: 1 runtime launch/call, 10.409482
  runtime-launch us/call, wall median 0.273881 ms.

## Open Hypotheses or Checks

- Keep `device_time_available=false` for this exporter until a matched TOPS/
  TOPSPTI device-duration path is established.
- Use wall time as the adoption metric; runtime launches are causal diagnostics.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `log/groupedtopk_baseline_forward_50iter.pt.trace.json` | `cfea5cd92a62d2eee78db6a2f801212f1920723121f21793dd9724c0194952a2` | 000 |
| `log/groupedtopk_round_001_forward_50iter.pt.trace.json` | `d2eb35974a9617b3e114397b54548883a822cbbabd6886a58f6c7955469e9ce6` | 001 |
| `rounds/report_001.md` | pending final commit | 001 |
