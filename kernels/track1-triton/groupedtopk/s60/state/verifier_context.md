# Verifier Context State

- role_contract_sha256: `6bc3277527523950cab2c9295b07e6a5dc66030618c3b7e648c69a095a48f0a2`
- context_epoch: `2`
- last_completed_round: `004`
- accepted_kernel: `triton_grouped_topk_003.py`
- accepted_report: `rounds/report_003.md`
- recent_three_round_evidence: `Round 004 valid no-improvement: correctness and all cache/lifecycle guardrails PASS; wall median 0.277370 ms -> 0.271659 ms, 2.058982586436897%; canonical remains Round 003.`
- open_hypotheses: `GCU device duration remains unavailable. Candidate-owned stream lookup was reduced to one explicit snapshot, but backend internal lookup remains; two valid no-improvement rounds remain before stop.`
- artifact_read_hashes: `Round 004 trace, stack trace, summary outputs, lifecycle command, formal samples, and report are recorded.`

## Current Bottleneck

- The GCU profiler exporter exposes `gcu_runtime` launch events but no
  `cat=kernel` device-duration events. Both accepted reference and Round 004
  candidate emit one `topsModuleLaunchKernel` per call.

## Recent Three-round Evidence

- Round 002, accepted, `rounds/report_002.md`: `9.02136875254568%` wall improvement.
- Round 003, accepted, `rounds/report_003.md`: `6.464721724746064%` wall improvement.
- Round 004, no-improvement, `rounds/report_004.md`: `2.058982586436897%` wall improvement; runtime launch `1.0/call`; device duration unavailable.

## Open Hypotheses or Checks

- Keep `device_time_available=false` for this exporter until a matched TOPS/
  TOPSPTI device-duration path is established.
- Preserve separate reference and candidate scopes and never relabel runtime
  launch time as device duration.
- Continue only with bounded host evidence; stop after two more valid
  no-improvement rounds unless another policy stop occurs.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `log/groupedtopk_round_004_forward_50iter.pt.trace.json` | `ba3ddc328cd2cb36b06cd4401c2db423994f5be5dba29ee245766dff3bc609db` | 004 |
| `rounds/report_004.md` | 5ded926 | 004 |
| `rounds/coder_result_004.md` | 5ded926 | 004 |
