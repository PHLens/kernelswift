# Verifier Context State

- role_contract_sha256: `6bc3277527523950cab2c9295b07e6a5dc66030618c3b7e648c69a095a48f0a2`
- context_epoch: `2`
- last_completed_round: `007`
- accepted_kernel: `triton_grouped_topk_003.py`
- accepted_report: `rounds/report_003.md`
- recent_three_round_evidence: `Round 008 named measurement probe completed without a candidate result: correctness passed for accepted canonical against its reference adapter at 0.282032 ms versus 0.282114 ms in one recorded pair; both scoped profile views retained 1.0 runtime launch/call and exposed no cat=kernel device-duration events. Round 007 aborted because no candidate-owned >=5% path was justified after environment access was restored. Round 004 valid no-improvement remained 2.058982586436897% below threshold.`
- open_hypotheses: `The matched S60 probe confirms execution and the unchanged one-launch path but does not identify a compressible device or candidate-owned host component. GCU device duration remains unavailable; any next proceeding decision needs a distinct, defensible host mechanism or a matched device-time/microbenchmark observation.`
- artifact_read_hashes: `Round 008 status and trace are recorded; remote benchmark/profile completed before recovery, trace was copied without rerunning measurement, and local scope summaries preserve runtime-launch-only classification.`

## Current Bottleneck

- The GCU profiler exporter exposes `gcu_runtime` launch events but no
  `cat=kernel` device-duration events. In the Round 008 probe, both the accepted
  reference and canonical emitted one `topsModuleLaunchKernel` per call.
- The matched one-pair wall result was reference `0.282114 ms` and canonical
  `0.282032 ms`; this is measurement-only evidence and not an optimization result.

## Recent Three-round Evidence

- Round 008, named measurement probe, `rounds/round_status_008.md`: correctness PASS; reference/canonical raw pair `0.282114/0.282032 ms`; runtime launches `1.0/call` in both scopes; device duration unavailable.
- Round 007, aborted, `rounds/decision_007.md`: no candidate-owned >=5% path after SSH execution access was restored.
- Round 004, no-improvement, `rounds/report_004.md`: `2.058982586436897%` wall improvement; runtime launch `1.0/call`; device duration unavailable.

## Open Hypotheses or Checks

- Keep `device_time_available=false` for this exporter until a matched TOPS/
  TOPSPTI device-duration path is established.
- Preserve separate reference and candidate scopes and never relabel runtime
  launch time as device duration.
- Route the probe evidence to Designer for a distinct candidate decision; do not
  treat the probe as accepted/no-improvement and do not repeat the failed
  launcher-context family without new causal evidence.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `rounds/round_status_008.md` | `1c04a827a50cbb065c1c9943e7c0f5ddf961aeca7f27c06aa2e912f5d2b1a7ec` | 008 |
| `log/groupedtopk_probe_round008_forward_50iter.pt.trace.json` | `1c04a827a50cbb065c1c9943e7c0f5ddf961aeca7f27c06aa2e912f5d2b1a7ec` | 008 |
| `rounds/report_004.md` | `5ded926` | 004 |
| `rounds/coder_result_004.md` | `5ded926` | 004 |
