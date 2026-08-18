# Verifier Context State

- role_contract_sha256: `6bc3277527523950cab2c9295b07e6a5dc66030618c3b7e648c69a095a48f0a2`
- context_epoch: `1`
- last_completed_round: `000`
- accepted_kernel: `baseline_adapter.py`
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `Round 000 baseline on BI150; correctness PASS; reference median 0.474612 ms; baseline adapter median 0.474995 ms; profiler device time available.`
- open_hypotheses: `Keep reference and candidate profiler scopes separate. Use BI150 cat=kernel device durations and top-k breakdowns for candidate evaluation.`
- artifact_read_hashes: `base.py, baseline_adapter.py, project.md, team-state.md, report_000.md, round_status_000.md, and baseline trace recorded.`

## Current Bottleneck

- The BI150 profiler provides attributable CUDA `cat=kernel` events. The largest baseline contributors are top-k gather at 48.7290625 us/call and bitonic sort at 36.879697265625 us/call.

## Recent Three-round Evidence

- Round 000, baseline, `rounds/report_000.md`: 177.181318359375 device us/call for base and 179.0703515625 device us/call for the generated adapter; 14.8 and 14.96 kernels/call respectively.

## Open Hypotheses or Checks

- Correctness must precede all candidate timing.
- Future reports must retain separate `baseline_base` and candidate scopes and normalize profiler totals by 50 forward calls.
- Adoption remains controlled by unrounded wall-time median and the 5% threshold.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `log/groupedtopk_baseline_forward_50iter.pt.trace.json` | `0ed6dfa64748d1226baac93d0cd32ec4f16c0b64555b3f16022ef103efc77af` | 000 |
| `rounds/report_000.md` | `39a512eed23f1f0889e7845cde5f854cf0c2ca9d377ff23588148f239139f1e5` | 000 |
