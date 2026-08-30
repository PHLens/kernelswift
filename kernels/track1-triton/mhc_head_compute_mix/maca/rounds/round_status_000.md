# Round Status 000

- Round: `000`
- Phase: `verification complete` (baseline)
- Result: `baseline`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Candidate SHA256: `c3cc90deb93c69f092e876ddc0057ddf6c2d7ac28020f1b7fd1ed260bca72fee`
- Accepted reference: `base.py`
- Accepted reference SHA256: `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Measurement fingerprint: `d8f4b63bfbf09ce8a32f3bdcd4d85553f34abce7384e495ba5f66baf49bf795e`

## Progress Log

| Step | Phase | Completed | Command result | Next safe action |
|---|---|---|---|---|
| identity check | verification start | yes | sha256sum of base.py/auto_bench.py/baseline_adapter.py all match declared values | run correctness |
| correctness | after correctness | yes | exit 0; `PASS accuracy; v0=1.541432 ms, v1=1.529807 ms, speedup=1.008x` | run authoritative wall timing |
| authoritative wall run 1 | timing | yes | exit 0; v0=1.512683 ms, v1=1.509571 ms | run run 2 |
| authoritative wall run 2 | timing | yes | exit 0; v0=1.522637 ms, v1=1.515187 ms | run run 3 |
| authoritative wall run 3 | timing | yes | exit 0; v0=1.528345 ms, v1=1.522095 ms | compute medians, run profiler |
| profiler | profiling | yes | exit 0; trace exported to `log/round_000_forward_50iter.pt.trace.json` | summarize scopes |
| summarize (raw) | profiling | yes | both scopes errored: `overlapping scope events` (C500 duplicate nested gpu_user_annotation) | filter duplicate markers |
| summarize (filtered) | profiling | yes | both scopes summarized; see report_000.md | write report |
| report + status | verification end | yes | `report_000.md` and `round_status_000.md` written | Orchestrator transition |

## Wall Timing Raw Samples (authoritative, unrounded medians per run)

- warmup `50`, repeat `100`
- reference (v0) samples ms: `[1.512683, 1.522637, 1.528345]`
- candidate (v1) samples ms: `[1.509571, 1.515187, 1.522095]`
- reference median ms: `1.522637`
- candidate median ms: `1.515187`

## Profiler Summary (filtered trace, iterations 50)

- baseline_base: device_total_us 26734.246, device_us_per_call 534.685, kernel_count_total 6650, kernel_count_per_call 133.0, device_ratio 0.351157
- candidate_baseline_adapter: device_total_us 26881.164, device_us_per_call 537.623, kernel_count_total 6650, kernel_count_per_call 133.0, device_ratio 0.354823

## Artifacts Written

- `rounds/report_000.md`
- `rounds/round_status_000.md`
- `state/verifier_context.md` (updated)
- `log/round_000_forward_50iter.pt.trace.json` (raw, preserved)
- `log/round_000_forward_50iter.filtered.pt.trace.json` (duplicate nested gpu_user_annotation scope markers removed)

## Classification

- Result: `baseline` (Phase 0). No adoption decision; Orchestrator applies the canonical
  pointer transition to `baseline_adapter.py`.
