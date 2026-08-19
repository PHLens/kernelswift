# Round Status 001

- Round: `001`
- Phase: `verification complete` (authoritative)
- Result: `accepted`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_mhcc_001.py`
- Candidate SHA256: `f29b71c87712aa3f674c2ec6e448bf1026a81986ecaa070645e691d66e969c39`
- Accepted reference: `baseline_adapter.py`
- Accepted reference SHA256: `c3cc90deb93c69f092e876ddc0057ddf6c2d7ac28020f1b7fd1ed260bca72fee`
- Base SHA256: `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Measurement fingerprint: `d8f4b63bfbf09ce8a32f3bdcd4d85553f34abce7384e495ba5f66baf49bf795e`

## Progress Log

| Step | Phase | Completed | Command result | Next safe action |
|---|---|---|---|---|
| identity check | verification start | yes | candidate/base/harness sha256 all match declared values | run correctness |
| correctness | after correctness | yes | exit 0; `PASS accuracy; v0=1.732976 ms, v1=0.124444 ms, speedup=13.926x` | run authoritative wall timing |
| authoritative wall run 1 | timing | yes | exit 0; v0=1.365771 ms, v1=0.118193 ms | run run 2 |
| authoritative wall run 2 | timing | yes | exit 0; v0=1.665487 ms, v1=0.118501 ms | run run 3 |
| authoritative wall run 3 | timing | yes | exit 0; v0=1.673250 ms, v1=0.118357 ms | compute medians, run profiler |
| profiler | profiling | yes | exit 0; trace exported to `log/round_001_forward_50iter.pt.trace.json` | summarize scopes |
| summarize (filtered) | profiling | yes | both scopes summarized (C500 duplicate gpu_user_annotation markers filtered) | write report |
| report + status | verification end | yes | `report_001.md` and `round_status_001.md` written | Orchestrator transition |

## Wall Timing Raw Samples (authoritative, unrounded medians per run)

- warmup `50`, repeat `100`
- reference (v0) samples ms: `[1.365771, 1.665487, 1.673250]`
- candidate (v1) samples ms: `[0.118193, 0.118501, 0.118357]`
- reference median ms: `1.665487`
- candidate median ms: `0.118357`
- improvement_pct: `92.893`

## Profiler Summary (filtered trace, iterations 50)

- baseline_base: device_total_us 26700.691, device_us_per_call 534.014, kernel_count_total 6650, kernel_count_per_call 133.0
- candidate_triton_mhcc_001: device_total_us 2189.568 (true, manual), device_us_per_call 43.791, kernel_count_total 50, kernel_count_per_call 1.0 (single `_mhc_head_compute_mix_kernel`)

## Artifacts Written

- `rounds/report_001.md`
- `rounds/round_status_001.md`
- `state/verifier_context.md` (updated)
- `log/round_001_forward_50iter.pt.trace.json` (raw, preserved)
- `log/round_001_forward_50iter.filtered.pt.trace.json` (duplicate nested gpu_user_annotation scope markers removed)

## Classification

- Result: `accepted`. Correctness PASS, all guardrails PASS, all five mechanism_observables
  met, wall improvement 92.9% (>> 5% threshold). Orchestrator applies the canonical
  pointer transition to `triton_mhcc_001.py` and the `accepted` counter/streak update.
