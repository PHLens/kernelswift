# Round Status 008

Status: accepted

- Candidate: `triton_grouped_topk_008.py`
- Candidate SHA-256: `d1fb6b03d3be92cdd6423f1f44f33ea81d13f0e4df18227fe2d5f7dceb582535`
- Accepted reference: `triton_grouped_topk_004.py`
- Decision: `rounds/decision_008.md`
- Decision SHA-256: `bec59b81693001fd27302a610ab48123e38a4a81c44b65cedfff9530b059e5d1`
- Measurement fingerprint: `57bf01d317ee03ca2b09730e648f0f93d2bf4f226639ca3af2b1ff57b2865575`
- Measurement exclusive: `true` until the terminal artifact commit

## Verification Gates

- Correctness: pass. The independent harness check and all-equal, two-expert-tie, and structured-group-tie suites matched reference IDs exactly and weights within tolerance.
- Timing pair 1: reference `0.430385 ms`; candidate `0.344360 ms`.
- Timing pair 2: reference `0.438276 ms`; candidate `0.347371 ms`.
- Timing pair 3: reference `0.411169 ms`; candidate `0.327867 ms`.
- Paired medians: reference `0.430385 ms`; candidate `0.344360 ms`; unrounded improvement `19.987917795%`.
- Profiler: pass. The raw trace is `log/groupedtopk_round008_forward_50iter.pt.trace.json`, SHA-256 `bfd01278b7487ec053467d677ac0e912a089db1d521be8b7614505e65af2910f`; normalized scope summaries report reference `127.45888671875 us/call`, `9.9 kernels/call`, and candidate `111.120595703125 us/call`, `8.96 kernels/call`.
- Terminal verdict: accepted.
