# Round Status 009

Status: accepted

- Candidate: `triton_grouped_topk_009.py`
- Candidate SHA-256: `9b58f861ef6c3de86577dfe819327895311298cc4edf4b3f514f7fe9f4bff194`
- Accepted reference: `triton_grouped_topk_008.py`
- Accepted reference SHA-256: `d1fb6b03d3be92cdd6423f1f44f33ea81d13f0e4df18227fe2d5f7dceb582535`
- Decision: `rounds/decision_009.md`
- Decision SHA-256: `066045e737fa1aedcc283c4058d2eceb28b8630013c7b93342abdb516af908b8`
- Base SHA-256: `d57ace7d9196e2e44bdcfd17d1738482e7fd1bbb2d86fc6c9449c43938953eb5`
- Harness SHA-256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Measurement fingerprint: `57bf01d317ee03ca2b09730e648f0f93d2bf4f226639ca3af2b1ff57b2865575`
- Measurement exclusive: `true` until the terminal artifact commit

## Verification Gates

- Correctness: pass. Independent harness check (`PASS accuracy`, v0 `0.490558 ms`, v1 `0.289040 ms`) and all three tie suites matched reference IDs exactly with weights within tolerance:
  - all-equal: `[7,6,4,5,1,0,2,3]`
  - structured-group-tie: `[32,0,64,96,4,3,1,2]`
  - two-expert-tie: `[1,0,2,3,4,5,7,6]` (base == candidate element-for-element)
- Accepted-reference wrapper SHA-256: `1f57bf6d12a593dfb622428f3973461ef4744dc5ce45d152e1697e10110e90bf` (removed after collection).
- Timing pair 1: reference `0.357771 ms`; candidate `0.277234 ms`.
- Timing pair 2: reference `0.353104 ms`; candidate `0.273446 ms`.
- Timing pair 3: reference `0.360300 ms`; candidate `0.277749 ms`.
- Paired medians: reference `0.357771 ms`; candidate `0.277234 ms`; unrounded improvement `22.510768061134083%`.
- Profiler: pass. Raw trace `log/groupedtopk_round009_forward_50iter.pt.trace.json`, SHA-256 `56dd0b9e1e3a6f772274b2efe0dab087f412f248a749271b8e50ee5f8a2a3036`. Normalized scopes: reference `109.198896484375 us/call`, `8.78 kernels/call`; candidate `14.898740234375 us/call`, `1.22 kernels/call` (CUDA Graph replay hides graph-internal kernels).
- Terminal verdict: accepted.
