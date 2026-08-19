# Final Summary — MHCHeadComputeMix C500 (MACA)

- schema_version: 1
- skill_version: 2.0.0
- run_epoch: 1
- run_branch: kernel-opt/mhc-head-compute-mix-c500-20260819
- base_branch: dev
- base_commit: 138f0caf13784399abdc29507a6ac1f29e0fd947
- measurement_fingerprint: d8f4b63bfbf09ce8a32f3bdcd4d85553f34abce7384e495ba5f66baf49bf795e
- stop_reason: user-intervention
- stop_timestamp: 2026-08-19T01:00:00Z
- total_rounds: 2
- accepted_round: 001
- canonical_kernel: triton_mhcc_001.py

## Outcome

- baseline wall: 1.515187 ms (warmup 50 / repeat 100)
- final wall: 0.118357 ms
- improvement: +92.89% (14.07x speedup; device 534.014 -> 43.791 us/call, kernels/call 133 -> 1)

## Round Summary

| Round | Decision | Result | Wall ms | Improvement | Canonical |
|---:|---|---|---:|---:|---|
| 000 | Phase 0 | baseline | 1.515187 | - | baseline_adapter.py |
| 001 | H-001 sinkhorn-loop-fusion | accepted | 0.118357 | +92.89% | triton_mhcc_001.py |
| 002 | abort (latency-floor) | aborted | - | - | triton_mhcc_001.py |

## Why stopped

Round 001 fused the entire forward — sigmoid gates, exp/row_max stabilization,
and the 20-iteration Sinkhorn alternating normalization — into one Triton kernel
(one program per (b,s) position, 16 programs). This collapsed 133 library
launches -> 1 and eliminated ~65% of wall (host launch overhead), reducing
device time 534.014 -> 43.791 us/call and wall 1.515187 -> 0.118357 ms
(+92.89%). The exact fp32 semantics (including the eps-placement asymmetry:
first row-normalize adds eps after division, all others add eps inside the
denominator) were reproduced precisely.

Round 002 Designer found the remaining time is at a latency floor: the 20-step
Sinkhorn forms a serial dependency chain (col-normalize depends on prior
row-normalize) that cannot be parallelized without changing exact fp32
semantics; the single kernel launch is already minimal (grid 16 -> 1 does not
reduce launch count and would lengthen the serial chain); and the ~63% host
component is harness-fixed (set_seed + sync_devices) plus 3 tiny output
allocations. No candidate-owned intervention clears the 5% threshold (5.92 us).
Conclusion: measurement-bound / latency-bound.

## Resume constraints

Resume only with a same-runtime microbenchmark proving a compressible
candidate-owned bottleneck >= 5.92 us/call, or a user-mandated contract change.
