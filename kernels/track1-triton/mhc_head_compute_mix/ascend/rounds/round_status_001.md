# Round Status 001

- phase: verifying (authoritative measurement) — COMPLETE
- current_round: 001
- verification_tier: authoritative
- result: accepted

## Completed Commands

1. correctness gate:
   `python3 auto_bench.py --v0_file base.py --v1_file candidate_001.py --warmup 50 --repeat 100`
   -> PASS accuracy
2. authoritative timing (3 interleaved pairs, warmup 50 / repeat 100):
   ref=[3.526815, 3.467300, 3.889960], cand=[0.392115, 0.394090, 0.386750]
   -> ref median 3.526815 ms, cand median 0.392115 ms, improvement 88.88%
3. profiler (CANN forward warmup 20 / iterations 50), dual scope:
   reference_baseline_adapter + candidate_candidate_001
4. summarize reference -> kernel_count_per_call 136.0, device_us_per_call 282.354 us
5. summarize candidate -> kernel_count_per_call 1.0, device_us_per_call 8.784 us

## Artifact Hashes

- accepted reference (baseline_adapter.py): `c3cc90deb93c69f092e876ddc0057ddf6c2d7ac28020f1b7fd1ed260bca72fee`
- candidate_001.py: `3eda8a14dede15a91f1a04c37bc5ff178a83fc87ecb7137b3569756c17f94f10`
- decision_001.md: `cfce60f6110bb21802b878f61a6238d89fed0320835560d2cfbd723107b881ef`

## Raw Samples

- reference_raw_samples_ms: [3.526815, 3.467300, 3.889960]
- candidate_raw_samples_ms: [0.392115, 0.394090, 0.386750]
- reference_median_ms: 3.526815
- candidate_median_ms: 0.392115
- improvement_pct: 88.8819

## Next Safe Action

- none (round 001 complete; report_001.md written). Hand off to Orchestrator for
  acceptance transition (last_accepted_kernel -> candidate_001.py).
