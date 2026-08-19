# Round Status 003

- phase: complete
- last update: verification complete
- current_round: 003
- terminal_result: accepted

## Completed commands

| Step | Command | Status | Result |
|---|---|---|---|
| Hash verify | sha256sum candidate + reference | done | candidate eb065f9a..., reference 1b5c8ecd... match |
| Correctness | --v0_file base.py --v1_file triton_fused_moe_003.py --warmup 5 --repeat 10 --full-traceback | done | PASS accuracy; v0=7.983585 ms, v1=0.375470 ms, speedup=21.263x |
| Interleaved timing (pair 1) | ref=002 cand=003 (v0=base.py anchor) --warmup 50 --repeat 100 | done | ref=0.387520, cand=0.373490 |
| Interleaved timing (pair 2) | same | done | ref=0.400320, cand=0.341015 |
| Interleaved timing (pair 3) | same | done | ref=0.414580, cand=0.383115 |
| Profiler | --profile forward, --profile-reference-file triton_fused_moe_002.py | done | two clean CANN scopes, summarized |

## Authoritative timing (median of 3 pairs)

- reference_median_ms (triton_fused_moe_002.py): `0.400320`
- candidate_median_ms (triton_fused_moe_003.py): `0.373490`
- improvement_pct: `6.702138`

## Profiler summary (iterations=50)

| Scope | Device us/call | Kernel count/call | Device ratio |
|---|---:|---:|---:|
| reference (triton_fused_moe_002.py) | 26.641 | 3.0 | 0.066549 |
| candidate (triton_fused_moe_003.py) | 26.622 | 3.0 | 0.071278 |

## Evaluation Contract H-003 observables

| Observable | Reference | Candidate | Verdict |
|---|---:|---:|---|
| output_allocations_per_call | 1 | 0 (lazy alloc, reused) | pass |
| host_us_per_call | (wall 0.400320) | (wall 0.373490, device unchanged) | pass (decrease) |
| device_us_per_call | 26.641 | 26.622 | pass (unchanged) |

Hypothesis verdict: `confirmed`

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| candidate (triton_fused_moe_003.py) | eb065f9a4371686b7ad028bb003501047b512265190b42438a559df05e85fb0d |
| accepted reference (triton_fused_moe_002.py) | 1b5c8ecded2008991f0f7cc039f0e06fa072bf5b8a7c6d5630574f64a43f4074 |

## Next safe action

Await Orchestrator. Round 3 complete; terminal result `accepted`.
