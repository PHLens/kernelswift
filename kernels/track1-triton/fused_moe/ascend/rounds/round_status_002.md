# Round Status 002

- phase: complete
- last update: verification complete
- current_round: 002
- terminal_result: accepted

## Completed commands

| Step | Command | Status | Result |
|---|---|---|---|
| Hash verify | sha256sum candidate + reference | done | candidate 1b5c8ecd..., reference e42d811a... match |
| Correctness | --v0_file base.py --v1_file triton_fused_moe_002.py --warmup 5 --repeat 10 --full-traceback | done | PASS accuracy; v0=8.740895 ms, v1=0.430155 ms, speedup=20.320x |
| Interleaved timing (pair 1) | ref=001 cand=002 (v0=base.py anchor) --warmup 50 --repeat 100 | done | ref=0.545465, cand=0.406495 |
| Interleaved timing (pair 2) | same | done | ref=0.598405, cand=0.352400 |
| Interleaved timing (pair 3) | same | done | ref=0.575985, cand=0.368980 |
| Profiler | --profile forward, --profile-reference-file triton_fused_moe_001.py | done | two clean CANN scopes, summarized |

## Authoritative timing (median of 3 pairs)

- reference_median_ms (triton_fused_moe_001.py): `0.575985`
- candidate_median_ms (triton_fused_moe_002.py): `0.368980`
- improvement_pct: `35.939304`

## Profiler summary (iterations=50)

| Scope | Device us/call | Kernel count/call | Device ratio |
|---|---:|---:|---:|
| reference (triton_fused_moe_001.py) | 94.931 | 12.0 | 0.164815 |
| candidate (triton_fused_moe_002.py) | 26.678 | 3.0 | 0.072302 |

## Evaluation Contract H-002 observables

| Observable | Reference | Candidate | Verdict |
|---|---:|---:|---|
| kernel_count_per_call | 12.0 | 3.0 | pass (decrease) |
| device_us_per_call | 94.931 | 26.678 | pass (decrease) |
| aclnnTopk_presence | present | absent | pass |
| aclnnSoftmax_presence | present | absent | pass |

Hypothesis verdict: `confirmed`

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| candidate (triton_fused_moe_002.py) | 1b5c8ecded2008991f0f7cc039f0e06fa072bf5b8a7c6d5630574f64a43f4074 |
| accepted reference (triton_fused_moe_001.py) | e42d811a7aee32f3fb34b19c00f88bf7922129faccc3d670dd07abc2df443287 |

## Next safe action

Await Orchestrator. Round 2 complete; terminal result `accepted`.
