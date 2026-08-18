# Round Status 001

- phase: complete
- last update: verification complete
- current_round: 001
- terminal_result: accepted

## Completed commands

| Step | Command | Status | Result |
|---|---|---|---|
| Hash verify | sha256sum candidate + reference | done | candidate e42d811a..., reference a7fc0001... match |
| Correctness | --v0_file base.py --v1_file triton_fused_moe_001.py --warmup 5 --repeat 10 --full-traceback | done | PASS accuracy; v0=8.190205 ms, v1=0.612535 ms, speedup=13.371x |
| Interleaved timing (pair 1) | --v0_file base.py --warmup 50 --repeat 100 | done | v0=7.816390, v1=0.588975 |
| Interleaved timing (pair 2) | same | done | v0=8.126030, v1=0.547805 |
| Interleaved timing (pair 3) | same | done | v0=7.242815, v1=0.569590 |
| Profiler | --profile forward, --profile-reference-file baseline_adapter.py | done | two clean CANN scopes, summarized |

## Authoritative timing (median of 3 pairs)

- reference_median_ms (base.py): `7.816390`
- candidate_median_ms (triton_fused_moe_001.py): `0.569590`
- improvement_pct: `92.712876`

## Profiler summary (iterations=50)

| Scope | Device us/call | Kernel count/call | Device ratio |
|---|---:|---:|---:|
| reference (baseline_adapter.py) | 746.641 | 126.0 | 0.095523 |
| candidate (triton_fused_moe_001.py) | 97.366 | 12.0 | 0.170941 |

## Evaluation Contract H-001 observables

| Observable | Reference | Candidate | Verdict |
|---|---:|---:|---|
| kernel_count_per_call | 126.0 | 12.0 | pass (decrease) |
| device_us_per_call | 746.641 | 97.366 | pass (decrease) |
| aclnnNonzeroV2_presence | present | absent | pass |
| aclnnIndexPutImpl_presence | present | absent | pass |

Hypothesis verdict: `confirmed`

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| candidate (triton_fused_moe_001.py) | e42d811a7aee32f3fb34b19c00f88bf7922129faccc3d670dd07abc2df443287 |
| accepted reference (baseline_adapter.py) | a7fc0001db3ee9e636241954d2c071b62acee518b23f4c59c19efee886203a02 |

## Next safe action

Await Orchestrator. Round 1 complete; terminal result `accepted`.
