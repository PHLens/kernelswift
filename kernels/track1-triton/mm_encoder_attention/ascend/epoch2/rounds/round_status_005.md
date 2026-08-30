# Round Status 005

Updated at: verification end.

## State

- phase: `verifying`
- round: `005`
- classification: **`accepted`**
- hypothesis verdict: **`confirmed`** (`H-005`) — every link held, including the wall link
- capability gate: `lifecycle.fast-launcher` — `proven` by citation of retained round-004 artifacts; mechanism `M2`, shipped as **M2b (per-call stream)**
- measurement_exclusive: `true`
- resume_safe: yes, complete. No further safe step pending for Verifier.

## Result summary

| Test | Result | Verdict |
|---|---|---|
| **Governing: `speedup(e2_005)/speedup(e2_003) - 1 >= 5%`**, strict alternation, one window | **+41.5498%** | **CLEARS by 36.55 pts** |
| Wall improvement over canonical `e2_003`, same window, raw | `28.8329%` | CLEARS |
| Wall improvement over canonical, in-process paired | `28.724%` | CLEARS |
| Standard protocol vs same-turn `base.py` (cumulative) | `40.7148%` | CLEARS |

5 of 5 alternating pairs clear the bar; weakest pair `+34.3058%`.

## Identity

| Artifact | SHA-256 |
|---|---|
| decision_005.md | `1fdd16d7ddca961760260b9e6130c7e6d2fb17b689728474ee9e5bea9b8ce551` |
| candidate `triton_mm_encoder_attention_e2_005.py` | `bf54cea2a1fcdafd8916c2e0bf607766a6e7ffc2981fd956e18e92bf51b88b26` |
| canonical `triton_mm_encoder_attention_e2_003.py` | `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe` |
| `base.py` (paired `--v0_file`) | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` |
| `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` |
| sketch_005.json | `f44ed2bfbef80e9dc603494221bbc2cd47db40a9d8d48d85ee2ae344cd11c4ee` |
| coder_result_005.md | `b8f8a06fddaa4328dc340ece21af997acab7250b6a4a0db33df335f68a087268` |

## Completed Commands

| # | Command | Status |
|---:|---|---|
| 1 | `auto_bench.py --v0_file base.py --v1_file ..._e2_005.py --warmup 5 --repeat 10 --full-traceback` | exit 0, `PASS accuracy; v0=0.364795 ms, v1=0.222275 ms, speedup=1.641x` |
| 2 | paired measurement `--warmup 50 --repeat 100`, run 1 | exit 0, `v0=0.358445 ms, v1=0.212025 ms` |
| 3 | paired measurement, run 2 | exit 0, `v0=0.358600 ms, v1=0.213610 ms` |
| 4 | paired measurement, run 3 | exit 0, `v0=0.352430 ms, v1=0.212505 ms` |
| 5 | **strict pair-by-pair alternation**, 5 pairs each of `e2_005` and `e2_003`, `50/100` | exit 0, 10 runs PASS |
| 6 | dual-scope profiler, `--profile-mode forward --profile-warmup 20 --profile-iterations 50`, reference `e2_003` | exit 0, both CANN captures parsed |
| 7 | `summarize_cann_trace.py` reference scope (explicit round-005 db), `--iterations 50 --wall-ms 0.299880` | exit 0, `device_us_per_call 13.4816`, `kernel_count_per_call 1.0` |
| 8 | `summarize_cann_trace.py` candidate scope, `--iterations 50 --wall-ms 0.210760` | exit 0, `device_us_per_call 13.4780`, `kernel_count_per_call 1.0` |
| 9 | reference-scope wall at `--warmup 200 --repeat 500` on `e2_003` | exit 0, `v0=0.375360 ms, v1=0.299880 ms` |
| 10 | `log/round_005_host_decomposition.py` | exit 0 |
| 11 | `log/round_005_guardrails.py` | exit 0 |
| 12 | `diff` lines 1-76 `e2_003` vs `e2_005`; mechanism `grep` | exit 0, `KERNEL_DEF_IDENTICAL=yes`, no `stream=` at the launch site |

## Raw Samples

Protocol (three pairs, `base.py` vs candidate, `50/100`):

| Pair | Reference median ms | Candidate median ms | Speedup | Improvement |
|---:|---:|---:|---:|---:|
| 1 | 0.358445 | 0.212025 | 1.690579 | 40.8487% |
| 2 | 0.358600 | 0.213610 | 1.678760 | 40.4322% |
| 3 | 0.352430 | 0.212505 | 1.658455 | 39.7029% |
| **median** | **0.358445** | **0.212505** | **1.678760** | **40.7148%** |

Decisive — strict alternation, one window (5 pairs each):

| Pair | e2_005 ref | e2_005 cand | speedup | e2_003 ref | e2_003 cand | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.353115 | 0.210710 | 1.675834 | 0.364575 | 0.292180 | 1.247775 |
| 2 | 0.352980 | 0.209905 | 1.681618 | 0.352875 | 0.296900 | 1.188531 |
| 3 | 0.370555 | 0.214640 | 1.726402 | 0.353560 | 0.288395 | 1.225957 |
| 4 | 0.364895 | 0.212420 | 1.717800 | 0.355295 | 0.303285 | 1.171489 |
| 5 | 0.355475 | 0.211295 | 1.682364 | 0.342280 | 0.303145 | 1.129097 |
| **median** | | | **1.682364** | | | **1.188531** |

```text
ratio = 1.682364 / 1.188531 = 1.415498  ->  +41.5498%   BAR +5.00%  ->  CLEARS
per-pair gains: +34.3058%  +41.4870%  +40.8207%  +46.6339%  +49.0008%  (5 of 5 clear)
raw wall improvement (same window): 0.296900 -> 0.211295 = 28.8329%
```

## Control Observables (gate)

| Observable | e2_003 | e2_005 | Verdict |
|---|---:|---:|---|
| `device_us_per_call` | 13.4816 | 13.4780 | held (`-0.027%`) |
| `kernel_count_per_call` | 1.00 | 1.00 | held |
| `launch_path_us_per_call` | M0 178.915 | M2b 89.220 | confirmed, `-89.695 us` |
| `host_us_per_call` (forward alone) | 203.485 | 115.185 | confirmed, `-88.300 us` |

Device moved `-0.0036 us` against a `-85.685 us` wall change: `0.004%` of the
movement. Attribution is host.

## Mechanism check (team-lead request)

Shipped path is **M2b**, confirmed two ways:
- source: `kernel[grid](query, key, value, out, stride0, stride1, S, scale)` — no `stream=`
- runtime: instrumented `driver.active.get_current_stream` → `1.00` resolutions/call

Bare launch costs in one process: M0 `178.915`, M2b `89.220`, M2a `66.220`;
per-call stream resolution costs `23.000 us`. A near-`89` reading is correct,
not a regression.

## Level 2 Summary

| Quantity | e2_003 | e2_005 |
|---|---:|---:|
| (a) harness wall | 298.130 | 212.445 |
| (b) forward alone | 203.485 | 115.185 |
| (c) forward + sync | 256.185 | 162.605 |
| (a) - (b) harness-fixed | 94.645 | 97.260 |

- wall lever: `-85.685 us` median-of-medians, `-85.635 us` paired
- forward lever: `-88.300 us` / `-86.900 us` paired (Coder reported `-90.955`; agreement within `2.7 us`)
- propagation `97.0%`, better than round 004's `~75%`
- residual wrapper unchanged: `24.570` vs `25.965 us`

## Artifacts Written

| Path | Purpose |
|---|---|
| `rounds/report_005.md` | terminal evidence |
| `rounds/round_status_005.md` | this file |
| `state/verifier_context.md` | refreshed to round 005 |
| `log/round_005_forward_50iter.pt.trace.json` | chrome trace (candidate scope only) |
| `log/profiling_data/reference_triton_mm_encoder_attention_e2_003/…` | CANN capture, reference scope (round-005 db) |
| `log/profiling_data/candidate_triton_mm_encoder_attention_e2_005/…` | CANN capture, candidate scope |
| `log/round_005_host_decomposition.py` | Level 2 probe |
| `log/round_005_guardrails.py` | safety guardrail probe |

No incident occurred; no `incident_005_*.md` was written. `base.py`, the
harness, the candidate, and `decision_005.md` are unmodified.

## Next Safe Action

Verifier's work for round 005 is complete. Orchestrator applies the state
transition: promote `triton_mm_encoder_attention_e2_005.py` to
`last_accepted_kernel` with `rounds/report_005.md` as `last_accepted_report`,
reset the no-improvement streak from `1` to `0`, and append the round row to
`project.md`. Verifier does not update canonical pointers.

Highest-value next step implied by this round's evidence: a decision amending
`device_stream_behavior` to permit a cached stream, worth `23.000 us`. The
supporting measurement exists from two independent probes.
