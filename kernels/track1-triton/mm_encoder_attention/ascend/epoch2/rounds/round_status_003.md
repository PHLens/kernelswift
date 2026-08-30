# Round Status 003

Updated at: verification end.

## State

- phase: `verifying`
- round: `003`
- classification: **`accepted`**
- hypothesis verdict: `confirmed` (`H-003`)
- measurement_exclusive: `true`
- resume_safe: yes, complete. No further safe step pending for Verifier.

## Identity

| Artifact | SHA-256 |
|---|---|
| decision_003.md | `a4956891de5fef4b9bd629fb3cceb270db5a247ba18b591aecee9480d96c5455` |
| candidate `triton_mm_encoder_attention_e2_003.py` | `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe` |
| accepted reference `triton_mm_encoder_attention_e2_001.py` | `c75ec5ffaab3883ef7c5b1e62778b39fbd5413619a625fd36a86d70390e92124` |
| `base.py` (paired `--v0_file`) | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` |
| `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` |
| sketch_003.json | `51ebe3a735c7659309e781fd2f35286fd4e67acc86b5d0a9f6676f08f08af69c` |
| coder_result_003.md | `d60e74e94f5e87ffbe2c535f8caea8d58c1fc7d4b104e1b0351fb9d854ac948d` |

## Completed Commands

| # | Command | Status |
|---:|---|---|
| 1 | `auto_bench.py --v0_file base.py --v1_file ..._e2_003.py --warmup 5 --repeat 10 --full-traceback` | exit 0, `PASS accuracy; v0=0.358365 ms, v1=0.307995 ms, speedup=1.164x` |
| 2 | paired measurement `--warmup 50 --repeat 100`, run 1 | exit 0, `v0=0.361050 ms, v1=0.298240 ms` |
| 3 | paired measurement `--warmup 50 --repeat 100`, run 2 | exit 0, `v0=0.362085 ms, v1=0.302220 ms` |
| 4 | paired measurement `--warmup 50 --repeat 100`, run 3 | exit 0, `v0=0.346350 ms, v1=0.296060 ms` |
| 5 | dual-scope profiler, `--profile-mode forward --profile-warmup 20 --profile-iterations 50`, reference `e2_001` | exit 0, both CANN captures parsed |
| 6 | `summarize_cann_trace.py` reference scope, `--iterations 50 --wall-ms 0.335740` | exit 0, `device_us_per_call 13.4096`, `kernel_count_per_call 1.0` |
| 7 | `summarize_cann_trace.py` candidate scope, `--iterations 50 --wall-ms 0.301730` | exit 0, `device_us_per_call 13.4224`, `kernel_count_per_call 1.0` |
| 8 | reference-scope wall at `--warmup 200 --repeat 500` on `e2_001` | exit 0, `v0=0.364975 ms, v1=0.335740 ms` |
| 9 | interleaved control, 3 blocks of `e2_001` then `e2_003` at `50/100` | exit 0, all six runs PASS |
| 10 | `log/round_003_host_decomposition.py` | exit 0 |
| 11 | `log/round_003_alloc_probe.py` | exit 0 |
| 12 | `log/round_003_harness_overhead.py` | exit 0 |
| 13 | `log/round_003_guardrails.py` | exit 0 |
| 14 | `diff` lines 1-74 `e2_001` vs `e2_003` | exit 0, `KERNEL_BODY_IDENTICAL=yes` |

## Raw Samples

| Pair | Reference median ms | Candidate median ms | Correctness |
|---:|---|---|---|
| 1 | 0.361050 | 0.298240 | PASS |
| 2 | 0.362085 | 0.302220 | PASS |
| 3 | 0.346350 | 0.296060 | PASS |
| **median of medians** | **0.361050** | **0.298240** | PASS |

```text
improvement_pct = (0.361050 - 0.298240) / 0.361050 * 100 = 17.3965
```

Interleaved control against the accepted kernel (`50/100`, medians of 3 blocks):

| | base.py | candidate |
|---|---:|---:|
| e2_001 | 0.358765 | 0.329810 |
| e2_003 | 0.349320 | 0.292845 |

```text
raw             = 11.2080%
base-normalized =  8.8072%
```

## Control Observables (gate)

| Observable | e2_001 | e2_003 | Verdict |
|---|---:|---:|---|
| `device_us_per_call` | 13.4096 | 13.4224 | held (`+0.095%`) |
| `kernel_count_per_call` | 1.00 | 1.00 | held |
| `output_allocations_per_call` | 1.00 | 0.00 | confirmed decrease |
| `host_us_per_call` (forward alone) | 233.645 | 206.375 | confirmed decrease (`-27.270 us`) |

Device moved `+0.0128 us` against a `-36.965 us` wall delta, i.e. `0.035%` of
the wall change. Attribution to host is intact.

## Level 2 Summary

| Quantity | e2_001 | e2_003 |
|---|---:|---:|
| (a) harness wall | 327.535 | 297.410 |
| (b) forward alone | 233.645 | 206.375 |
| (c) forward + sync | 288.290 | 258.190 |
| (a) - (b) harness-fixed | 93.890 | 91.035 |

- (d) allocation-free direct launch: `183.740`
- lever `(b_e2_001) - (b_e2_003)`: `27.270`
- residual wrapper `(b_e2_003) - (d)`: `22.635`
- harness-fixed ceiling: `91.035 us/call` = `30.61%` of wall

## Artifacts Written

| Path | Purpose |
|---|---|
| `rounds/report_003.md` | terminal evidence |
| `rounds/round_status_003.md` | this file |
| `state/verifier_context.md` | refreshed to round 003 |
| `log/round_003_forward_50iter.pt.trace.json` | chrome trace (candidate scope only) |
| `log/profiling_data/reference_triton_mm_encoder_attention_e2_001/…` | CANN capture, reference scope |
| `log/profiling_data/candidate_triton_mm_encoder_attention_e2_003/…` | CANN capture, candidate scope |
| `log/round_003_host_decomposition.py` | Level 2 probe |
| `log/round_003_alloc_probe.py` | allocation counter |
| `log/round_003_harness_overhead.py` | harness overhead probe |
| `log/round_003_guardrails.py` | safety guardrail probe |

No incident occurred; no `incident_003_*.md` was written. `base.py`, the
harness, the candidate, and `decision_003.md` are unmodified.

## Next Safe Action

Verifier's work for round 003 is complete. Orchestrator applies the state
transition: promote `triton_mm_encoder_attention_e2_003.py` to
`last_accepted_kernel` with `rounds/report_003.md` as
`last_accepted_report`, reset the no-improvement streak, and append the round row
to `project.md`. Verifier does not update canonical pointers.
