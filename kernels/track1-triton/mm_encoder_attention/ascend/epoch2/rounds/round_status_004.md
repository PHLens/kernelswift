# Round Status 004

Updated at: verification end.

## State

- phase: `verifying`
- round: `004`
- classification: **`no-improvement`** (Verifier's read of the decision's table)
- hypothesis verdict: `partially-confirmed` (`H-004`)
- capability gate: `lifecycle.fast-launcher` → `proven` (Coder); Verifier re-verified its invariants
- measurement_exclusive: `true`
- resume_safe: yes, complete. No further safe step pending for Verifier.

### The classification, against every bar that was applied

| Bar | Result | Verdict |
|---|---|---|
| **+5% vs previous round's accepted candidate** (governing, advisory 2) | **`+2.8874%`** ratio-of-speedups, strict same-turn alternation | **FAILS by 2.11 pts** |
| Same bar, cross-turn derived form (`1.271132`) | `1.270171` protocol window / `1.217744` interleaved | **FAILS** both |
| +5% over canonical `e2_003` (discriminating form) | `2.5939%` raw / `1.9317%` normalized / `4.35%` in-process | **FAILS** |
| +5% vs same-turn `base.py` (protocol number) | `21.3249%` | CLEARS, but not attributable to this round |
| ~~epoch-1 deliverable, bar `1.02626`~~ *(superseded)* | median `1.270171` | clears, but the incumbent clears it too |

**All three adoption tests fail.** Verifier classified `no-improvement`.
Confidence is high: every estimator puts the gain between `1.9%` and `4.4%`, and
under the governing bar the **best of four** interleaved pairs reaches only
`+3.474%`, so 5% is outside the spread rather than inside it. This is not a
coin-flip on direction — `e2_004` is genuinely faster in `11 of 12` blocks and
in all `4` strictly alternated pairs — but the magnitude does not reach the bar
and cannot be made to by more sampling.

## Identity

| Artifact | SHA-256 |
|---|---|
| decision_004.md | `30758ad4dd30ccb0087534e47f61ea0443bdeead40ba64d41c28dd052c397088` |
| candidate `triton_mm_encoder_attention_e2_004.py` | `f5aa1d709e4deeb1562757d795dd4da41217238dd15c53d33c2c338da1938020` |
| canonical `triton_mm_encoder_attention_e2_003.py` | `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe` |
| `base.py` (paired `--v0_file`) | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` |
| `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` |
| sketch_004.json | `d3e52f6af032014381908e03e87a6b1c3f5694090686df2af3bfe3a6d9474dbf` |
| coder_result_004.md | `9c8c46ef1b58233e464a30022fd2b0dedf2fce7b95410a501d95e2e24ac59e0e` |

## Completed Commands

| # | Command | Status |
|---:|---|---|
| 1 | `auto_bench.py --v0_file base.py --v1_file ..._e2_004.py --warmup 5 --repeat 10 --full-traceback` | exit 0, `PASS accuracy; v0=0.373405 ms, v1=0.310160 ms, speedup=1.204x` |
| 2 | paired measurement `--warmup 50 --repeat 100`, run 1 | exit 0, `v0=0.362315 ms, v1=0.287335 ms` |
| 3 | paired measurement, run 2 | exit 0, `v0=0.380105 ms, v1=0.299255 ms` |
| 4 | paired measurement, run 3 | exit 0, `v0=0.376040 ms, v1=0.295850 ms` |
| 5 | interleaved control blocks 1-6 (`e2_003` first), `50/100` | exit 0, 12 runs PASS |
| 6 | interleaved control blocks 7-12 (`e2_004` first), `50/100` | exit 0, 12 runs PASS |
| 7 | dual-scope profiler, `--profile-mode forward --profile-warmup 20 --profile-iterations 50`, reference `e2_003` | exit 0, both CANN captures parsed |
| 8 | `summarize_cann_trace.py` reference scope, `--iterations 50 --wall-ms 0.297245` | exit 0, `device_us_per_call 13.3272`, `kernel_count_per_call 1.0` |
| 9 | `summarize_cann_trace.py` candidate scope, `--iterations 50 --wall-ms 0.280915` | exit 0, `device_us_per_call 13.3228`, `kernel_count_per_call 1.0` |
| 10 | reference-scope wall at `--warmup 200 --repeat 500` on `e2_003` | exit 0, `v0=0.365930 ms, v1=0.297245 ms` |
| 11 | `log/round_004_host_decomposition.py` | exit 0 |
| 12 | `log/round_004_guardrails.py` | exit 0 |
| 13 | `diff` lines 1-76 `e2_003` vs `e2_004` | exit 0, `KERNEL_DEF_IDENTICAL=yes` |
| 14 | epoch-1 bar verified from `../rounds/report_001.md` | `0.348605 / 0.339685`, improvement `2.5588` |
| 15 | **strict pair-by-pair alternation control** (advisory 2), 4 pairs each of `e2_004` and `e2_003`, `50/100` | exit 0, 8 runs PASS |

## Raw Samples

Protocol (three pairs, `base.py` vs candidate, `50/100`):

| Pair | Reference median ms | Candidate median ms | Speedup | Correctness |
|---:|---:|---:|---:|---|
| 1 | 0.362315 | 0.287335 | 1.260950 | PASS |
| 2 | 0.380105 | 0.299255 | 1.270171 | PASS |
| 3 | 0.376040 | 0.295850 | 1.271050 | PASS |
| **median** | **0.376040** | **0.295850** | **1.270171** | PASS |

```text
improvement_pct (vs base.py) = (0.376040 - 0.295850) / 0.376040 * 100 = 21.3249
```

Controlled comparison (12 interleaved blocks, `50/100`):

```text
e2_003 candidate median 0.299935 ms   e2_004 candidate median 0.292155 ms
RAW             = 2.5939%   (7.780 us)
BASE-NORMALIZED = 1.9317%   (5.794 us)
median paired diff          = -8.623 us   (11 of 12 blocks favour e2_004)
threshold (report_003 wall) = 14.871 us
```

Strict pair-by-pair alternation control (the governing bar, advisory 2):

| Pair | e2_004 ref | e2_004 cand | speedup | e2_003 ref | e2_003 cand | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.350730 | 0.288005 | 1.217791 | 0.350265 | 0.292450 | 1.197692 |
| 2 | 0.358325 | 0.295770 | 1.211499 | 0.345970 | 0.295295 | 1.171608 |
| 3 | 0.355535 | 0.289480 | 1.228185 | 0.358745 | 0.302240 | 1.186954 |
| 4 | 0.351695 | 0.288820 | 1.217696 | 0.357425 | 0.302855 | 1.180185 |
| **median** | | | **1.217744** | | | **1.183570** |

```text
ratio = 1.217744 / 1.183570 = 1.028874  ->  +2.8874%   BAR +5.00%  ->  FAILS
per-pair gains: +1.678%  +3.405%  +3.474%  +3.178%   (best pair +3.474%, below 5%)
```

Derived cross-turn bar: `speedup(e2_003)` round 003 = `1.210602`, bar =
`1.271132`. `e2_004` measures `1.270171` (protocol window, `-0.0756%`) or
`1.217744` (interleaved window, `-4.2001%`) — fails on both. The `-0.0756%`
near-miss is not meaningful: `e2_004`'s speedup moved `4.31%` between two
windows of this same turn because `base.py` moved `6.0%` while the candidate
moved `2.3%`.

## Control Observables (gate)

| Observable | e2_003 | e2_004 | Verdict |
|---|---:|---:|---|
| `device_us_per_call` | 13.3272 | 13.3228 | held (`-0.03%`) |
| `kernel_count_per_call` | 1.00 | 1.00 | held |
| `launch_path_us_per_call` | 192.255 (M0) | 172.950 (M1) | confirmed, `-19.305 us` |
| `host_us_per_call` (forward alone) | 221.605 | 202.640 | confirmed, `-18.965 us` |

Device moved `-0.0044 us` — attribution to the host launch link is intact.

## Level 2 Summary

| Quantity | e2_003 | e2_004 |
|---|---:|---:|
| (a) harness wall | 317.325 | 302.995 |
| (b) forward alone | 221.605 | 202.640 |
| (c) forward + sync | 273.350 | 257.090 |
| (a) - (b) harness-fixed | 95.720 | 100.355 |

- wall lever: `-14.330 us` (median of medians), `-13.815 us` (paired) — **below the 14.871 us threshold by 1.056 us**
- forward lever: `-18.965 us` / `-16.260 us` (Coder reported `-18.470`; agreement within ~2.7 us)
- wall conversion is lossy: only `-14.330` of `-18.965 us` reaches wall

## Artifacts Written

| Path | Purpose |
|---|---|
| `rounds/report_004.md` | terminal evidence |
| `rounds/round_status_004.md` | this file |
| `state/verifier_context.md` | refreshed to round 004 |
| `log/round_004_forward_50iter.pt.trace.json` | chrome trace (candidate scope only) |
| `log/profiling_data/reference_triton_mm_encoder_attention_e2_003/…` | CANN capture, reference scope |
| `log/profiling_data/candidate_triton_mm_encoder_attention_e2_004/…` | CANN capture, candidate scope |
| `log/round_004_host_decomposition.py` | Level 2 probe |
| `log/round_004_guardrails.py` | safety guardrail probe |

No incident occurred; no `incident_004_*.md` was written. `base.py`, the
harness, the candidate, and `decision_004.md` are unmodified.

## Next Safe Action

Verifier's work for round 004 is complete. Orchestrator applies the state
transition. Under a `no-improvement` reading: the canonical kernel stays
`triton_mm_encoder_attention_e2_003.py`, the no-improvement streak moves to `1`
of `3`, and `total_rounds` moves to `3` of `20`. Verifier does not update
canonical pointers.

Highest-value next step implied by this round's evidence: a decision naming **M2
(cached `CompiledKernel`, `-119.360 us`) or M3 (`NPULauncher` C entry,
`-139.580 us`)** by name. The capability is already proven and the measurements
already exist in `log/probes/`; no new probe is required.
