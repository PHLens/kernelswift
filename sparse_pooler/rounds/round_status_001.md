# Round Status 001

## Verification Start

- phase: `verifying`
- round: `001`
- started_at: `2026-08-14T14:00:00Z`
- verifier: claude-code (sequential main-session fallback not active; agent-team path)

## Artifacts and hashes

| Artifact | Path | SHA-256 |
|---|---|---|
| Decision | `rounds/decision_001.md` | `0816c943dfcfd157c9c4268196f4779b9804b9107de5fff0ba135d66f4f5bc75` |
| Candidate | `triton_sparse_pooler_001.py` | `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd` |
| Coder result | `rounds/coder_result_001.md` | (read; result=candidate-ready) |
| Accepted reference kernel | `baseline_adapter.py` | `d7e69ed4b66a4193a4475cc307fc0929cef807f875785652df6cc36fb2c487e5` |
| Accepted reference report | `rounds/report_000.md` | (baseline median 0.909974 ms; device 180.05 us/call; 10 kernels/call) |
| Source (immutable) | `base.py` | `ccccbbefadf1d697341451b542f17392acc8a2b9e4a3a41e50b2f9d58dbf61de` |
| Harness | `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` |
| Measurement fingerprint | - | `a0208c7da7e371d45c88f82ebddd3850d01669aa5d912f31db9234a7a56ebab7` |

## Decision summary

- Hypothesis H-001: fuse relu+log1p+per-segment max pooling into one Triton kernel.
- Expected wall improvement: 15%. Adoption threshold: 5%.
- Change scope: mixed (host Python loop elimination + device kernel fusion; inseparable and separately observable).
- Evaluation Contract observables: `kernel_count_per_call` (10→5), `device_us_per_call` (drop ~158 us minus fused cost), `host_sync_count_per_call` (decrease via D2H sync elimination).

## Completed steps

- Verifier role contract and runtime adapter read.
- All inputs read completely: team-state, decision_001, coder_result_001, candidate source, project.md, verifier_state, report_000, baseline_adapter, base.py, invariants, bottleneck-judgment.
- Hash verification: candidate and decision hashes match coder_result_001.md.

## Correctness gate

- command: `python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_001.py --warmup 50 --repeat 100`
- exit_code: `0`
- stdout: `PASS accuracy; v0=0.890184 ms, v1=0.596012 ms, speedup=1.494x` / `Summary: 1 passed, 0 failed, 1 total.`
- verdict: pass
- Note: this single run is the correctness gate (not the authoritative 3-pair timing). Speedup here is indicative only.

## Authoritative wall timing

- warmup: 50, repeat: 100, order: interleaved accepted-reference/candidate
- accepted reference = baseline_adapter.py (compared as v1 against base.py v0); candidate = triton_sparse_pooler_001.py (compared as v1 against base.py v0)

### Pair 1

- reference run: exit 0; `PASS accuracy; v0=0.909853 ms, v1=0.899378 ms` → reference sample = 0.899378 ms
- candidate run: exit 0; `PASS accuracy; v0=0.900469 ms, v1=0.606758 ms` → candidate sample = 0.606758 ms

### Pair 2

- reference run: exit 0; `PASS accuracy; v0=0.905048 ms, v1=0.910847 ms` → reference sample = 0.910847 ms
- candidate run: exit 0; `PASS accuracy; v0=0.898329 ms, v1=0.610001 ms` → candidate sample = 0.610001 ms

### Pair 3

- reference run: exit 0; `PASS accuracy; v0=0.909076 ms, v1=0.919767 ms` → reference sample = 0.919767 ms
- candidate run: exit 0; `PASS accuracy; v0=0.887124 ms, v1=0.600999 ms` → candidate sample = 0.600999 ms

### Summary

- reference_raw_samples_ms: `[0.899378, 0.910847, 0.919767]`
- candidate_raw_samples_ms: `[0.606758, 0.610001, 0.600999]`
- reference_median_ms: `0.910847`
- candidate_median_ms: `0.606758`
- improvement_pct: `33.38529961673036` (unrounded)

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
                = (0.910847 - 0.606758) / 0.910847 * 100
                = 33.38529961673036
```

Improvement exceeds the 5% adoption threshold.

## Level 1 profiler evidence

- command: `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_001.py --profile --profile-reference-file sparse_pooler/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output sparse_pooler/log/round_001_forward_50iter.pt.trace.json`
- exit_code: 0
- trace: `log/round_001_forward_50iter.pt.trace.json`
- iterations: 50
- summarize_trace.py hit "overlapping scope events" (Phase 0 known issue: 2 events per scope: user_annotation CPU + gpu_user_annotation GPU). Fell back to a custom GPU-scope summarizer using only `gpu_user_annotation` device-side intervals.

### Scope summary

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| reference_baseline_adapter | 8966.35 | 179.33 | 500 | 10.0 | 0.910847 | 0.1969 |
| candidate_triton_sparse_pooler_001 | 10506.03 | 210.12 | 250 | 5.0 | 0.606758 | 0.3463 |

- `kernel_count_per_call`: 10.0 → 5.0 (matches expectation exactly)
- `device_us_per_call`: 179.33 → 210.12 us/call (increased by 30.79 us; contradicts expectation of decrease)
- `host_sync_count_per_call`: baseline calls `seq_lens.tolist()` once per forward (1 D2H sync); candidate computes offsets on-device, eliminating the sync (matches expectation of decrease)

## Classification

- correctness: pass
- guardrails: pass
- improvement_pct: 33.38529961673036 (>= 5.0)
- terminal result: `accepted`

## Hypothesis verdict

- `kernel_count_per_call`: confirmed (10→5)
- `device_us_per_call`: falsified (increased from 179.33 to 210.12 us/call; the fused `_sparse_pooler_max_kernel` at 98.73 us/call is slower than the 6 baseline kernels it replaced at 67.87 us/call combined)
- `host_sync_count_per_call`: confirmed (D2H sync eliminated)
- improvement_pct (33.39%) >= 5.0% and >= 15% expected, but one observable contradicts → `partially-confirmed`

The wall improvement is driven by host-side Python-loop and D2H-sync elimination, not by device-side kernel cost reduction.

## Verification End

- phase: `verification complete`
- ended_at: `2026-08-14T14:06:00Z`
- terminal result: `accepted`
- hypothesis verdict: `partially-confirmed`
- improvement_pct: `33.38529961673036` (unrounded)
- stop recommendation: `continue`
- report: `rounds/report_001.md`
- state: `state/verifier_state.md` updated
- cleanup: no `__pycache__` created in the project root

## Next safe action

- Hand off to Orchestrator: validate `rounds/report_001.md`, apply the `accepted` terminal transition, update canonical pointers (`last_accepted_kernel` -> `triton_sparse_pooler_001.py`, `last_accepted_report` -> `rounds/report_001.md`), increment `total_rounds`, reset both streaks, commit, and evaluate stop criteria.
