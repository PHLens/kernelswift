# Round Status 000

Result: baseline

## Phase

- current_phase: `verifying` (measurement complete, awaiting Orchestrator transition)
- round: `000`
- decision: `not-applicable: Phase 0`

## Completed Commands

1. Correctness (warmup 5, repeat 10, full-traceback) — `RETURN_CODE=0`
   - Output: `PASS accuracy; v0=0.130614 ms, v1=0.122941 ms, speedup=1.062x`
2. Authoritative wall timing run 1 (warmup 50, repeat 100) — `RETURN_CODE=0`
   - `v0=0.117305 ms, v1=0.115726 ms, speedup=1.014x`
3. Authoritative wall timing run 2 (warmup 50, repeat 100) — `RETURN_CODE=0`
   - `v0=0.117117 ms, v1=0.115881 ms, speedup=1.011x`
4. Authoritative wall timing run 3 (warmup 50, repeat 100) — `RETURN_CODE=0`
   - `v0=0.117557 ms, v1=0.115761 ms, speedup=1.016x`
5. Profiler (forward, warmup 20, iterations 50) — `RETURN_CODE=0`
   - `profile=kernels/track1-triton/mm_encoder_attention/maca/log/round_000_forward_50iter.pt.trace.json`
   - Also produced filtered trace `round_000_forward_50iter.filtered.pt.trace.json`
     (dropped 2 duplicate `cat=user_annotation` X events to resolve the known
     C500 `overlapping scope events` issue).

## Raw Samples and Medians

- reference_raw_samples_ms (v0): `[0.117305, 0.117117, 0.117557]`
- candidate_raw_samples_ms (v1): `[0.115726, 0.115881, 0.115761]`
- reference_median_ms: `0.117305`
- candidate_median_ms: `0.115761`
- improvement_pct: `1.316073818`

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` |
| `baseline_adapter.py` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` |

## Profiler Summary

| Scope | Device us/call | Kernel count/call | Dominant kernels |
|---|---:|---:|---|
| baseline_base (v0) | 15.068212890625 | 2.0 | `flash_fwd_splitkv_kernel`, `flash_fwd_splitkv_combine_kernel` |
| candidate_baseline_adapter (v1) | 14.98119140625 | 2.0 | `flash_fwd_splitkv_kernel`, `flash_fwd_splitkv_combine_kernel` |

SDPA lowers to flash attention (mcFlashAttn split-KV) on C500.

## Next Safe Action

Orchestrator records `baseline` terminal result, sets canonical
`baseline_adapter.py`, and advances to Round 1 design.
