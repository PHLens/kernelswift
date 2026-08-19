# Round Status 001

Result: accepted

## Phase

- current_phase: `verifying` (measurement complete, awaiting Orchestrator transition)
- round: `001`
- decision: `rounds/decision_001.md`

## Completed Commands

1. Correctness (warmup 5, repeat 10, full-traceback) — `RETURN_CODE=0`
   - Output: `PASS accuracy; v0=0.139143 ms, v1=0.186143 ms, speedup=0.748x`
2. Authoritative wall timing run 1 (warmup 50, repeat 100) — `RETURN_CODE=0`
   - `v0=0.108863 ms, v1=0.164166 ms, speedup=0.663x`
3. Authoritative wall timing run 2 (warmup 50, repeat 100) — `RETURN_CODE=0`
   - `v0=0.111197 ms, v1=0.166405 ms, speedup=0.668x`
4. Authoritative wall timing run 3 (warmup 50, repeat 100) — `RETURN_CODE=0`
   - `v0=0.110801 ms, v1=0.163559 ms, speedup=0.677x`
5. Profiler (forward, warmup 20, iterations 50) — `RETURN_CODE=0`
   - `profile=kernels/track1-triton/mm_encoder_attention/maca/log/round_001_forward_50iter.pt.trace.json`
   - Also produced filtered trace `round_001_forward_50iter.filtered.pt.trace.json`
     (dropped 2 duplicate `cat=user_annotation` X events to resolve the known
     C500 `overlapping scope events` issue).

## Raw Samples and Medians

- reference_raw_samples_ms (v0): `[0.108863, 0.111197, 0.110801]`
- candidate_raw_samples_ms (v1): `[0.164166, 0.166405, 0.163559]`
- reference_median_ms: `0.110801`
- candidate_median_ms: `0.164166`
- improvement_pct: `-48.162922717304006`

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` |
| `triton_mha_001.py` (candidate) | `9fac12aa0298a970c208dbc6af7a602da4f34e43d44921b62aad571ca662c00b` |
| `baseline_adapter.py` (accepted reference) | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` |
| `decision_001.md` | `ab2f5bb98a8f491ed67e2a05850fc28e9bf0958a09ef89ec3f32c8f24a0a949d` |

## Profiler Summary

| Scope | Device us/call | Kernel count/call | Dominant kernels |
|---|---:|---:|---|
| baseline_base (v0) | 15.08359375 | 2.0 | `flash_fwd_splitkv_kernel`, `flash_fwd_splitkv_combine_kernel` |
| candidate_triton_mha_001 (v1) | 79.697666015625 | 5.0 | `_mha_fwd_kernel` (1.0/call), `transpose12_copy_64` (4.0/call) |

The candidate emits one fused `_mha_fwd_kernel` (replacing the two mcFlashAttn
kernels) plus four `transpose12_copy_64` copy kernels from the `.contiguous()`
calls on q/k/v in the benchmark path.

## Next Safe Action

Orchestrator records `accepted` terminal result (epoch-2 deliverable policy:
correctness parity is the acceptance gate) and advances the canonical pointer
to `triton_mha_001.py`.
