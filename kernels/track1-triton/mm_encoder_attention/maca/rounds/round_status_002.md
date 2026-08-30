# Round Status 002

Result: accepted

## Phase

- current_phase: `verifying` (measurement complete, awaiting Orchestrator transition)
- round: `002`
- decision: `rounds/decision_002.md`

## Completed Commands

1. Correctness (warmup 5, repeat 10, full-traceback) — `RETURN_CODE=0`
   - Output: `PASS accuracy; v0=0.123155 ms, v1=0.130640 ms, speedup=0.943x`
2. Authoritative wall timing run 1 (warmup 50, repeat 100) — `RETURN_CODE=0`
   - `v0=0.110857 ms, v1=0.127422 ms, speedup=0.870x`
3. Authoritative wall timing run 2 (warmup 50, repeat 100) — `RETURN_CODE=0`
   - `v0=0.125705 ms, v1=0.130439 ms, speedup=0.964x`
4. Authoritative wall timing run 3 (warmup 50, repeat 100) — `RETURN_CODE=0`
   - `v0=0.110812 ms, v1=0.127777 ms, speedup=0.867x`
5. Profiler (forward, warmup 20, iterations 50) — `RETURN_CODE=0`
   - `profile=kernels/track1-triton/mm_encoder_attention/maca/log/round_002_forward_50iter.pt.trace.json`
   - Also produced filtered trace `round_002_forward_50iter.filtered.pt.trace.json`
     (dropped 2 duplicate `cat=user_annotation` X events to resolve the known
     C500 `overlapping scope events` issue).
6. Canonical reference wall timing (triton_mha_001.py, 3 runs warmup 50 repeat 100) — `RETURN_CODE=0`
   - samples: `[0.179934, 0.167111, 0.165897]`, median `0.167111`

## Raw Samples and Medians

- reference_raw_samples_ms (v0 base.py): `[0.110857, 0.125705, 0.110812]`
- candidate_raw_samples_ms (v1 triton_mha_002.py): `[0.127422, 0.130439, 0.127777]`
- reference_median_ms (v0): `0.110857`
- candidate_median_ms (v1): `0.127777`
- improvement_pct vs base.py (v0): `-15.26290626663179`
- canonical (triton_mha_001.py) same-session samples: `[0.179934, 0.167111, 0.165897]`
- canonical_median_ms: `0.167111`
- improvement_pct vs canonical: `23.537648628755743`

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` |
| `triton_mha_002.py` (candidate) | `29e6b192bf778f0264fb7657c9a33b97819c406896a2ad86e1daf22f3c9ff0a1` |
| `triton_mha_001.py` (accepted reference / canonical) | `9fac12aa0298a970c208dbc6af7a602da4f34e43d44921b62aad571ca662c00b` |
| `decision_002.md` | `be804f497dcb6070e1a07d290b43c6c8acc65e3007d88657985026aa5640ac7e` |

## Profiler Summary

| Scope | Device us/call | Kernel count/call | Dominant kernels |
|---|---:|---:|---|
| baseline_base (v0) | 15.0989453125 | 2.0 | `flash_fwd_splitkv_kernel`, `flash_fwd_splitkv_combine_kernel` |
| candidate_triton_mha_002 (v1) | 67.727314453125 | 2.0 | `_mha_fwd_kernel` (1.0/call, 64.85 us/call), `transpose12_copy_64` (1.0/call, 2.88 us/call) |

Transpose copy reduction: the four `.contiguous()` copy kernels are gone. The
candidate now emits 1 fused `_mha_fwd_kernel` + 1 `transpose12_copy_64` (the
single unavoidable output reshape) = 2.0 kernels/call, down from 5.0 in round
001. Device time dropped from 79.70 to 67.73 us/call.

## Next Safe Action

Orchestrator records `accepted` terminal result and advances the canonical
pointer to `triton_mha_002.py`.
