# Report 000

Result: baseline

## Identity

- Round: `000`
- Candidate: `baseline_adapter.py` @`7d4a79ae96328fc03a4489710f68b7f639ddea9cbd5c0f7bb45e1cec5472061b`
- Accepted reference: `../../base.py` @`02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint` (match)
- verification_tier: `baseline`

## Correctness and Guardrails

| Check | Observation | Verdict |
|---|---|---|
| correctness | `PASS accuracy` in all 3 timing pairs | pass |
| runtime bootstrap | torch_gcu/triton_gcu matched fingerprint | pass |
| immutable base | sha256 unchanged | pass |

## Interleaved Wall Timing

- warmup 50 / repeat 100 / seed 42 / interleaved pairs

| Invocation | Reference ms | Candidate ms | speedup |
|---:|---:|---:|---:|
| 1 | 2.437778 | 2.427073 | 1.004x |
| 2 | 2.342548 | 2.340553 | 1.001x |
| 3 | 2.342042 | 2.336651 | 1.002x |

Baseline reference median ≈ 2.342 ms (identity ~1.00x).

## Profiler Evidence (census, from preflight forward-mode profile)

- runtime_launch_count_per_call: **78.0** `topsLaunchKernel`/call (launch-bound)
- dominant aten ops: mul (66/call), empty (70/call), as_strided, add, sub, sqrt, sin, cos, rand, uniform, stack, cat, reshape, expand, contiguous
- device_time_available: false (GCU launch-only trace)

## evidence_for_next_round

- base is launch-bound (78 launches/call) — the fused_moe class, fusion wins.
- epoch-1 fused only rot_vec_mul (saved 1 launch) → 0.95x (no real gain).
- preflight: full fusion (quaternion→R + rot_vec_mul + translation + mask into a
  single kernel, host generates only u1/u2/u3/T) reached ~1.59x (correctness 4.77e-7).
- Random numbers MUST stay host-generated; sequence/order must match base exactly.

## Stop Recommendation

- recommendation: `continue`
