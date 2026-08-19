# Round Status 000

Phase 0 baseline verification (mhc_post_layer_mix, BI150).

## Verification Start

- phase: `verifying`
- current_round: `000`
- verification_tier: `baseline`
- started_at: `2026-08-18T18:00:00Z`

## Frozen File Hashes (before measurement)

| Artifact | SHA-256 | Match project.md |
|---|---|---|
| `kernels/track1-triton/mhc_post_layer_mix/base.py` | `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3` | pass |
| `kernels/track1-triton/mhc_post_layer_mix/bi150/baseline_adapter.py` | `66a3a2c31863d18c725a52ab57fd1b9f89fe655dd7bab7cb4da158b8130b5d07` | pass |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | pass |

Note: the adapter hash is the Orchestrator-repaired value (generation defect
`super(Model, ...)` fixed to `super(ModelNew, ...)`); prior frozen value
`ceaff44f4...` was superseded and reconciled in `project.md`.

## Runtime Fingerprint

- torch `2.7.1`, triton `3.1.0`, device `Iluvatar BI-V150`, capability `(7, 1)`.
- `target_profile_match: pass`.

## Correctness

```bash
python3 auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py \
  --v1_file kernels/track1-triton/mhc_post_layer_mix/bi150/baseline_adapter.py \
  --warmup 50 --repeat 100 --full-traceback
```

- Result: **PASS accuracy**; `v0=8.154829 ms, v1=8.184198 ms, speedup=0.996x`; return code `0`.

## Interleaved Wall Timing (three independent 50/100 invocations)

| Invocation | Reference wall ms (v0) | Candidate wall ms (v1) | Return code |
|---:|---:|---:|---:|
| 1 | `8.189047` | `8.185878` | `0` |
| 2 | `8.198384` | `8.193696` | `0` |
| 3 | `8.176003` | `8.104427` | `0` |

- reference_raw_samples_ms: `[8.189047, 8.198384, 8.176003]`
- reference_median_ms: `8.189047`

## Profiler (forward, warmup 20, iterations 50)

```bash
python3 auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py \
  --v1_file kernels/track1-triton/mhc_post_layer_mix/bi150/baseline_adapter.py \
  --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 \
  --profile-output kernels/track1-triton/mhc_post_layer_mix/bi150/log/round_000_forward_50iter.pt.trace.json
```

- Return code `0`. Trace SHA256 `8c21c97ddca24e78ebb0f4dd37e4aba65f7e942fbddfdfc76a65b7d406ea9b26`.

| Scope | Device us/call | Kernel count/call | Device ratio |
|---|---:|---:|---:|
| `baseline_base` | `7323.8471875` | `5.48` | `0.8943467032854981` |
| `candidate_baseline_adapter` | `7956.452177734375` | `5.96` | `0.971596838769441` |

## Frozen File Hashes (after measurement)

Identical to before-measurement values (no mutation).

## Next Safe Action

- Phase 0 baseline is valid. Report `Result=baseline` to Orchestrator with a
  `continue` stop recommendation.
