# Round Status 000

## Phase

Phase 0 — baseline establishment (Verifier, task `mm_encoder_attention`, BI150 backend).

## Frozen Artifact Hash Verification

| Artifact | Expected SHA-256 | Actual SHA-256 | Match |
|---|---|---|---|
| `kernels/track1-triton/mm_encoder_attention/base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | pass |
| `kernels/track1-triton/mm_encoder_attention/bi150/baseline_adapter.py` | `c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f` | `c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f` | pass |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | pass |

All three frozen hashes match the project.md `measurement_fingerprint` inputs exactly.

## Runtime Fingerprint

- torch `2.7.1`, triton `3.1.0`, device `Iluvatar BI-V150`, capability `(7, 1)`.
- Bootstrap: `export COREX_VERSION=4.4.0 && . /usr/local/corex/enable`.

## Step Log

### Step 1 — Correctness (return code 0)

Command:
```bash
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/baseline_adapter.py --warmup 50 --repeat 100 --full-traceback
```
Output: `PASS accuracy; v0=0.148653 ms, v1=0.148315 ms, speedup=1.002x`. Exit code `0`.

### Step 2 — Baseline Benchmark (three interleaved raw samples)

Command (three independent invocations):
```bash
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/baseline_adapter.py --warmup 50 --repeat 100
```

| Invocation | v0 wall ms | v1 wall ms | Exit code |
|---:|---:|---:|---:|
| 1 | `0.150876` | `0.149624` | `0` |
| 2 | `0.151139` | `0.150183` | `0` |
| 3 | `0.151994` | `0.149352` | `0` |

v0 raw samples: `[0.150876, 0.151139, 0.151994]`.
v0 unrounded median: `0.151139` ms → baseline wall_time_ms.

### Step 3 — Baseline Profiler (return code 0)

Command:
```bash
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mm_encoder_attention/bi150/log/round_000_forward_50iter.pt.trace.json
```
Exit code `0`. Trace: `log/round_000_forward_50iter.pt.trace.json`.

Trace SHA-256: `140ce325b62c0ac03e08f1e8f9f9bbbe586ed382e18407d212c8d02ad985b94c`.

Summarize `baseline_base` (iterations 50): device_total_us `747.462890625`, device_us_per_call `14.9492578125`, kernel_count_total `43`, kernel_count_per_call `0.86`. Exit code `0`.

Summarize `candidate_baseline_adapter` (iterations 50): device_total_us `794.85400390625`, device_us_per_call `15.897080078125`, kernel_count_total `46`, kernel_count_per_call `0.92`. Exit code `0`.

Top kernel (both scopes, 100% of device time):
`void ixattnbkd::src::impl::MR::FlashAttnFwdF16Ixmma<128u, 128u, 16u, 64u, 64u, (CausalM_t)0, (AlibiMode_t)0, false, __half, false>(...)`.

## Next Safe Action

Write `rounds/report_000.md` (Result = baseline), update `state/verifier_context.md`, then report to Orchestrator.
