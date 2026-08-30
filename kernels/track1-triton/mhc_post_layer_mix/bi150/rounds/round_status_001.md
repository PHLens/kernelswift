# Round Status 001

Round 001 verification (mhc_post_layer_mix, BI150).

## Verification Start

- phase: `verifying`
- current_round: `001`
- verification_tier: `authoritative`
- started_at: `2026-08-18T19:00:00Z`

## Frozen Artifact Hashes (before measurement)

| Artifact | SHA-256 | Match |
|---|---|---|
| candidate `triton_mhc_post_layer_mix_001.py` | `08a9d59f17ffa80224943b19bdcce390d908ca8ba15bf2e06ae469f45787d9fb` | pass |
| canonical `baseline_adapter.py` | `66a3a2c31863d18c725a52ab57fd1b9f89fe655dd7bab7cb4da158b8130b5d07` | pass |
| `decision_001.md` | `335389df2498f37fb9f2c5c7ebc10986ab4edf555d939525413900e0e885ecfc` | pass |
| `base.py` | `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3` | pass |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | pass |

## Correctness (vs base.py)

```bash
python3 auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py \
  --v1_file kernels/track1-triton/mhc_post_layer_mix/bi150/triton_mhc_post_layer_mix_001.py \
  --warmup 50 --repeat 100 --full-traceback
```

- Result: **PASS accuracy**; `v0=8.041220 ms, v1=6.469838 ms, speedup=1.243x`; return code `0`.

Independent numerical probe (harness AST loader, same inputs, direct fp32
reference): `allclose(base, candidate)=True`, `allclose(base, direct_fp32_ref)=True`,
shape `[2,4096,4,1280]` bf16, `max_abs_diff=0.03125`, `mean_abs_diff=1.73e-8`.

## Authoritative Wall Timing (three interleaved 50/100 pairs vs baseline_adapter)

Baseline wrapper `/tmp/mplm_baseline_model_001.py` (class `Model` + `super(Model, self)`)
SHA256 `654dc9350d6d893761e7e499cd91f41207ff0679f256a70390ca4568960dcb9e` (deleted after run).

| Invocation | Reference wall ms (v0) | Candidate wall ms (v1) | Return code |
|---:|---:|---:|---:|
| 1 | `8.022035` | `6.423922` | `0` |
| 2 | `8.152739` | `6.427432` | `0` |
| 3 | `8.043548` | `6.513436` | `0` |

- reference_raw_samples_ms: `[8.022035, 8.152739, 8.043548]`; median `8.043548`
- candidate_raw_samples_ms: `[6.423922, 6.427432, 6.513436]`; median `6.427432`
- improvement_pct: `20.09`

## Targeted Profiler (forward, warmup 20, iterations 50, dual scope)

```bash
python3 auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py \
  --v1_file kernels/track1-triton/mhc_post_layer_mix/bi150/triton_mhc_post_layer_mix_001.py \
  --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 \
  --profile-reference-file kernels/track1-triton/mhc_post_layer_mix/bi150/baseline_adapter.py \
  --profile-output kernels/track1-triton/mhc_post_layer_mix/bi150/log/round_001_forward_50iter.pt.trace.json
```

- Return code `0`. Trace SHA256 `8543b6ea4418b53292632e5e4321d07e96e881c320072edfde1d53a79115720a`.
- `reference_baseline_adapter` scope (summarizer return `0`):
  - device_us_per_call `7516.836240234375`, kernel_count_per_call `5.66`.
- `candidate_triton_mhc_post_layer_mix_001` scope: the unmodified summarizer
  returned `overlapping scope events` (code `2`) because the Triton launch inside
  `forward` produced two nested same-name `record_function` markers whose
  intervals overlap. Manual extraction over the outermost/innermost scope
  interval yielded the candidate kernel evidence below.

### Candidate kernel evidence (manual scope extraction, iteration 50)

| Kernel | Count | Us/call |
|---|---:|---:|
| `gemm_tcu_h` (GEMM, unchanged) | `49` | `5183.49` |
| `_fused_tail_kernel` (Triton fused tail) | `50` | `496.18` |
| `direct_copy_kernel_cuda` (residual.float() cast) | `49` | `442.86` |

- candidate device_us_per_call ≈ `6122.54`, kernel_count_per_call ≈ `2.96`.

## Frozen Artifact Hashes (after measurement)

Identical to before-measurement values (no mutation).

## Next Safe Action

- Report `Result=accepted` to Orchestrator (improvement 20.09% ≥ 5%, all
  guardrails pass).
