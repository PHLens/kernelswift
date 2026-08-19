# Round Status 001

- phase: `verifying` (complete)
- result: `accepted`
- completed_steps:
  - `frozen-hash-verification` (all pass)
  - `runtime-fingerprint-check` (pass)
  - `correctness gate` (PASS accuracy)
  - `independent numerical probe` (max abs diff pre=5.96e-08, post=0.0, comb=1.19e-07)
  - `authoritative interleaved timing` (3 pairs)
  - `targeted profiler` (forward 20/50, both scopes)
  - `reports written` (report_001.md, round_status_001.md, verifier_context.md)
- next_safe_action: `none — report to Orchestrator`

## Frozen artifact hashes (verified before measurement)

| Artifact | SHA-256 | Match |
|---|---|---|
| base.py | `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5` | pass |
| baseline_adapter.py | `ceebdc6185de4c980156a7833073678a0964fb7ccb5edf74b42be6156652eaed` | pass |
| triton_mhc_head_compute_mix_001.py | `a98b1b12593d858ca29c787afa939a3ae0061df4ec6b51aa9a0fe7fa43c6b473` | pass |
| auto_bench.py | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | pass |

## Runtime fingerprint

torch 2.7.1, triton 3.1.0, Iluvatar BI-V150 (7,1) — matches `project.md#runtime-fingerprint`.

## Correctness

`PASS accuracy; v0=1.429648 ms, v1=0.180503 ms, speedup=7.920x` (return code 0).
Independent probe max abs diff: `pre=5.96e-08`, `post=0.0`, `comb=1.19e-07`.

## Authoritative Interleaved Timing (3 pairs, warmup 50 / repeat 100)

- reference_raw_samples_ms: `[1.420944, 1.462009, 1.433128]`
- candidate_raw_samples_ms: `[0.180531, 0.183889, 0.188159]`
- reference_median_ms: `1.433128`
- candidate_median_ms: `0.183889`
- improvement_pct: `87.16869672492618`

## Profiler (forward 20/50)

- reference (`reference_baseline_adapter`): device `924.793 us/call`, `132.88 kernels/call`, device_ratio `0.6453`
- candidate (`candidate_triton_mhc_head_compute_mix_001`): device `13.879 us/call`, `1.12 kernels/call` (steady-state `1.0` fused kernel/call), device_ratio `0.0755`
- trace: `log/round_001_forward_50iter.pt.trace.json`, SHA256 `961335d11c644fa987f3c32f1d1be9e0f170b633070eea4f6b357afa83492b94`

## Result

`accepted` — improvement `87.17%` (≥ 5%), correctness and all guardrails pass.
