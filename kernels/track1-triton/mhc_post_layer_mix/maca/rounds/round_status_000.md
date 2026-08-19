# Round Status 000

- phase: `verifying` (complete)
- round: `000`
- result: `baseline` (Phase 0)
- verification_tier: `baseline`

## Progress

| Step | Status | Command | Exit | Observation |
|---|---|---|---|---|
| identity hashes | done | sha256sum | 0 | base/auto_bench/baseline_adapter all match frozen values |
| correctness | done | warmup 5 repeat 10 --full-traceback | 0 | PASS; v0=7.640548 ms, v1=7.640149 ms, speedup=1.000x |
| authoritative wall #1 | done | warmup 50 repeat 100 | 0 | v0=7.639543, v1=7.636774 |
| authoritative wall #2 | done | warmup 50 repeat 100 | 0 | v0=7.634985, v1=7.636353 |
| authoritative wall #3 | done | warmup 50 repeat 100 | 0 | v0=7.635598, v1=7.636740 |
| profiler | done | forward warmup 20 iter 50 | 0 | trace written |
| summarize trace | done | summarize_trace.py x2 scopes | 0 | baseline & candidate summarized (filtered copy) |

## Results

- reference_median_ms: `7.635598` (v0 `Model`)
- candidate_median_ms: `7.636740` (v1 `ModelNew`)
- improvement_pct: `-0.014956` (Phase 0 baseline)
- baseline scope: device 7559.20 us/call, 6.0 kernels/call, ratio 0.990
- candidate scope: device 7561.75 us/call, 6.0 kernels/call, ratio 0.990
- dominant kernel: `mcblas__Mck_tf32gemm_nt_64x64x128_4m4n1k_256t_fp32_fp32_tf32_sb_0_0` (~6071 us/call, ~80% device time)

## Next safe action

Terminal result `baseline`; hand off to Orchestrator for canonical pointer update.
