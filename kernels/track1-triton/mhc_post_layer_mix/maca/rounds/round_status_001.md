# Round Status 001

- phase: `verifying` (complete)
- round: `001`
- result: `accepted`

## Progress

| Step | Status | Command | Exit | Observation |
|---|---|---|---|---|
| identity hashes | done | sha256sum | 0 | candidate e54e5b2e...b944, decision 9f3795f5...da5a match coder_result |
| correctness | done | warmup 5 repeat 10 --full-traceback | 0 | PASS; v0=7.666721 ms, v1=0.247846 ms, speedup=30.933x |
| authoritative wall #1 | done | warmup 50 repeat 100 | 0 | v0=7.635280, v1=0.242602 |
| authoritative wall #2 | done | warmup 50 repeat 100 | 0 | v0=7.633116, v1=0.240464 |
| authoritative wall #3 | done | warmup 50 repeat 100 | 0 | v0=7.633507, v1=0.241083 |
| profiler | done | forward warmup 20 iter 50 | 0 | trace written |
| summarize trace | done | summarize_trace.py x2 scopes | 0 | baseline & candidate summarized (filtered copy) |

## Results

- reference_median_ms: `7.633507` (v0 `Model`)
- candidate_median_ms: `0.241083` (v1 `ModelNew`)
- improvement_pct: `96.841779` (speedup 31.663x)
- baseline scope: device 7560.89 us/call, 6.0 kernels/call
- candidate scope: device 168.56 us/call, 1.0 kernels/call (`_mhc_post_layer_mix_fused_kernel`)
- tf32 GEMM eliminated (0 occurrences in candidate scope)
- Result: `accepted` (correctness PASS + 96.84% >> 5% threshold)

## Next safe action

Terminal result `accepted`; hand off to Orchestrator for canonical pointer advance (`last_accepted_kernel = triton_mhc_001.py`).
