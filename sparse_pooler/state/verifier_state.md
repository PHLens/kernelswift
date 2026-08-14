# Verifier State

Concise runtime facts and resume context for the current report.
No canonical-state claims.

## Round 001 (completed)

- phase: verification complete
- terminal result: `accepted`
- hypothesis verdict: `partially-confirmed`
- improvement_pct: `33.38529961673036` (unrounded)
- stop recommendation: `continue`
- report: `rounds/report_001.md`
- round status: `rounds/round_status_001.md`
- candidate: `triton_sparse_pooler_001.py` (SHA-256 `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd`)
- accepted reference: `baseline_adapter.py` (SHA-256 `d7e69ed4b66a4193a4475cc307fc0929cef807f875785652df6cc36fb2c487e5`)

## Round 002 (completed)

- phase: verification complete
- terminal result: `no-improvement`
- hypothesis verdict: `falsified`
- improvement_pct: `0.6531019145727505` (unrounded)
- stop recommendation: `continue`
- report: `rounds/report_002.md`
- round status: `rounds/round_status_002.md`
- candidate: `triton_sparse_pooler_002.py` (SHA-256 `62dc853db5423cb5d99ad53433f3fb35919abe901a64d6e3acb3d815ac678248`)
- accepted reference: `triton_sparse_pooler_001.py` (SHA-256 `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd`)

## Key measurements (Round 002)

- reference_median_ms: 0.625936 (triton_sparse_pooler_001, 3-pair median)
- candidate_median_ms: 0.621848 (triton_sparse_pooler_002, 3-pair median)
- reference device_us_per_call: 209.50 (5 kernels/call)
- candidate device_us_per_call: 213.02 (5 kernels/call)
- candidate device_ratio: 0.3425 (mixed)
- fused `_sparse_pooler_max_kernel`: reference 99.71 -> candidate 102.42 us/call (regressed by 2.71 us/call with BLOCK_V=2048)

## Resume context

- Correctness passed on first attempt; no Coder repair was needed.
- `summarize_trace.py` hits "overlapping scope events" on this harness's traces; use the gpu_user_annotation-only fallback (documented in report_001.md and report_002.md).
- Verifier does not update `last_accepted_kernel`; Orchestrator applies the accepted transition.
- BLOCK_V=2048 regressed device time vs BLOCK_V=1024. BLOCK_V=1024 remains the best-known tiling parameter for this kernel.
- The fused kernel remains slower than the 6 library kernels it replaced (102.42 vs 67.87 us/call combined); the 34.55 us/call device regression persists.
- performance_miss_streak becomes 1 after this round (Orchestrator applies).
