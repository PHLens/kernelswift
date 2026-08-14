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

## Key measurements

- reference_median_ms: 0.910847 (baseline_adapter, 3-pair median)
- candidate_median_ms: 0.606758 (triton_sparse_pooler_001, 3-pair median)
- reference device_us_per_call: 179.33 (10 kernels/call)
- candidate device_us_per_call: 210.12 (5 kernels/call)
- candidate device_ratio: 0.3463 (mixed)
- fused `_sparse_pooler_max_kernel`: 98.73 us/call (slower than the 67.87 us/call combined cost of the 6 baseline kernels it replaced)

## Resume context

- Correctness passed on first attempt; no Coder repair was needed.
- `summarize_trace.py` hits "overlapping scope events" on this harness's traces; use the gpu_user_annotation-only fallback (documented in report_001.md).
- Verifier does not update `last_accepted_kernel`; Orchestrator applies the accepted transition.
