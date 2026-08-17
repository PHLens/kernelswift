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

## Round 003 (completed)

- phase: verification complete
- terminal result: `no-improvement`
- hypothesis verdict: `falsified`
- improvement_pct: `not computed (candidate screened-out in v2 screening; both pairs >=10% slower than accepted reference)`
- stop recommendation: `continue`
- report: `rounds/report_003.md`
- round status: `rounds/round_status_003.md`
- candidate: `triton_sparse_pooler_003.py` (SHA-256 `3406f7c9a731e1fd7560ab95bf1d903fd4d6f8287c5880d9801e9d41e5ed7860`)
- accepted reference: `triton_sparse_pooler_001.py` (SHA-256 `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd`)
- decision: `rounds/decision_003.md` (SHA-256 `8f78d0425148e387ba82fc827012c63440e8d38edcdf19750a0e79825c8505bb`)
- change_family: `kernel-matmul-fusion`

## Key measurements (Round 003)

- correctness: `PASS accuracy` on first attempt (warmup 50, repeat 100); v0=0.913044 ms, v1=0.840942 ms, speedup=1.086x
- v2 screening (warmup 5, repeat 5, two interleaved pairs):
  - pair 1 reference (triton_sparse_pooler_001): 0.623747 ms
  - pair 1 candidate (triton_sparse_pooler_003): 0.835409 ms (+33.90% slower)
  - pair 2 reference (triton_sparse_pooler_001): 0.658215 ms
  - pair 2 candidate (triton_sparse_pooler_003): 0.869966 ms (+32.16% slower)
  - verdict: `screened-out` (both pairs >=10% slower); authoritative 3-pair timing NOT run
- profiler (Level 1, 50 iterations, gpu_user_annotation-only fallback):
  - reference_triton_sparse_pooler_001: 212.32 us/call device, 5.0 kernels/call, ratio 0.3312 (diagnostic)
  - candidate_triton_sparse_pooler_003: 392.94 us/call device, 4.0 kernels/call, ratio 0.4608 (diagnostic)
  - new fused `_sparse_pooler_fused_matmul_max_kernel`: 373.31 us/call (184.22 us/call SLOWER than the 191.44 us/call combined cost of the two kernels it replaced: 99.52 + 91.92)
  - library `MLUFusedMatMulGepm` decoder matmul: ELIMINATED (1.0 -> 0.0 count/call)
  - dense matmul 8.83->7.83, LayerNorm 7.48->7.24, GELU 4.57->4.56 us/call (unchanged within noise)

## Resume context

- Correctness passed on first attempt; no Verifier-to-Coder repair was needed. (Coder's own attempt ledger records one internal tile-size accommodation before handoff: BLOCK_K=128->64, BLOCK_V=1024->512, within the decision's allowed probe space.)
- v2 screening protocol: after correctness passes, run exactly two short interleaved accepted-reference/candidate pairs (warmup 5, repeat 5). A correct candidate is `screened-out` only when BOTH pairs are at least 10% slower than the accepted reference. Otherwise proceed to authoritative 3-pair timing. Round 003's candidate was screened-out (both pairs ~33% slower), so authoritative timing was not run.
- `summarize_trace.py` hits "overlapping scope events" on this harness's traces; use the gpu_user_annotation-only fallback (documented in report_001.md, report_002.md, and verifier_context.md).
- Verifier does not update `last_accepted_kernel`; Orchestrator applies the accepted transition.
- `kernel-matmul-fusion` via `tl.dot` with small M (BLOCK_M=32, actual seq_len 18-25) is falsified on MLU590-H8: the new fused kernel is 184.22 us/call slower than the two kernels it replaced. `tl.dot` is not competitive with the library `MLUFusedMatMulGepm` decoder matmul at this M size on this runtime.
- Accepted Round 001 reference device profile: `_sparse_pooler_max_kernel` 99.52, `MLUFusedMatMulGepm` 91.92, dense matmul 8.83, LayerNorm 7.48, GELU 4.57 us/call. The two largest kernels together account for 191.44 us/call (90% of device time).
- performance_miss_streak becomes 2 after this round (Orchestrator applies).

## Round 004 (completed)

- phase: verification complete
- terminal result: `accepted`
- hypothesis verdict: `confirmed`
- improvement_pct: `5.788494443244682` (unrounded; 3-pair median)
- stop recommendation: `continue`
- report: `rounds/report_004.md`
- round status: `rounds/round_status_004.md`
- candidate: `triton_sparse_pooler_004.py` (SHA-256 `81cdea2b958c288e1382aef0b30cfc6dffb544c55a0e44825fab51b53cac7842`)
- accepted reference: `triton_sparse_pooler_001.py` (SHA-256 `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd`)
- decision: `rounds/decision_004.md` (SHA-256 `dc33e45ee2c95319608bc08f9ed8a5a3e3ae0882305f52eb07f0a449ea33f111`)
- change_family: `host-allocation-reuse`

## Key measurements (Round 004)

- correctness: `PASS accuracy` on first attempt (warmup 50, repeat 100); v0=0.888383 ms, v1=0.563431 ms, speedup=1.577x
- v2 screening (warmup 5, repeat 5, two interleaved pairs):
  - pair 1 reference (triton_sparse_pooler_001): 0.611090 ms
  - pair 1 candidate (triton_sparse_pooler_004): 0.554890 ms (-9.19% faster)
  - pair 2 reference (triton_sparse_pooler_001): 0.622712 ms
  - pair 2 candidate (triton_sparse_pooler_004): 0.568455 ms (-8.71% faster)
  - verdict: `NOT screened-out` (both pairs faster than accepted reference); authoritative 3-pair timing run
- authoritative 3-pair timing (warmup 50, repeat 100):
  - reference raw samples ms: `[0.596537, 0.601970, 0.603689]` (median 0.601970)
  - candidate raw samples ms: `[0.567125, 0.570159, 0.561303]` (median 0.567125)
  - improvement_pct: `5.788494443244682` (>= 5.0 adoption threshold)
- profiler (Level 1, 50 iterations, gpu_user_annotation-only fallback):
  - reference_triton_sparse_pooler_001: 208.02 us/call device, 5.0 kernels/call, ratio 0.3457 (diagnostic)
  - candidate_triton_sparse_pooler_004: 208.32 us/call device, 5.0 kernels/call, ratio 0.3672 (diagnostic)
  - `_sparse_pooler_max_kernel`: 98.76 -> 98.75 us/call (unchanged within noise; kernel body byte-identical)
  - `MLUFusedMatMulGepm` decoder matmul: 89.37 -> 89.63 us/call (unchanged within noise)
  - dense matmul 8.32->8.37, LayerNorm 7.19->7.19, GELU 4.38->4.39 us/call (unchanged within noise)
  - `aten::empty` count: reference 200 -> candidate 150 over 50 iterations (output buffer allocation eliminated by `_out_cache`)

## Resume context (Round 004)

- Correctness passed on first attempt; no Verifier-to-Coder repair was needed (0 of 2 repairs used).
- The `fast_libentry` wrapper and `_out_cache` output buffer are host-side changes; the device side is untouched. All 4 mechanism observables confirmed:
  - `fused_kernel_us_per_call`: 98.76 -> 98.75 (unchanged within noise) — confirmed
  - `device_us_per_call`: 208.02 -> 208.32 (+0.30, unchanged within noise) — confirmed
  - `kernel_count_per_call`: 5.0 -> 5.0 (exactly) — confirmed
  - `output_allocations_per_call`: 1 -> 0 on steady-state cache hits (200 -> 150 `aten::empty` over 50 iter) — confirmed
- Wall saving: 34.845 us/call; host saving: 35.145 us/call; device delta: +0.30 us/call (within noise).
- The class-body `globals()` trick (from `flexattention/triton_flexattention_003.py`) successfully injected the `fast_libentry`-wrapped kernel into module globals via the harness AST loader.
- `_out_cache` is a plain Python attribute (NOT registered via `register_buffer`/`register_parameter`); `load_state_dict` compatibility maintained.
- performance_miss_streak resets to 0 after this round (Orchestrator applies).
- Accepted Round 004 candidate device profile: `_sparse_pooler_max_kernel` 98.75, `MLUFusedMatMulGepm` 89.63, dense matmul 8.37, LayerNorm 7.19, GELU 4.39 us/call. The two largest kernels together account for 188.38 us/call (90.5% of device time).
- Cumulative improvement from baseline to Round 004: (0.909974 - 0.567125) / 0.909974 * 100 = 37.67% wall time.
- Candidate device ratio is 0.367 (diagnostic, mixed). Wall time 567.13 us/call, device 208.32 us/call, so roughly 359 us/call (~63%) is now host-side (launcher residue, wrapper, harness-fixed cost).
