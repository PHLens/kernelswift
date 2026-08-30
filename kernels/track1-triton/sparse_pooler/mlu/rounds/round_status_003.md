# Round Status 003

## Verification start

- phase: `verifying`
- verifier: sole runtime owner (`measurement_exclusive=true`)
- candidate: `triton_sparse_pooler_003.py` (SHA-256 `3406f7c9a731e1fd7560ab95bf1d903fd4d6f8287c5880d9801e9d41e5ed7860`)
- accepted reference: `triton_sparse_pooler_001.py` (SHA-256 `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd`)
- decision: `rounds/decision_003.md` (SHA-256 `8f78d0425148e387ba82fc827012c63440e8d38edcdf19750a0e79825c8505bb`)
- base SHA-256: `ccccbbefadf1d697341451b542f17392acc8a2b9e4a3a41e50b2f9d58dbf61de`
- harness SHA-256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- measurement fingerprint: `a0208c7da7e371d45c88f82ebddd3850d01669aa5d912f31db9234a7a56ebab7`
- runtime fingerprint: triton 3.2.0, torch_mlu 1.32.0+torch2.11.0, MLU driver 6.5.49, MLU590-H8 capability (5, 0)

## Pre-flight checks

- artifact hashes match `coder_result_003.md` and `project.md`
- runtime fingerprint matches `project.md#runtime-fingerprint`
- interpreter `/projs/framework/lipenghui/venv/pytorch_main/bin/python3` (python3 on PATH) has triton 3.2.0, torch_mlu 1.32.0+torch2.11.0, MLU590-H8 cap 5.0
- accepted reference is `triton_sparse_pooler_001.py` (Round 001 canonical); Round 002's rejected candidate is never the comparison source

## Safe-step plan

1. Correctness gate: `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_003.py --warmup 50 --repeat 100`. Required: `PASS accuracy` and all guardrails pass. On a local implementation defect, return `implementation-repair-required` with the failing guardrail and minimal diagnosis.
2. v2 screening: two short interleaved pairs (warmup 5, repeat 5) of accepted reference vs candidate. `screened-out` only if BOTH pairs are >=10% slower than the accepted reference; otherwise proceed to authoritative timing.
3. Authoritative 3-pair wall timing (warmup 50, repeat 100, interleaved accepted/candidate). Persist all 6 raw samples; compare UNROUNDED medians.
4. Level 1 profiler: `--v1_file sparse_pooler/triton_sparse_pooler_003.py --profile-reference-file sparse_pooler/triton_sparse_pooler_001.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output sparse_pooler/log/round_003_forward_50iter.pt.trace.json`. Use `gpu_user_annotation`-only fallback scoping.
5. Write `rounds/report_003.md` (Evaluation Contract mirror with 4 mechanism observables, hypothesis verdict, improvement_pct, stop recommendation, exact reproduction commands) and update `state/verifier_state.md`.

## After correctness gate

- command: `python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_003.py --warmup 50 --repeat 100`
- exit code: 0
- stdout: `PASS accuracy; v0=0.913044 ms, v1=0.840942 ms, speedup=1.086x`
- correctness verdict: pass (all outputs match within atol=1e-2 rtol=1e-2 equal_nan=True)
- guardrails: pass (output structure, numerical semantics, device/stream preservation, dense/GELU/LayerNorm library ops preserved, ModelNew signature unchanged, load_state_dict compatibility confirmed, num_warps=1 used, kernel_count_per_call decreases per Coder result)
- candidate SHA-256 after correctness: `3406f7c9a731e1fd7560ab95bf1d903fd4d6f8287c5880d9801e9d41e5ed7860` (unchanged; Verifier does not edit the candidate)

## After v2 screening

- protocol: two short interleaved accepted-reference/candidate pairs (warmup 5, repeat 5)
- pair 1 reference: `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_001.py --warmup 5 --repeat 5` → exit 0, `PASS accuracy; v0=0.918679 ms, v1=0.623747 ms, speedup=1.473x`
- pair 1 candidate: `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_003.py --warmup 5 --repeat 5` → exit 0, `PASS accuracy; v0=0.895909 ms, v1=0.835409 ms, speedup=1.072x`
- pair 2 reference: `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_001.py --warmup 5 --repeat 5` → exit 0, `PASS accuracy; v0=1.108457 ms, v1=0.658215 ms, speedup=1.684x`
- pair 2 candidate: `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_003.py --warmup 5 --repeat 5` → exit 0, `PASS accuracy; v0=0.970991 ms, v1=0.869966 ms, speedup=1.116x`

### Screening computation

- pair 1: candidate is `(0.835409 - 0.623747) / 0.623747 * 100 = 33.90%` slower than accepted reference
- pair 2: candidate is `(0.869966 - 0.658215) / 0.658215 * 100 = 32.16%` slower than accepted reference
- v2 screening verdict: BOTH pairs are >=10% slower than the accepted reference → `screened-out`
- authoritative 3-pair timing is NOT run (v2 protocol: proceed to authoritative timing only when the candidate is NOT screened-out)

## After Level 1 profiler

- command: `python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_003.py --profile --profile-reference-file sparse_pooler/triton_sparse_pooler_001.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output sparse_pooler/log/round_003_forward_50iter.pt.trace.json`
- exit code: 0
- stdout: `PASS accuracy; v0=0.888508 ms, v1=0.822277 ms, speedup=1.081x` (diagnostic only)
- trace: `/projs/framework/lipenghui/projects/kernelswift/sparse_pooler/log/round_003_forward_50iter.pt.trace.json` (1893209 bytes)
- `summarize_trace.py` hits `overlapping scope events` on both `reference_triton_sparse_pooler_001` and `candidate_triton_sparse_pooler_003` scopes (each scope has 2 overlapping events: `user_annotation` on CPU + `gpu_user_annotation` on GPU). Used the gpu_user_annotation-only fallback scoping documented in `state/verifier_context.md`.

### Reference scope (triton_sparse_pooler_001, 50 iterations)

- device_total_us: 10615.88
- device_us_per_call: 212.32
- kernel_count_total: 250
- kernel_count_per_call: 5.0
- top kernels (us/call):
  - `_sparse_pooler_max_kernel`: 99.52
  - `MLUFusedMatMulGepm` (decoder matmul 768->30522): 91.92
  - `MLUFusedMatMulGepdot` (dense matmul 768->768): 8.83
  - `layerNormForwardKernel`: 7.48
  - `MLUBlockKernel3StagePipelineGeluHighAccCubic` (GELU): 4.57

### Candidate scope (triton_sparse_pooler_003, 50 iterations)

- device_total_us: 19647.08
- device_us_per_call: 392.94
- kernel_count_total: 200
- kernel_count_per_call: 4.0
- top kernels (us/call):
  - `_sparse_pooler_fused_matmul_max_kernel` (new fused matmul+bias+relu+log1p+max): 373.31
  - `MLUFusedMatMulGepdot` (dense matmul 768->768): 7.83
  - `layerNormForwardKernel`: 7.24
  - `MLUBlockKernel3StagePipelineGeluHighAccCubic` (GELU): 4.56
  - `MLUFusedMatMulGepm` (decoder matmul): ELIMINATED (0 us/call, 0 count)

### Observable summary

- `decoder_matmul_kernel_count_per_call`: 1 -> 0 (decrease; confirmed)
- `total_kernel_count_per_call`: 5 -> 4 (decrease; confirmed)
- `device_us_per_call`: 212.32 -> 392.94 (INCREASED by 180.63 us/call; falsified — expected decrease from ~210.12)
- `fused_kernel_us_per_call`: new fused kernel 373.31 us/call vs 189.09 us/call combined cost of the two replaced kernels (99.52 + 91.92); the new fused kernel is 184.22 us/call SLOWER (falsified — expected less than 189.09)

## Verification end

- terminal classification: `no-improvement`
- hypothesis verdict: `falsified`
- reason: candidate is correct and conforms to the decision, but is screened-out in v2 screening (both pairs >=10% slower than accepted reference: +33.90% and +32.16% slower). The authoritative 3-pair timing was NOT run per the v2 screening protocol. Profiler evidence confirms the hypothesis is falsified: the new fused `_sparse_pooler_fused_matmul_max_kernel` at 373.31 us/call is 184.22 us/call slower than the 189.09 us/call combined cost of the two kernels it replaced, because `tl.dot` with small M (BLOCK_M=32, actual seq_len 18-25) is inefficient on MLU590-H8. Device time increased from 212.32 to 392.94 us/call.
- next safe action: Orchestrator applies the `no-improvement` terminal result; performance_miss_streak becomes 2.
