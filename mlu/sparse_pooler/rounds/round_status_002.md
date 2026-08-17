# Round Status 002

phase: verifying

## Verification start

- timestamp: 2026-08-14T15:05:00Z
- phase: verifying
- current_round: 002
- accepted_reference: triton_sparse_pooler_001.py (SHA-256 `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd`)
- candidate: triton_sparse_pooler_002.py (SHA-256 `62dc853db5423cb5d99ad53433f3fb35919abe901a64d6e3acb3d815ac678248`)
- decision: rounds/decision_002.md (SHA-256 `0d39de9e280f6ffa2cc3d1d3322d393fa400eb8f405b7e7ee3ceb3ef845b3dd4`)
- coder_result: rounds/coder_result_002.md (SHA-256 `ac308dddaee3314cb15a7156313201b3a5ec75b3bff61552c79ebb47d9b52a2e`)
- base: base.py (SHA-256 `ccccbbefadf1d697341451b542f17392acc8a2b9e4a3a41e50b2f9d58dbf61de`)
- harness: auto_bench.py (SHA-256 `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`)
- runtime_fingerprint: project.md#runtime-fingerprint
- measurement_fingerprint: a0208c7da7e371d45c88f82ebddd3850d01669aa5d912f31db9234a7a56ebab7

## Artifacts read

- team-state.md
- rounds/decision_002.md
- rounds/coder_result_002.md
- triton_sparse_pooler_002.py (candidate)
- project.md
- state/verifier_state.md
- rounds/report_001.md (last_accepted_report)
- triton_sparse_pooler_001.py (last_accepted_kernel)
- base.py
- references/invariants.md
- references/bottleneck-judgment.md

## Decision summary

Kernel-only tuning. Change BLOCK_V from 1024 to 2048 in the fused `_sparse_pooler_max_kernel` dispatch. Grid drops from (4,30)=120 programs to (4,15)=60 programs. Expected wall improvement 7.0%. Adoption threshold 5.0%. Change scope: kernel. Hypothesis H-002.

## Next safe action

Run correctness gate (before timing). Correctness command:

```bash
/projs/framework/lipenghui/venv/pytorch_main/bin/python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/triton_sparse_pooler_002.py \
  --warmup 50 --repeat 100
```

## Correctness gate

- timestamp: 2026-08-14T15:08:00Z
- command: see above
- exit code: 0
- result: `PASS accuracy; v0=1.068945 ms, v1=0.670753 ms, speedup=1.594x` (1 passed, 0 failed)
- verdict: pass
- next safe action: run authoritative wall timing (3 interleaved pairs: reference=triton_sparse_pooler_001.py, candidate=triton_sparse_pooler_002.py)

## Authoritative wall timing

- warmup: 50
- repeat: 100
- order: interleaved accepted-reference/candidate
- accepted reference: triton_sparse_pooler_001.py
- candidate: triton_sparse_pooler_002.py

### Pair 1

| Role | v0 ms | v1 ms | speedup | exit |
|---|---:|---:|---:|---:|
| reference (triton_sparse_pooler_001) | 0.975735 | 0.690104 | 1.414x | 0 |
| candidate (triton_sparse_pooler_002) | 0.934121 | 0.621848 | 1.502x | 0 |

- reference_raw_ms: [0.690104]
- candidate_raw_ms: [0.621848]
- next safe action: run pair 2

### Pair 2

| Role | v0 ms | v1 ms | speedup | exit |
|---|---:|---:|---:|---:|
| reference (triton_sparse_pooler_001) | 0.932037 | 0.625936 | 1.489x | 0 |
| candidate (triton_sparse_pooler_002) | 0.935818 | 0.631596 | 1.482x | 0 |

- reference_raw_ms: [0.690104, 0.625936]
- candidate_raw_ms: [0.621848, 0.631596]
- next safe action: run pair 3

### Pair 3

| Role | v0 ms | v1 ms | speedup | exit |
|---|---:|---:|---:|---:|
| reference (triton_sparse_pooler_001) | 0.890085 | 0.604951 | 1.471x | 0 |
| candidate (triton_sparse_pooler_002) | 0.906601 | 0.613131 | 1.479x | 0 |

- reference_raw_ms: [0.690104, 0.625936, 0.604951]
- candidate_raw_ms: [0.621848, 0.631596, 0.613131]
- reference_median_ms: 0.625936
- candidate_median_ms: 0.621848
- improvement_pct: 0.6531 (below 5% adoption threshold)
- next safe action: run Level 1 profiler to collect mechanism observables

## Level 1 profiler

- command: `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_002.py --profile --profile-reference-file sparse_pooler/triton_sparse_pooler_001.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output sparse_pooler/log/round_002_forward_50iter.pt.trace.json`
- exit code: 0
- diagnostic wall: `PASS accuracy; v0=0.894074 ms, v1=0.609396 ms, speedup=1.467x`
- trace: log/round_002_forward_50iter.pt.trace.json
- summarize_trace.py: failed with `overlapping scope events` (each scope has 1 user_annotation + 1 gpu_user_annotation that overlap)
- fallback: custom Python summarizer using only gpu_user_annotation device-side intervals

### Scope summaries

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| reference_triton_sparse_pooler_001 | 10475.00 | 209.50 | 250 | 5.0 | 0.625936 | 0.3347 |
| candidate_triton_sparse_pooler_002 | 10651.25 | 213.02 | 250 | 5.0 | 0.621848 | 0.3425 |

### Top kernels (reference_triton_sparse_pooler_001)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _sparse_pooler_max_kernel | 50 | 1.00 | 4985.56 | 99.71 |
| MLUFusedMatMulGepm (decoder matmul) | 50 | 1.00 | 4480.56 | 89.61 |
| MLUFusedMatMulGepdot (dense matmul) | 50 | 1.00 | 420.52 | 8.41 |
| layerNormForwardKernel (LayerNorm) | 50 | 1.00 | 360.44 | 7.21 |
| MLUBlockKernel3StagePipelineGeluHighAccCubic (GELU) | 50 | 1.00 | 227.92 | 4.56 |

### Top kernels (candidate_triton_sparse_pooler_002)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _sparse_pooler_max_kernel | 50 | 1.00 | 5121.04 | 102.42 |
| MLUFusedMatMulGepm (decoder matmul) | 50 | 1.00 | 4516.72 | 90.33 |
| MLUFusedMatMulGepdot (dense matmul) | 50 | 1.00 | 426.64 | 8.53 |
| layerNormForwardKernel (LayerNorm) | 50 | 1.00 | 359.84 | 7.20 |
| MLUBlockKernel3StagePipelineGeluHighAccCubic (GELU) | 50 | 1.00 | 227.00 | 4.54 |

## Mechanism observables summary

| Observable | Expectation | Observation | Verdict |
|---|---|---|---|
| fused_kernel_us_per_call | decrease from 98.73 | reference 99.71 -> candidate 102.42 (increased by 2.71 us/call) | falsified |
| device_us_per_call | decrease from 210.12 | reference 209.50 -> candidate 213.02 (increased by 3.52 us/call) | falsified |
| fused_kernel_grid_programs | decrease from 120 to 60 | reference grid (4,30)=120 -> candidate grid (4,15)=60 (confirmed by code inspection: BLOCK_V 1024 -> 2048) | confirmed |

## Verification end

- terminal result: no-improvement
- hypothesis verdict: falsified
- improvement_pct: 0.6531 (below 5% adoption threshold)
- stop recommendation: continue
- evidence: candidate is marginally faster in wall time (0.65% < 5% threshold) but device time increased; the fused kernel got slower with BLOCK_V=2048, falsifying the hypothesis that halving grid programs reduces per-program overhead
- next owner: Orchestrator (apply terminal transition)
