# Report 000 — Phase 0 Baseline Establishment

- round: `000`
- result: `baseline`
- phase: Phase 0 (baseline establishment)
- decision: baseline established (no candidate comparison; this is the reference)

## Candidate and Reference

- base (reference, v0): `../base.py`
- baseline_adapter (v1): `baseline_adapter.py`
- Both implement identical semantics (baseline adapter is a `Model`→`ModelNew` rename of base).

## Source Hashes

| Artifact | SHA-256 |
|---|---|
| `../base.py` | `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3` |
| `baseline_adapter.py` | `a4f0aa8ac2d59c57059223b1710d20718af1b0f892cd7c373174e531c927133e` |

## Correctness

- reproduction command (warmup 50 / repeat 100): `PASS accuracy`
- v0 vs v1 outputs compared within harness default atol=1e-2 / rtol=1e-2: identical within bound.
- Note: a prior baseline adapter defect (`super(Model, self)` NameError) was repaired by
  Orchestrator (regeneration); recorded in `incident_000_2026-08-18T15-44-24Z.md`.

## Benchmark (authoritative timing)

| Metric | v0 (base.py) | v1 (baseline_adapter.py) |
|---|---:|---:|
| median wall_time_ms | 3.212215 | 3.206765 |
| speedup (v0/v1) | — | 1.002x |

- warmup: 50, repeat: 100, device: npu:0, interpreter: /usr/local/python3.11.15/bin/python3
- Both runs within noise of each other (as expected: identical code).

## Profiler Evidence (CANN per-scope)

Profiled forward mode, 20 warmup / 50 iterations, separate CANN scope per target.

### reference_baseline_adapter
- iterations: 50
- device_total_us: 154747.5
- device_us_per_call: 3094.95
- kernel_count_per_call: 6.0
- device_ratio: 0.9635 (vs wall 3.212215 ms)

Top kernels (by total_us):
| Kernel | count/call | us/call | total_us |
|---|---:|---:|---:|
| aclnnBatchMatMul_BatchMatMulNd_BatchMatMulV2 | 1.0 | 1111.40 | 55569.86 |
| aclnnAdd_AddAiCore_Add | 1.0 | 892.11 | 44605.40 |
| aclnnInplaceCopy_CastAiCore_Cast | 3.0 | 801.14 | 40057.16 |
| aclnnMul_MulAiCore_Mul | 1.0 | 290.30 | 14515.08 |

### candidate_baseline_adapter
- iterations: 50
- device_total_us: 154142.62
- device_us_per_call: 3082.85
- kernel_count_per_call: 6.0
- device_ratio: 0.9614 (vs wall 3.206765 ms)

Top kernels (by total_us):
| Kernel | count/call | us/call | total_us |
|---|---:|---:|---:|
| aclnnBatchMatMul_BatchMatMulNd_BatchMatMulV2 | 1.0 | 1111.39 | 55569.46 |
| aclnnAdd_AddAiCore_Add | 1.0 | 890.38 | 44518.86 |
| aclnnInplaceCopy_CastAiCore_Cast | 3.0 | 798.94 | 39947.02 |
| aclnnMul_MulAiCore_Mul | 1.0 | 282.15 | 14107.28 |

## Bottleneck Analysis

The kernel breakdown per forward call is stable across both scopes (6 kernels):
1. `aclnnBatchMatMul_BatchMatMulNd_BatchMatMulV2` — the einsum matmul
   `comb_res_mix[4,4] @ residual[4,1280]` batched over (2,4096) → Cube path.
   ~1111 us/call, ~36% of device time. This is the single largest kernel.
2. `aclnnAdd_AddAiCore_Add` — the `x*post_layer_mix + term2` broadcast add.
   ~892 us/call, ~29%.
3. `aclnnInplaceCopy_CastAiCore_Cast` ×3 — bf16/fp32 casts (residual.float(),
   x.float(), and the final .bfloat16() output cast). ~801 us/call total (3 casts),
   ~26%.
4. `aclnnMul_MulAiCore_Mul` — broadcast mul `x*post_layer_mix`. ~290 us/call, ~9%.

Matmul (Cube) is the top kernel but does not dominate: cast/broadcast (Vector)
kernels together (~892 + 801 + 290 ≈ 1983 us) exceed the matmul (~1111 us).
device_ratio ≈ 0.96 indicates the operator is device-bound, with host launch
overhead small relative to wall time.

## Evaluation Contract Mirror

| Mechanism | Expectation | Observation | Verdict |
|---|---|---:|---|
| einsum maps to Cube matmul kernel | aclnnBatchMatMul present | confirmed, top kernel | confirmed |
| broadcast-mul + add map to Vector kernels | aclnnMul + aclnnAdd present | confirmed | confirmed |
| bf16 cast map to Cast kernel | aclnnInplaceCopy_Cast present ×3 | confirmed | confirmed |
| device_time_available | true | true (CANN ai_core_op_summary.db) | confirmed |
| baseline correctness | v0 == v1 within bound | PASS | confirmed |

## Hypothesis Verdict

- `confirmed` — baseline is device-bound with a real Cube matmul kernel as the
  single largest kernel, but cast/broadcast Vector kernels collectively exceed it.

## evidence_for_next_round

- Observed fact: baseline median wall ≈ 3.21 ms; 6 kernels/call; device_ratio ≈ 0.96.
- Top device kernels: BatchMatMul ~1111 us (Cube), Add ~892 us, Cast ×3 ~801 us,
  Mul ~290 us (Vector).
- Bottleneck: single BatchMatMul is the largest individual kernel, but the three
  Cast kernels + Add + Mul (Vector path) together are the larger aggregate cost.
  Falsified/remaining: none yet; this is the baseline.
- Next-round focus (informational, not prescribed): matmul kernel time vs
  cast/broadcast kernel time both matter; the matmul is per-(a,b) batch of
  [4,4]@[4,1280] which is very small per batch — potential for a fused/vectorized
  formulation, but decision left to Designer.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/mhc-post-layer-mix-ascend
/usr/local/python3.11.15/bin/python3 auto_bench.py \
  --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py \
  --v1_file kernels/track1-triton/mhc_post_layer_mix/ascend/baseline_adapter.py \
  --warmup 50 --repeat 100

/usr/local/python3.11.15/bin/python3 auto_bench.py \
  --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py \
  --v1_file kernels/track1-triton/mhc_post_layer_mix/ascend/baseline_adapter.py \
  --profile --profile-reference-file kernels/track1-triton/mhc_post_layer_mix/ascend/baseline_adapter.py \
  --profile-mode forward --profile-warmup 20 --profile-iterations 50 \
  --profile-output kernels/track1-triton/mhc_post_layer_mix/ascend/log/round_000_forward_50iter.pt.trace.json
```
