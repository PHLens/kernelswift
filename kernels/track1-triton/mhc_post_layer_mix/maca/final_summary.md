# Final Summary — MHCPostLayerMix C500 (MACA)

- schema_version: 1
- skill_version: 2.0.0
- run_epoch: 1
- run_branch: kernel-opt/mhc-post-layer-mix-c500-20260818
- base_branch: dev
- base_commit: 8c1ebcd04afe4da31357bf426bc3e523129e411c
- measurement_fingerprint: 17bf289997ea6c7a2961ba2640125464ed046471dbff9261a8dcba7fbfccc17e
- stop_reason: user-intervention
- stop_timestamp: 2026-08-19T00:00:00Z
- total_rounds: 2
- accepted_round: 001
- canonical_kernel: triton_mhc_001.py

## Outcome

- baseline wall: 7.635598 ms (warmup 50 / repeat 100)
- final wall: 0.241083 ms
- improvement: +96.84% (31.66x speedup; device 7560.89 -> 168.56 us/call, kernels/call 6 -> 1)

## Round Summary

| Round | Decision | Result | Wall ms | Improvement | Canonical |
|---:|---|---|---:|---:|---|
| 000 | Phase 0 | baseline | 7.635598 | - | baseline_adapter.py |
| 001 | H-001 tiny-k-gemm-fusion | accepted | 0.241083 | +96.84% | triton_mhc_001.py |
| 002 | abort (memory-bound) | aborted | - | - | triton_mhc_001.py |

## Why stopped

Round 001 replaced the badly-sized tf32 GEMM with a single hand-written Triton
kernel. The baseline einsum `'abmn,abmc->abnc'` (contraction K = mhc_mult = 4)
lowered to `mcblas__Mck_tf32gemm_nt_64x64x128` using a 64x64x128 tile, wasting
~97% of K-work; it consumed 6071 us/call = 80% of device time. The fused kernel
does 4 explicit fp32 multiply-accumulates over K=4 and folds the elementwise
tail (mul + add + 2 bf16 casts) into one kernel (6 kernels -> 1), reducing
device time to 168.56 us/call and wall to 0.241083 ms.

Round 002 Designer found the fused kernel is now at the memory-bandwidth floor:
~170 MB of traffic (80 MB output write + 80 MB residual read + 10 MB x read) in
168.56 us ≈ 1 TB/s, near a C500-class HBM ceiling. No candidate-owned lever
(BLOCK size, num_warps>1 (Unknown), index-decode, output-allocation caching)
clears the 5% threshold (12.05 us). Conclusion: memory-bound.

## Resume constraints

Resume only with a same-runtime microbenchmark proving a compressible
candidate-owned bottleneck >= 12.05 us/call, or a user-mandated contract change.
