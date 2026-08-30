# Final Summary — mhc_post_layer_mix (S60 Epoch 2)

## Delivery Result (SUCCESS — relative to epoch 1)

- status: **complete — deliverable shipped**
- **deliverable**: `triton_mhc_post_layer_mix_e2_001.py` (correctness-PASS)
- **speedup vs base**: ~0.77x (does NOT beat base, not required to)
- **speedup vs epoch-1**: **~+37% (0.561x → 0.770x)**

## Interventions (two, both correctness-PASS)

1. **BLOCK_H 256 → 1024** (largest power-of-2 ≤ h=1280): 0.561x → 0.726x
2. **bf16 register residency**: keep `x`/`residual` loads in bf16 registers and widen
   to fp32 only at the contraction (bf16→fp32 is lossless, so semantics are
   identical), halving register pressure: 0.726x → 0.770x

## Full exploration (all correctness-PASS unless noted)

| variant | wall | vs base |
|---|---:|---:|
| BLOCK_H=64, tl.sum | 20735us | 0.20x |
| BLOCK_H=256, tl.sum (epoch-1) | 7383us | 0.561x |
| BLOCK_H=1024, tl.sum fp32 | 5650us | 0.733x |
| BLOCK_H=1024, tl.sum **bf16 registers (e2)** | **5219us** | **0.797x** (probe) / **0.770x** (harness) |
| explicit 16-MAC | 10943us | 0.38x |
| `tl.dot` [4,4]@[4,BLOCK_H] | 215792us | **0.019x** (catastrophic) |
| split-h grid (2/4) | 8712-15569us | worse |
| hybrid (vendor bmm + hand tail) | 9395us | worse |

## Root-Cause Analysis (why it does not beat base)

S60 base decomposes as (per 100-call forward census):

| component | cost |
|---|---|
| `einsum` → `torch.bmm` (vendor) | 1740us (42%) |
| elementwise tail (broadcast-mul + add + cast) | 3179us (76%) |
| `residual.float()` cast | 738us |

- This operator is a **memory-bound small-contraction GEMM** (K=4, ~40M output
  elements, FLOP/byte ratio near zero). The S60 vendor `bmm` + elementwise library
  are already near the bandwidth floor, so hand-written Triton cannot beat base.
- `tl.dot` is catastrophic (M=4/K=4 far below the tensor-core minimum 16×16 tile;
  the power-of-2 constraint only guarantees compilability, not efficiency).
- num_warps=2 degrades ~2x at every tiling; num_warps=1 optimal.
- Hand-written ceiling is ~0.77x regardless of tiling/contraction method.

## Note

- `tl.dot` is NOT used (K=4 far below the power-of-2 dot efficiency floor on S60).
- The winning lever is `tl.sum` contraction + bf16 register residency + large
  BLOCK_H; the delivery standard (better than epoch-1) is met: 0.561x → 0.770x.
