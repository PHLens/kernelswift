# Final Summary — mm_encoder_attention (S60 Epoch 2)

## Delivery Result (SUCCESS — relative to epoch 1)

- status: **complete — deliverable shipped**
- **deliverable**: `triton_mm_encoder_attention_e2_002.py` (correctness-PASS)
- **speedup vs base**: ~0.915x (does NOT beat base, and is not required to)
- **speedup vs epoch-1 naive**: **~3.39x (0.27x → 0.915x)**
- best candidate: e2_002 (fp16 QK^T tl.dot + fp32 PV, num_warps=1)

## The Delivery Standard

The campaign objective is to deliver a Triton operator that is **better than the
epoch-1 submission** — not to beat the vendor library `base.py`. Under this
standard the epoch-2 campaign is a clear SUCCESS:

| Submission | Architecture | vs base |
|---|---|---|
| epoch-1 (`../triton_mm_encoder_attention_001.py`) | 1328 programs (1 per query token), `tl.sum` scalar-expanded QK^T (no tl.dot), `3× .contiguous()` layout copies | 0.27x |
| epoch-2 `e2_001.py` | single-tile fp32 tl.dot, 16 programs, zero `.contiguous()` | 0.906x |
| epoch-2 `e2_002.py` (DELIVERABLE) | single-tile fp16 QK^T tl.dot + fp32 PV, num_warps=1, 16 programs, zero `.contiguous()` | **0.915x** |

## What Improved (root causes fixed)

1. **tl.dot enabled**: epoch-1 believed "tl.dot is Unknown on GCU" and used
   `tl.sum` scalar expansion for both GEMMs. Epoch-2 probed that tl.dot IS
   available (constrained to power-of-2 tiles) and switched both QK^T and PV to
   tl.dot (fp16 tensor-core QK^T in e2_002).
2. **parallelism collapse**: 1328 programs (1 per query token, each re-loading
   full K/V) → 16 programs (1 per (batch,head), full 128×128 tile in registers).
3. **layout-copy elimination**: `3× .contiguous()` host copies → zero
   `.contiguous()` (direct strided addressing of the [B,S,HD] layout).

## Why It Does Not Beat base (expected, not a failure)

S60 is device-bound: base `F.scaled_dot_product_attention` dispatches to a
vendor CNNL flash-attention kernel (~158us device floor). The power-of-2
constraint on `tl.dot`/`tl.arange` forces S=83 → TP=128 (58% FLOP waste), and
the S60 launcher tax is only 17.4us (vs BI150's 84.77us), so there is no
graph-replay prize. Beating the vendor library was never the goal; shipping a
Triton operator 3.4x faster than epoch-1 is the goal, and it is achieved.

## Profile Corrections (written back to triton_gcu)

- `tl.dot` and `tl.arange` constraint corrected from "mult-of-16" to **power-of-2** (probe-backed).
- `tl.max`/`tl.sum` do NOT support `keepdim` on triton_gcu 3.6.0.
- `tl.dot` requires same-dtype operands.

## Recommendation for Remaining Operators

The same delivery standard applies: ship a Triton operator better than epoch-1,
not necessarily beating base. Under this standard:

- `mhc_post_layer_mix` (epoch-1 0.56x): viable — C500 template (31.66x) + K=4
  reduction unaffected by the dot constraint.
- `flexattention` (epoch-1 0.42x): shares the attention device-bound root cause;
  expect ~3x-class improvement from the same tl.dot switch, even if it does not
  beat base.
- `groupedtopk` (epoch-1 1.68x, already accepted): already strong.
