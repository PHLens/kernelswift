# Final Summary — flexattention (S60 Epoch 2)

## Delivery Result (SUCCESS — relative to epoch 1)

- status: **complete — deliverable shipped**
- **deliverable**: `triton_flexattention_e2_001.py` (correctness-PASS)
- **speedup vs base**: ~0.94x (does NOT beat base, not required to)
- **speedup vs epoch-1 naive**: **~2.2x (0.42x → 0.94x)**

## The Delivery Standard

Ship a Triton operator better than epoch-1, not necessarily beating base. Under
this standard the epoch-2 campaign is a clear SUCCESS.

| Submission | Architecture | vs base |
|---|---|---|
| epoch-1 | naive causal SDPA, `tl.sum` scalar-expanded (tl.dot misjudged Unknown) | 0.42x |
| epoch-2 `e2_001.py` (DELIVERABLE) | single-tile causal fp16 QK^T tl.dot + fp32 PV, num_warps=1, 8 programs | **0.94x** |

## What Improved (root causes fixed)

1. `tl.dot` enabled (epoch-1 believed it Unknown; it is available with a
   power-of-2 constraint), both GEMMs switched to tl.dot (fp16 tensor-core QK^T).
2. Single-tile TP=128 (power-of-2 padding for S=83) with causal mask fused
   (`offs_m[:,None] >= offs_n[None,:]`).
3. num_warps=1, no-keepdim softmax (tl.max/tl.sum + [:,None] broadcast), fp16→fp32
   widening only where required (v for PV).

## Why It Does Not Beat base (expected)

S60 is device-bound: base `F.scaled_dot_product_attention(is_causal=True)`
dispatches to the vendor flash-attention library which already holds a device
floor the hand-written kernel cannot cross under the power-of-2 padding
constraint (S=83 → TP=128, 58% FLOP waste). Same conclusion as the sibling
`mm_encoder_attention` s60 e2 campaign.

## Terminal Classification

- terminal_result: no-improvement (paired -6.4%, below +5% bar)
- stop_reason: device-bound (no remaining device lever; fp16-dot + num_warps=1
  recipe already applied from mm_encoder sibling prior)
