# Final Summary — mm_encoder_attention (S60 Epoch 2)

## Terminal Result

- status: **terminal (measurement-bound)**
- terminal_result: no-improvement
- best deliverable: `triton_mm_encoder_attention_e2_002.py` (correctness-PASS, ~0.915x)
- last_accepted: baseline_adapter.py (round-000; no candidate beat the +5% adoption bar)
- total_rounds: 2 (round-001 fp32 direct MHA, round-002 fp16 QK^T direct MHA)

## What Was Tried

| Round | Intervention | Wall vs base | Result |
|---|---|---|---|
| 000 | baseline (identity) | 0.995x | baseline established |
| 001 | direct single-kernel MHA, fp32 tl.dot, num_warps=2 | -10.5% | no-improvement |
| 002 | fp16 QK^T tl.dot + fp32 PV, num_warps=1 | -9.3% | no-improvement |

## Root Cause (conclusive)

S60 (Enflame GCU) is **DEVICE-BOUND** for this operator:

1. Base `F.scaled_dot_product_attention` dispatches to a vendor `_scaled_dot_product_flash_attention` CNNL library kernel with a ~158us device floor.
2. The S60 `tl.dot` and `tl.arange` constraints are **power-of-2** (NOT mult-of-16 — 96=16×6 fails), forcing S=83 to pad to TP=128 → 58% FLOP waste that is structurally unavoidable.
3. The compressible host+launcher total is only ~28us (11us host chain + 17.4us launcher tax), far below the device deficit.
4. The BI150 win lever (graph-replay to collapse an 84.77us launcher tax) has **no prize on S60** (launcher tax is 5x smaller at 17.4us).

All device levers are exhausted: fp32→fp16 dtype (-10.5%→-9.3%, real but insufficient), num_warps 2→1, grid-split (measured WORSE in r001, 195-228us).

## Deliverable

Per the DELIVERABLE RULE, the campaign's primary contractual product is the best correctness-PASS Triton submission, which is banked:

- `triton_mm_encoder_attention_e2_002.py` — single-tile direct MHA, fp16 QK^T + fp32 PV, num_warps=1, bidirectional (non-causal), scale=0.125, stateless, `forward` + 4-arg `run_out`, correctness PASS (max_abs ~1.5e-3 < 1e-2).

## Profile Corrections (written back to triton_gcu)

- `tl.dot` and `tl.arange` constraint corrected from "mult-of-16" to **power-of-2** (probe-backed: 16/32/64/128 pass; 48/80/96/112/160/192 fail).
- `tl.max`/`tl.sum` do NOT support `keepdim` on triton_gcu 3.6.0.
- `tl.dot` requires same-dtype operands.

## Recommendation for Remaining Operators

The power-of-2 + device-bound finding transfers directly to the other S60 epoch-2 targets:

- `flexattention` (0.42x) shares the same attention device-bound root cause (T=83 power-of-2 padding + CNNL SDPA floor); expect the same terminal outcome unless its base is host-bound (unlike mm_encoder_attention, flexattention's base may have more host overhead worth collapsing).
- `mhc_post_layer_mix` (0.56x, K=4 reduction) does NOT use tl.dot (K too small for power-of-2 dot); it remains a viable target per the C500 31.66x template.
- `groupedtopk` (1.68x, already accepted) is a reduction workload, not GEMM; unaffected by the dot constraint.
