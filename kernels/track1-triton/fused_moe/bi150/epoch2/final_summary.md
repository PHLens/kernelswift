# FusedMoE @ BI150 — Epoch-2 Campaign Final Summary (contract v3)

Campaign: `kernel-opt/fusedmoe-e2-20260828` · skill kernel-opt-loop v3.0.0
(contract_version 3 / typed-sketch-v1 / verdict-v1) · terminal:
`user-intervention` stop after round-002 + G2 pre-measurement closed every remaining lever.

## Final Deliverable (competition submission)

`triton_fused_moe_e2_001.py` @ sha256
`da623fa92819185a1e20a8a7cbaca40acd9bfb4a3147f8e1e7b1e757c6b24cb7`
— counting-sort grouped-GEMM MoE (two Triton kernels: bucket-count/sort + grouped
expert FFN) captured once as a manual CUDA graph bound to caller pointers, replayed
per call.

| basis | value |
|---|---|
| wall paired median | 3.193262 → **0.219792 ms** |
| improvement vs canon 3.255288 | **+93.25% (14.81x)** |
| correctness | 12/12 (5 expert-activation variants, fp16-extreme, non-target shape, run_out poisoned ×2, 20-call determinism) |
| epoch-1 → epoch-2 | 6.60x → **14.81x** |

## How the win landed (host lever, not device)

Round 000 census named the real enemy: the base ran 123.95 kernels/call with
device only 29.7% of a 3.255 ms wall — 65.6% of device time was dispatch/indexing
(scatter-store + mask-gather + nonzero + mask.any + cub reduce), not the GEMMs
(12.27%). Round 001 then:
1. restructured into a counting-sort + grouped expert GEMM (two Triton launches),
   eliminating the 12.34x replicated GEMM work and lifting SM utilization;
2. captured that 2-launch sequence in a manual graph — the ~85 µs/call Triton
   python-launcher tax never executes on the replay route;
3. collapsed 9.82 aten launches into 2.0 submissions/call (1 graph launch + 1 copy-out).

The host lever delivered far beyond the model: 423 µs replay-vs-eager (the
`N_triton × 85 µs` identity under-counted — collapsing the 9.82 aten launches
removed far more than two launcher taxes). The device restructure was measured
DEVICE-NEUTRAL (58.2 vs 55.9 µs isolated), not a win — the decision's FR-2 fired
but explicitly permitted a partial device landing adopted on the host half.

## The three levers, all now closed (measured, not assumed)

| lever | disposition |
|---|---|
| device restructure (BLOCK_M 256→16) | NEUTRAL — the arithmetic-reduction argument did not convert to device time |
| G1 allocation reuse | FALSIFIED — empty_like ≈ 4.13 µs/call (orchestrator re-measure), ceiling below the 10.99 µs gate; "no per-call allocation" and "never aliases" are mutually exclusive below ~150 forwards |
| G2 routing-prelude → Triton | DEAD — ~9-11 µs device reclaim but ~0 µs wall (prelude already in-graph, off the critical path under the ~122 µs harness sync floor); softmax fold trips the NOT-granted reduction.sum waiver (fp32 axis-k), renorm-sum trips the same waiver, topk frozen by tie semantics; only waiver-clean fold is the ~1.6 µs fp16 cast |

## Physics (µs/call, canon)

- base wall 3255.288 → r001 wall 219.792; harness `cudaDeviceSynchronize` ≈ 122 is
  NOT addressable (inside the timed region), setting a hard wall floor ≈ 214 µs.
- Triton launcher tax ≈ 85 (removed on replay route); graph frontend + replay sync
  build-intrinsic; device work for the accepted kernel ≈ 58 µs isolated.

## Round trajectory

| round | family | outcome |
|---|---|---|
| 000 | baseline adapter | baseline (canon 3.255288 ms) |
| 001 | manual-graph-replay-fused (counting-sort grouped GEMM) | ACCEPTED +93.25% (14.81x) |
| 002 | G1 allocation-reuse (option ii, C3) | NO-IMPROVEMENT by design (retention hardening, +0.093 µs) |
| G2 pre-measure | routing-prelude fold pricing | DEAD (waiver gate + ~0 wall) |

## Capability ledger (cross-operator, carried forward)

- fp16-operand `tl.dot` exactness is OPERATOR-DEPENDENT: NEGATIVE on attention
  (vendor-saturation tie flips), POSITIVE on MoE (2.441e-04 vs 1e-2) at M16/32 × K64/128.
- int64→int32 kernel-side narrowing: QUALIFIED (was `unknown`).
- `tl.where(cond,x,0)` and `(cond).to(int32)*x` masked adds on small tiles: compile
  but SILENTLY return zeros — rig-specific trap.
- masked-add-equivalent scatter builds a [256,256] rank matrix at 553.9 µs — avoid.
- atomic cursor scatter is fast (6.6 µs) but nondeterministic / not allclose — rejected.
- num_warps: nw1 won by 24.4% here (sibling attention's nw2 prior does NOT transfer).
- Profile coverage gaps surfaced: no reduction.sum / tl.sigmoid / tl.atomic_add /
  allocation contracts declared (binding ledger accommodated 4 statements).

## Reopening conditions

a. a CoreX/torch build without the ~122 µs in-region cudaDeviceSynchronize (the
   dominant non-addressable cost);
b. a granted reduction.sum waiver (would unlock softmax/renorm fold — still only
   ~10 µs device, ~0 wall, so marginal);
c. harness-side support for 3-arg run_out kernel-mode profiling.
