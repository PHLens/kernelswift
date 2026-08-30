# MM Encoder Attention @ BI150 — Epoch-2 Campaign Final Summary (contract v3)

Campaign: `kernel-opt/mmenc-attn-e2-20260828` · skill kernel-opt-loop v3.0.0
(contract_version 3 / typed-sketch-v1 / verdict-v1) · terminal:
`user-intervention` stop at round-003 acceptance (streak reset, run converged).

## Final Deliverable (competition submission)

`triton_mm_encoder_attention_e2_003.py` @ sha256
`d503e845d1d95ad033e616730d78d67d0775df3a473b17a3c37843d463654d81`
— composed graph-replayed Triton full-attention: a self-written Triton kernel
(num_warps=2, grid 48, fp32-widened (32,32) dots, direct strided addressing)
captured once as a manual CUDA graph reading the caller's own input pointers,
replayed per call.

| basis | value |
|---|---|
| wall paired median (protocol statistic) | 0.149939 → **0.142327 ms** |
| improvement | **+5.077% (1.0535x)** — bar cleared by 0.077 pp; 8/8 win rate |
| correctness | PASS everywhere; 6-way bitwise retention (tier-1/2/3 + run_out poisoned ×2 + r002 twin) |
| deliverable trajectory | epoch-1 naive 0.547x → r001 0.6033x → r002 0.6258x → **r003 ~1.05x** |

## How the reversal happened (three steps)

1. **r001 (0.6033x)**: self-written correct Triton attention. Lost, but the census
   named the enemy — not compute, but an ~85 µs/call Triton python-launcher tax
   per call, larger than the whole host stack it removed.
2. **r002 (0.6258x)**: changed ONE literal — `num_warps` 1→2. Register pressure
   halved; device time 28.203 → 19.555 µs/call (−30.7%) with bitwise-identical
   outputs. Still lost (launcher tax unchanged), but the bullet was sharpened.
3. **r003 (~1.05x)**: captured that kernel launch into a manual CUDA graph bound
   to the caller's own pointers — the 85 µs launcher never executes on the
   serving route. Per call = 3×data_ptr guard + one replay + one copy-out.

## The favourable falsification (why it beat the optimistic forecast)

Priced identity expected +3.28 µs WORSE (parity-class at best). Measured −7.6 µs.
Root causes now measured:
- **R (replay sync) and the in-graph round-trip are NOT additive** — the sync API
  wait absorbs the graph round-trip; the two costs overlap instead of stacking.
  Worth ~7 µs — this is the exact margin that turned a predicted loss into a win.
- The replaceable base host stack is ~131 µs (larger than T_launcher alone).
- Boundary aten cost is 55.36 µs/call, not the ~2 µs assumed.

## Campaign physics closure (µs/call)

| quantity | measured |
|---|---|
| T_launcher (Triton python launcher tax) | +84.765 (r001) / +84.571 (r002) — invariant, and REMOVED on the r003 replay route |
| D_cand floor trajectory | 28.203 (nw1) → 19.555 (nw2) → 64.467 in-graph round-trip (frontend ~46 build-intrinsic; kernel math ~18.4 unchanged) |
| R-term (replay sync penalty) | 65.76 at bsz=2 vs sibling 69.02 at bsz=1 — **TRANSFERS** |
| base vendor kernel | Ixmma `FlashAttnFwdF16Ixmma` CausalM=0, 17.42 µs whole-trace / 15.36 attributed, single launch covering both batches |
| base host share | ~89% (device_ratio ~0.11) |

## Capability matrix (canonical, cross-operator value)

- **fp16-operand `tl.dot` @fp32-acc**: COMPILES on BI150 triton 3.1.0 (8.6–11.7 µs
  kernel-only) but FAILS exactness on fp16-extreme at EVERY warp count
  (max_abs 1459, one-hot tie-flip) — the identical failure mode the vendor Ixmma
  kernel itself exhibits (1457). Capability-NEGATIVE for this lineage.
- **num_warps**: nw2 optimal; nw4 no gain (15.441 vs 15.317 probe-method); nw1
  spill-class (~288 regs/thread vs 255 budget).
- **Triton-launch capturability inside a manual graph**: PROVEN at scale (this
  campaign's enabling fact).
- **kineto graph-interior blindness** (D2′): graph-interior kernels emit no
  `cat=kernel` events on this build; census substitution via API census + CUDA
  events is the canonical workaround.
- **Correctness supremacy datapoint**: on fp16-extreme inputs the candidate
  matches fp32 ground truth to 3.05e-05 while the vendor base diverges by 1457.

## Reopening conditions

a. a CoreX/torch build reducing the ~46 µs graph frontend or the ~66 µs replay
   sync — the single highest-leverage unknown remaining;
b. eliminating the ~34.7 µs fresh-destination allocation (empty_like +
   empty_strided) — the only remaining attributable host lever;
c. harness-side 4-arg run_out profiling support (make_profile_call arity) to
   restore kernel-mode evidence.
