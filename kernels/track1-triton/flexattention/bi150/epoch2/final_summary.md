# FlexAttention @ BI150 — Epoch-2 Campaign Final Summary (contract v3)

Campaign: `kernel-opt/flexattention-e2-20260828` · skill kernel-opt-loop v3.0.0
(contract_version 3 / typed-sketch-v1 / verdict-v1) · terminal
`valid-no-improvement-limit` (3/3 no-improvement) at round 003.

## Final Deliverable (Triton submission)

`triton_flexattention_e2_003.py` @ sha256 `6ffb0c94bf6b126317acddcf14119bfd27fab5709c20a1f33cfdf8883d58bf1e`
— the round-003 Triton candidate (graph-replayed-triton-direct-address:
single `_causal_attn_fwd` Triton kernel, three-tier chain), wall 0.148821 ms
paired median ≈ **1.00x parity with base** (0.149147 ms same-session; +0.2186%,
below the 5% adoption bar). Correctness PASS everywhere (5-way bitwise incl.
run_out poisoned ×2; fp16-extreme within 1e-2). Per SKILL deliverable rule the
competition submission is this correctness-PASS Triton implementation — the
campaign manifest keeps `baseline_adapter.py` as canonical anchor only because
adoption never crossed 5% (never advance to a slower/no-gain candidate for
comparison purposes), NOT as the submission. Round-002 Triton candidate
`_002.py` remains in-tree as additional audit evidence.

## Why: The Full Physics (five-number decomposition, µs/call)

| # | layer | value |
|---|---|---|
| 1 | base host path (~38 cheap aten ops + sdpa C++ stack + fixed seed/sync floor) | ≈134 |
| 2 | r001 wrapper net (aten-captured replay, fat boundary) | +2.6 |
| 3 | r002 launcher tax (direct python-launched Triton) | ≈+86–89 |
| 4 | r003 composed net remaining (lean direct-address graph) | **−0.3 (wash)** |
| 5 | device floors: Ixmma 13.61–15.0 vs Triton `_causal_attn_fwd` 16.51 | Δ +2.9…+5.2 |

Root cause of the wash: **build-intrinsic replay-sync R = 69.02 µs/call**
(`cudaDeviceSynchronize` observed on the LEAN tier-1 route whose model code contains
zero sync/query — reading (c) triggered). It absorbs every python-side prize this
operator has to offer.

## Round Trajectory

| round | family | outcome | what it proved |
|---|---|---|---|
| 000 | baseline adapter | baseline | Ixmma = ONE fused kernel 13.56 µs/call; host ≈91% of wall |
| 001 | manual-cuda-graph over base pipeline | NO-IMPROVEMENT −1.69% | mechanism engaged (branch-A collapse) but base had only 1 launch to compress |
| 002 | triton-attention-dispatch-collapse | NO-IMPROVEMENT −60.34% | aten 38→1 + launch structure exact; kernel within +2.9 µs of vendor; Triton python launcher tax ≈85 µs/call dominates |
| 003 | graph-replayed-triton-direct-address | NO-IMPROVEMENT +0.22% (wash) | premise verified at scale (100/100 tier-1, lean census exact); R-term 69 µs is build-intrinsic — reading (c) |

All mechanisms ENGAGED; every failure is attributed to a named, quantified, physical
cost on this CoreX 4.4.0 build. No correctness failures at any point (tie-free
operator; fp16-extreme diff 7.8e-3 = output quantization within 1e-2).

## Reopening Conditions (of record, decision_003 + r003 report)

a. a CoreX/torch build whose cudaGraphLaunch path does not carry the ~69 µs replay
   sync (R-term) — this single fact flips r003 from wash to large win;
b. maintainer-authorized reduction.sum substitution stays unnecessary while the
   primary matrix.dot path remains blocked only by the launcher tax, not by device
   capability (kernel is near-vendor already);
c. harness-side support for 4-arg run_out profiling (make_profile_call arity) would
   unlock kernel-mode evidence without accommodations.

## Governance Note

Fresh v3 campaign ran at the canonical `bi150/` path (epoch-1 archive preserved
in-place after an erratum migration, see team-state transition log). triton_cuda
machine profile v1 reused from the groupedtopk-e2 promotion. All three decisions,
sketches, coder results, reports and verdicts passed their deterministic gates;
measurement fingerprint `6dc07009…` unchanged start-to-stop; groupedtopk fingerprint
reproduced as positive control during Phase 0.
