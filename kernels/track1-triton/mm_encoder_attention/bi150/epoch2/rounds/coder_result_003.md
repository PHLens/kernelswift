# Coder Result 003

Result: candidate-ready

## Identity

- Round: `003` (FINAL ROUND of this campaign)
- Decision: `rounds/decision_003.md` @`0a678da87a877b9c521b6c280eb3518b20f98e352786e9df129435e2cc918413` (hash re-verified from file; matches dispatch-validated value; family "graph-replayed-triton-direct-address" (F2 composition), expected_wall_improvement_pct 0.0 — honest; the round's product = the composed correctness-PASS Triton submission + graph-family physics closure)
- Sketch: `rounds/sketch_003.json` @`bdf423556e7c80369ae38d4980529a739a52a3d18033e572927354b23e0a4e64` (hash re-verified; matches decision sketch_sha256)
- Candidate: `triton_mm_encoder_attention_e2_003.py` @`d503e845d1d95ad033e616730d78d67d0775df3a473b17a3c37843d463654d81`
- Canonical source (kernel): the banked r002 deliverable `triton_mm_encoder_attention_e2_002.py` @`cc98318bbfed0d9cdbf7a99769c4a44077dc15590496a4e8410ba9eda99f3078` — **kernel BYTE-IDENTITY machine-verified** (extraction-diff of the `@triton.jit`…`class ModelNew` segment: equal, 4168/4168 chars); round 003 changes ONLY the execution boundary (the three-tier graph-replay chain around it)
- Sibling template: `../../flexattention/bi150/epoch2/triton_flexattention_e2_003.py` (same architecture — read fully; its p13–p19 probe suite was the checklist template)
- Base (immutable reference): `../../base.py` @`86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (unchanged)
- Harness: `/root/CodeBuddy/20260818191200/kernelswift/auto_bench.py` @`71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (untouched)
- Runtime fingerprint: torch 2.7.1 / triton 3.1.0 (corex-4.4.0) / Iluvatar BI-V150 sm71 mp16 / cuda:0 / `export COREX_VERSION=4.4.0; . /usr/local/corex/enable` in every shell; interpreter `/usr/local/bin/python3` — matches `project.md#runtime-fingerprint` (live re-probed in p13)
- Binding statement artifact: `log/probes/binding_statement_report_003.json` @`4b3985a81b134cc947ae2cbaf1436e67885365a9be54fda8aef6961e5779c9b6`

## Implementation Summary (decision-003 exact)

Three-tier permanent chain around the byte-identical r002 kernel:

- **tier-1 direct-address replay**: the single r002 kernel launch captured ONCE per first-seen input pointer set as a manual `torch.cuda.CUDAGraph` on a dedicated side stream (3 warmup launches freeze the JIT specialization BEFORE the capture window; the captured region is EXACTLY one kernel launch — no branches, no prints, no host reads), bound to the CALLER'S OWN q/k/v data_ptrs (ZERO copy-ins) and writing the static `out_ws` [2,83,512] fp16 workspace. Per served call: 3-way `data_ptr` guard + ONE replay + ONE ~166 KB copy-out into a fresh invocation-owned buffer (forward) or the caller buffer (run_out). The r002 python launcher (T_launcher=+84.57 µs/call) never executes on this route.
- **tier-2 copy-in replay**: static `q_in/k_in/v_in` workspaces captured once; per call 3 copy-ins + ONE replay + copy-out; serves pointer mismatches, same-set revisits, and budget overflow — bitwise-identical results, never stale-address service.
- **tier-3 eager**: the r002 direct-launch path (`torch.empty` + ONE launch, nw2) for non-target regimes and any replay failure.
- **Recapture**: bounded ≤4 lifetime (initial binding budget-free; every rebind decrements irreversibly; first-seen pointer sets only — `bound_sets` ≤5-entry guarded history; same-set revisits ride tier-2 so alternation never re-binds); never triggered after budget exhaustion; any capture/replay exception binds the offending tier permanently downward while the triggering call stays correct through the next tier.
- **Zero model-code sync/query** (data_ptr reads only); results NEVER served from graph-resident memory (forward returns fresh buffers; run_out fills the caller buffer via copy-out outside the replay boundary); caller device and current stream preserved (captures once per binding on a side capture stream; replays/copy-outs on the caller's current stream).
- **Bounded state**: exactly the declared set — 2 graph handles, out_ws, q_in/k_in/v_in, anchor triple, bound_sets (≤5), recapture counter, 2 monotone tier flags, _ws_device (+4 constructor-config attrs; live-verified — all other `__dict__` entries are stock nn.Module framework infrastructure present identically in baseline_adapter/r001/r002).
- **run_out** 4-arg surface across ALL tiers; **forward** signature unchanged; `get_inputs`/`get_init_inputs` unchanged.

## Decision-scoped Checks (log/probes/ only — non-authoritative, NO timing/benchmarks/profilers)

| Probe | Verdict | Key evidence |
|---|---|---|
| p13 cold-capture smoke (harness-sequence reproduction) | PASS exit 0 | correctness call on fresh clone pointers → initial binding budget-free (bound_sets=1, budget=4); warmup call 1 on the stable set → EXACTLY one recapture (bound_sets=2, budget=3, 2 graphs); warmup calls 2-5 zero recaptures; timed segment 10/10 replay-served with ZERO python-launcher executions and zero captures/recaptures; every output bitwise==r002 twin on CURRENT live bytes (in-place mutation between serves) |
| p14 capture-fired multi-fact | PASS exit 0 | f1 live direct handle + first-serve bitwise vs twin; f2 static out_ws address stable; f3 live-byte freshness WITHOUT recapture (mutated bits flow into next replay bitwise); f4 lower-tier absence (replay+5, launcher+0, tier-2 never built); f5 stale-address impossibility (new pointer set recaptured, served from ITS OWN bytes); f6 zero sync/query calls across 10 replay-served serves — intersection {f1..f6} all True |
| p15 recapture exercise | PASS exit 0 | 7 distinct sets: ledger EXACT [(1,4),(2,3),(3,2),(4,1),(5,0),(5,0+t2),(5,0+t2)]; budget exhausted at 0, never negative; overflow sets 6-7 ride tier-2 (copy-in graph built at first overflow); same-set revisit never re-binds (tier-2 from live A bytes; anchors stay on B) |
| p16 tier edges | PASS exit 0 | vector A (tier-1 capture fault once) → tier-2 serves same-call, permanence honored; vector A2 (total capture failure) → tier-3 eager serves same-call, both tiers permanently down, artifacts dropped; vector B (tier-1 replay fault on captured instance) → tier-2 same-call; vector C (tier-2 replay fault) → tier-3 same-call — all outputs bitwise-correct through every edge |
| p17 run_out poison + alt surface | PASS exit 0 | poisoned buffers ×2 orderings bitwise on ALL THREE tier instances; data_ptr preserved; zero aliasing to out_ws/q_in; forward results ALWAYS fresh buffers (never graph-resident); cross-tier run_out bits identical for identical inputs |
| p18 cross-instance alternation | PASS exit 0 | 12 interleaved serves across 4 instances (2× tier-1, tier-2-pinned, tier-3-pinned) × 3 mutation rounds — every output bitwise-matches its own instance's CURRENT live bits; tier state stable; no cross-instance contamination |
| p19 bitwise sweep (machine table `p19_r003_sweep_result.json`) | PASS exit 0 | see table below |

### Sweep table (6-way bitwise = tier-1 / tier-2 / tier-3 / run_out poisoned ×2 / r002 twin)

| Suite | 6-way bitwise | allclose | max_abs | shape |
|---|---|---|---|---|
| seed42_B2S83 (vs base) | True | PASS | 4.883e-04 | ok |
| boundary_B2S83 (tile-boundary spikes at tokens 0/32/64/82; vs base) | True | PASS | 4.883e-04 | ok |
| extreme_B2S83 (vs fp32 ground truth, r001-established basis) | True | PASS | 3.052e-05 | ok |
| B1S41 non-target (tier-3, zero graph artifacts) | bitwise vs twin | PASS vs base | — | ok |
| B2S82 non-target (tier-3, zero graph artifacts) | bitwise vs twin | PASS vs base | — | ok |
| B2S96 non-target (tier-3, zero graph artifacts) | bitwise vs twin | PASS vs base | — | ok |

max_abs readings are r001/r002-identical (4.883e-04 vs base on randn suites; 3.052e-05 vs fp32 ground truth on extreme) — the composed chain preserves the full correctness pedigree bit-for-bit.

## Binding Statement

- **Kernel byte-identity vs r002**: machine-verified (extraction-diff `@triton.jit`…`class ModelNew` equal, 4168/4168 chars) — same mathematics, same (32,32)@(32,32) fp32-widened dots (4 sites, 6 widen casts), same `num_warps=2` at the single launch site, same 48-program grid, `num_stages` absent.
- **DANGER tokens** (14 tokens incl. compile family, `synchronize`, `.query(`, `cudaDeviceSynchronize`, `DriverGet`, graph-resident-return patterns, `.contiguous(`): ALL-ZERO over the final source.
- **Intentional mechanism sites** (the round's sanctioned machinery, machine-counted): `torch.cuda.CUDAGraph` = 1 code site (manual capture) + 1 docstring mention; `is_contiguous` = 1 site (the `_is_target_regime` layout gate — metadata-only query, NOT a layout-copy call); `copy_` = 5 sites (tier-1 copy-out ×1 + tier-2 copy-ins/copy-out ×4), ZERO copies inside the captured region (capture window = exactly one kernel launch).
- **Graph-resident-return rule**: forward returns only fresh buffers; run_out fills the caller buffer and returns None; `return self.out_ws`/`return out_ws` count 0; p17 proves fresh data_ptrs and zero aliasing on all three tiers.
- **Zero model-code sync/query**: `synchronize`/`.query(`/`DriverGet` count 0 in source; p14 f6 proves zero python-visible sync/query calls across 10 replay-served serves (torch.cuda.synchronize + Stream/Event synchronize/query all patched and counted). Driver-intrinsic replay sync (the R-term) is invisible to python spies and reserved for the Verifier's census (the `rterm_transfer_at_bsz2` observable).
- **Bounded state**: candidate-owned attrs = the declared set exactly (live-verified after serving); bound_sets ≤5; recapture ≤4; monotone tier flags; nothing else persists; results never stored across calls.
- **AST-loader composition**: 4 imports + 4 literal assigns + `@triton.jit` FunctionDef + ClassDef + 2 helper FunctionDefs — all retained node types; real-harness loader gate PASS.

## Attempt Ledger

| Attempt | Command (abridged) | Exit | Defect | Candidate SHA |
|---|---|---|---|---|
| 1 | author candidate (kernel copied verbatim from r002; three-tier chain adapted from the sibling template) | — | none | `d503e845…` |
| 1 | gates: ast.parse + kernel byte-identity extraction-diff + 14-token DANGER scan + mechanism counts | 0 | none — KERNEL_BYTE_IDENTICAL (4168/4168); all-zero; 4 dots/6 widens/1 num_warps=2 site | `d503e845…` |
| 1 | harness smoke (real AST loader) | 0 | none — `PASS accuracy`; composition ENGAGED (0.863x vs r002 smoke 0.623x) | `d503e845…` |
| 1 | probes p13–p19 (each first-attempt) | 0 ×7 | none — ALL PASS first attempt (probe-suite template discipline from the sibling) | `d503e845…` |
| 1 | bounded-state audit shell check | 0 | AUDIT-SCRIPT-SIDE only: my quick comparison included `_ws_device` in the expected list while filtering underscore keys from the actual list, and nn.Module stock framework attrs (hooks/_parameters/…) appear in every module — re-run with the correct expected set: candidate-owned attrs = declared set exactly; NOT a candidate defect | `d503e845…` |

Zero same-round candidate repairs consumed.

## Deviations

- **D1 (observation, no code change)**: harness smoke at `--warmup 5 --repeat 10` printed `PASS accuracy; v0=0.155737 ms, v1=0.180555 ms, speedup=0.863x` (exit 0). Harness-printed observation only, NOT a coder measurement claim. The composition engaged at smoke scale (0.240 → 0.181 ms vs r002's smoke). The decision's pre-declared honest band for the warmup-50/repeat-100 paired medians is 0.94–1.01x; the smoke reading sits BELOW that band — consistent with either genuine sub-band performance at smoke scale or smoke-scale variance; explicitly NOT extrapolated by coder. The Verifier's paired medians + census (tier1_hit_rate, submissions, R-term at bsz=2, kernel-in-graph regime) decide.
- **D2 (conformance note, budget interpretation)**: "recapture ≤4 lifetime" implemented as initial-binding-free + ≤4 recaptures (≤5 captures lifetime) — identical to the sibling campaign's accepted interpretation; documented in the binding statement for Orchestrator/Verifier.
- **D3 (conformance note)**: `is_contiguous` appears once (the `_is_target_regime` layout gate — a metadata-only query ensuring a non-row-major tensor never rides the constexpr-stride graph path); the layout-COPY call token `.contiguous(` counts 0. The r001/r002 "zero layout-copy" discipline is preserved.
- **D4 (conformance note)**: total-capture-failure reading (e) was NOT taken — capture succeeded on the first attempt (probe-supported by r002's p13 and confirmed by all seven r003 probes + the harness smoke); the tier-2/tier-3 fallbacks exist and were exercised by fault-injection (p16/p17/p18/p19), per the decision's permanence requirements.
- **D5 (conformance note)**: GQA (`num_kv_heads != num_heads`), non-fp16 dtypes, non-cuda devices remain outside the target regime (route tier-3 with r002's own out-of-contract properties — unchanged boundary from the accepted r002).

## Evidence for Verifier

- Candidate: `triton_mm_encoder_attention_e2_003.py` @`d503e845d1d95ad033e616730d78d67d0775df3a473b17a3c37843d463654d81`.
- Probe logs under `log/probes/` (hashes below). Canonical measurement route: unchanged harness, default stream, seed 42, warmup 50 / repeat 100 interleaved pairs.

### Artifact hash ledger

```text
d503e845d1d95ad033e616730d78d67d0775df3a473b17a3c37843d463654d81  triton_mm_encoder_attention_e2_003.py
cc98318bbfed0d9cdbf7a99769c4a44077dc15590496a4e8410ba9eda99f3078  triton_mm_encoder_attention_e2_002.py (kernel source, byte-identical segment)
0a678da87a877b9c521b6c280eb3518b20f98e352786e9df129435e2cc918413  rounds/decision_003.md
bdf423556e7c80369ae38d4980529a739a52a3d18033e572927354b23e0a4e64  rounds/sketch_003.json
c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f  baseline_adapter.py
86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  /root/CodeBuddy/20260818191200/kernelswift/auto_bench.py
c7d4699e42524cade1f3ed25e3378ff2e2f9b4486709be80366609701f78bce8  log/probes/p13_r003_cold_capture_smoke.py
0a9a6bfb8b9c9b1b582cd7a7e8a2b094cb4a25752e62f5ae42039227ba7c4ab2  log/probes/p13_r003_cold_capture_smoke.log
1da78285a1dd8f2f6cd740a39819ccb7e2e6dd30524a0ab7300f253310d4fe8f  log/probes/p14_r003_capture_fired_multifact.py
64fd05db51b5309286d89fb51d4e490cde64cb8f39e4b63dedabb5a77e6b90c2  log/probes/p14_r003_capture_fired_multifact.log
4fc7e76e8c76d0d1ec4d05fd102addb448b446faa0b0813784d488eefe0927af  log/probes/p15_r003_recapture_exercise.py
5f33cdabaa9c3eb6ad9cf519e47c671adc42d989793720406f38c5d39f83dd68  log/probes/p15_r003_recapture_exercise.log
e9cc3ff94d8052cf5100912f18bcdfcda9b1f86f0b6146a9f82bef79229d9048  log/probes/p16_r003_tier_edges.py
92235ef6ce4243b3642f8a8f31104de3f8da53d0e52a653f9fc4540cac03f08e  log/probes/p16_r003_tier_edges.log
6f285f4336fe9ca8f8fc3af778c5f6f7d86eb5adb715971095585f81385140b4  log/probes/p17_r003_run_out_poison_altsurface.py
b71ceb0c14bbea3d6892ad6f9dcf0f6934dee8c39206fc8e4f4ab269752ba6de  log/probes/p17_r003_run_out_poison_altsurface.log
9271180b9cb4c7b342237b2d07d78f09d03bf07e2a012d5e0b0902d348bc2ba4  log/probes/p18_r003_cross_instance_alternation.py
a38e5762ec0899637ea117eb053eff04da641971f23d9d9c8bc95d07c98e0828  log/probes/p18_r003_cross_instance_alternation.log
4ae6767332d1052dbf6c9a03ed3e79d3478b498b02768a344c10c1945c9bd938  log/probes/p19_r003_bitwise_sweep.py
a78df0b8abf487e999e57c1ae351e2d5c9cce52ff299ac9197028afc26d76e71  log/probes/p19_r003_bitwise_sweep.log
eefeddd6b336de9781e4c4c8498035e2d9c58fe5e9923cbf94c7ef762eda0348  log/probes/p19_r003_sweep_result.json
4b3985a81b134cc947ae2cbaf1436e67885365a9be54fda8aef6961e5779c9b6  log/probes/binding_statement_report_003.json
```

## Exact Commands (all with `cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable;` prefix; interpreter `/usr/local/bin/python3`; device cuda:0)

```bash
# kernel byte-identity vs r002 (machine-verified extraction-diff)
/usr/local/bin/python3 -c "seg=lambda p:(lambda s:s[s.index('@triton.jit'):s.index('class ModelNew')])(open(p).read()); assert seg('kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_002.py')==seg('kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_003.py'); print('KERNEL_BYTE_IDENTICAL')"

# gates + real-harness smoke (exit 0, PASS accuracy)
/usr/local/bin/python3 -c "import ast; ast.parse(open('kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_003.py').read()); print('AST_PARSE_OK')"
/usr/local/bin/python3 auto_bench.py \
  --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
  --v1_file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_003.py \
  --warmup 5 --repeat 10 --full-traceback
# 14-token DANGER scan (all-zero) + mechanism counts: 1 CUDAGraph code site; 1 is_contiguous gate; 5 copy_ sites; 4 dots; 6 widens; 1 num_warps=2 site

# probes p13-p19 (each exit 0, all first-attempt PASS)
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/p13_r003_cold_capture_smoke.py
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/p14_r003_capture_fired_multifact.py
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/p15_r003_recapture_exercise.py
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/p16_r003_tier_edges.py
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/p17_r003_run_out_poison_altsurface.py
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/p18_r003_cross_instance_alternation.py
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/p19_r003_bitwise_sweep.py
```

Coder claims no measurement and no verdict; classification is candidate-ready. Orchestrator owns the verification dispatch — this is the campaign's final coding round, and the composed candidate is the DELIVERABLE-RULE product (correctness-PASS Triton submission at the composed boundary).
