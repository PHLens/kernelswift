# Coder Result 002

Result: candidate-ready

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md` @`20b360ac936bf4d9d41afadac90c40578f0a758e628ec40af2d3c759eb22d3fb` (hash re-verified from file; matches dispatch-validated value; family "triton-attention-kernel-config" (F3), expected_wall_improvement_pct 0.0 — honest round; the round's product = capability qualification + best-config deliverable)
- Sketch: `rounds/sketch_002.json` @`c16b1528b25ae1a3bbfc72b3e459462505d940677e62b30a0585e3b41b46e9e9` (hash re-verified; matches decision sketch_sha256)
- Candidate: `triton_mm_encoder_attention_e2_002.py` @`cc98318bbfed0d9cdbf7a99769c4a44077dc15590496a4e8410ba9eda99f3078`
- Canonical source (base for this round): the banked r001 deliverable `triton_mm_encoder_attention_e2_001.py` @`4171de8dcb1df6478942b0861756d660ef7955c5741e60072f03476a5fab2fc2` — the ENTIRE boundary stays (2-op forward host, run_out 4-arg surface, grid 48, (32,32) tiles, online softmax, direct strided addressing); ONLY the kernel execution config varies: `num_warps` 1 → 2 as a fixed module literal at the single launch site. Kernel arithmetic textually identical; p14 proves the outputs are BITWISE-equal to r001 on identical inputs (warp-count invariance on this rig).
- Base (immutable reference): `../../base.py` @`86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (unchanged; the r001 verification cycle did not mutate it)
- Harness: `/root/CodeBuddy/20260818191200/kernelswift/auto_bench.py` @`71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (untouched)
- Runtime fingerprint: torch 2.7.1 / triton 3.1.0 (corex-4.4.0) / Iluvatar BI-V150 sm71 mp16 / cuda:0 / `export COREX_VERSION=4.4.0; . /usr/local/corex/enable` in every shell; interpreter `/usr/local/bin/python3` — matches `project.md#runtime-fingerprint`
- Binding statement artifact: `log/probes/binding_statement_report_002.json` @`322d932902fb161d7f4529576db972f1a858b2a731ff31a4f8b9f2ca1bde3863`

## Implementation Summary (decision-002 exact)

- Candidate = r001 boundary verbatim + selected execution config as fixed module literals: `num_warps=2` at the single shared launch site (`ModelNew._launch`), fp32-widened dot operands unchanged (6 widening casts at the tile-load sites, all 4 dot sites at the proven (32,32)@(32,32) fp32/fp32→fp32 envelope). No runtime selection, no config search machinery — the final candidate carries exactly ONE config.
- Sequencing honored: the capability sweep ran BEFORE finalizing the shipped config (p13, under `log/probes/`), then the candidate was built from the sweep's selection.

## Pre-adoption Capability Sweep (p13, Decision-scoped, log/probes/ — non-authoritative)

6 configs = num_warps {1,2,4} × dot-operand {fp32-widened, fp16-operand@fp32-acc} of the r001 kernel parametrized by a constexpr `FP16_DOT` flag (fp32 path structurally identical to r001; fp16 path keeps tiles to the dots with fp32 accumulator + fp32 softmax state).

### Sweep TABLE (probe-device-time = graph-assisted kernel-only CUDA-event timing at the target shape [2,83,512]; N=100 launches captured once, R=10 replays per segment, median of 3 segments; PROBE INSTRUMENTATION ONLY — not wall benchmarking of the harness route)

| Config | Compile (cuda:0) | Exactness (4 suites vs fp32 ground truth, atol=rtol=1e-2 equal_nan) | bitwise-stable | kernel-only us/call | Selectable |
|---|---|---|---|---|---|
| nw1_fp32widen (r001 config) | PASS | PASS (extreme 3.052e-05; seed42 4.883e-04; B1S41 4.883e-04; B2S96 2.441e-04) | True | 23.492 | yes (new_caps=0) |
| **nw2_fp32widen (SELECTED)** | PASS | PASS (identical max_abs to nw1) | True | **15.317** | yes (new_caps=1) |
| nw4_fp32widen | PASS | PASS (identical max_abs to nw1) | True | 15.441 | yes (new_caps=1) |
| nw1_fp16dot | PASS | **FAIL** — fp16_extreme max_abs=1459.0 (one-hot tie-flip, vendor-class signature; vendor base itself diverges 1457); randn suites PASS (4.883e-04) | True | 11.747 | NO (exactness) |
| nw2_fp16dot | PASS | **FAIL** — same extreme signature | True | 8.634 | NO (exactness) |
| nw4_fp16dot | PASS | **FAIL** — same extreme signature | True | 9.145 | NO (exactness) |

- Control arm (method validation): the r001 module's own kernel timed identically = 23.492–23.665 µs/call vs the Verifier's authoritative D_cand = 28.203 µs/call — same class (graph back-to-back replay vs the Verifier's loop overhead), and the segment spread is ±1.2%, validating the method's ranking resolution (the nw1→nw2 gap of 8.175 µs is ~35%, far beyond noise).
- Run-to-run reproduction: run 2.1 vs 2.2 gave nw1 23.573→23.492, nw2 15.262→15.317, nw4 15.450→15.441 (sub-1% variance).

### Selection rule trace (decision-002, applied exactly)

1. exactness-passing configs: nw1_fp32widen (23.492), nw2_fp32widen (15.317), nw4_fp32widen (15.441) — fp16-operand dots excluded at every warp count (exactness gate).
2. fastest = 15.317 µs/call (nw2). Tie band 0.5 µs → tied = {nw2 15.317, nw4 15.441} (Δ=0.124 µs); nw1 eliminated (8.175 µs slower).
3. fewer-new-capabilities among tied: nw2 and nw4 both new_caps=1 (`num_warps != 1`; fp32-widened dots = 0 new).
4. residual tie → lower num_warps → **SELECTED: nw2_fp32widen**.

Capability-negative datapoints recorded per dispatch expectation: fp16-operand dots COMPILE on this rig (no compile failure) but fail the exactness gate on the fp16-extreme suite with the vendor-class one-hot tie-flip signature (max_abs 1459 ≈ vendor 1457; the fp32-widened path passes the same suite at 3.052e-05) — "exactness-passing with the fp32 accumulator" is NOT satisfied, so fp16 dots stay unshipped despite being faster (8.6–11.7 µs/call). num_warps=4 buys nothing over 2 (15.441 vs 15.317, within noise).

## Candidate Gates (p14, log/probes/ — non-authoritative; NO timing)

- Compile-smoke on cuda:0 via the real candidate path: PASS.
- Correctness under the UNCHANGED comparator (allclose atol=rtol=1e-2 equal_nan):
  - seed42_B2S83 vs base: PASS, max_abs=4.883e-04
  - fp16_extreme_B2S83 vs fp32 ground truth (r001-established basis): PASS, max_abs=3.052e-05 (r001-identical)
  - seed7_B1S41 (non-target) vs base: PASS, max_abs=4.883e-04
  - seed19_B2S96 (boundary, zero padding) vs base: PASS, max_abs=4.883e-04
  - seed13_B2S82 (padded tail) vs base: PASS, max_abs=4.883e-04
- run_out poisoned caller buffers ×2 orderings (forward-first poison −777.0; run_out-first poison +555.0): BOTH bitwise-equal to forward, data_ptr preserved, returns None. Forward bitwise-stable with fresh allocation.
- STATELESS audit: instance attrs exactly the 4 constructor-config attrs (+`training` from `.eval()`), identical before/after; module level = 3 immutable int literals.
- Config audit: exactly ONE `num_warps` site with fixed value 2; 4 dot sites; 6 fp32-widening casts at the load sites (r001 structure unchanged).
- Diagnostics: r002 (nw2) output BITWISE-equal to r001 (nw1) module output on identical inputs (max_abs=0.0) — warp-count invariance of the compiled reductions; 4 distinct shapes → 4 distinct framework JIT specializations (stateless recompile routing).
- Real-harness loader gate: `auto_bench.py --v0 base.py --v1 triton_mm_encoder_attention_e2_002.py --warmup 5 --repeat 10 --full-traceback` → exit 0, `PASS accuracy` (AST-loader path exercised).

## Binding Statement

- **Selected config**: `num_warps=2`, fp32-widened dot operands — fixed module literal at the single launch site; no runtime selection; the candidate carries exactly ONE config.
- **Dot-shape audit**: 4 `tl.dot` call sites, each (32,32)@(32,32) fp32/fp32→fp32 via the 6 load-site widening casts — IDENTICAL envelope to r001 (proven); NO new dot capability shipped (fp16-operand dots were tested and rejected on exactness — recorded as capability-negative evidence, not shipped, so no out-of-old-envelope dot audit is needed for this candidate).
- **New capability shipped**: `num_warps=2` only — qualified by the round-002 pre-adoption sweep (compile PASS, exactness PASS on all 4 suites, bitwise-stable) and the p14 gates (all suites PASS, outputs bitwise-equal to the r001-proven nw1 config).
- **num_warps audit**: exactly 1 site, value 2. `num_stages` count 0. Kernel count 1.
- **DANGER tokens** (18 tokens incl. `torch.compile` family, graph family, `capture`/`replay`, `num_stages`, `.contiguous`/`contiguous`, `copy_`, `autotune`): ALL-ZERO over the final source; graphs live ONLY in the `log/probes/` instrumentation (p13 timing methodology), never in the candidate.
- **Addressing audit**: direct strided addressing of the [B,S,H*D] fp16 inputs and direct final-layout stores — unchanged from r001; the word "contiguous" does not appear anywhere in the source (grep 0, case-insensitive 0).
- **AST-loader composition**: 4 imports + 3 literal assigns (`_BM/_BN/_BD`) + `@triton.jit` FunctionDef + ClassDef + 2 helper FunctionDefs — all retained node types; real-harness loader gate PASS.

## Attempt Ledger

| Attempt | Command (abridged) | Exit | Defect | Candidate SHA |
|---|---|---|---|---|
| 1 | p13 run 1 (python launch-loop timing) | 0 | PROBE-SIDE: the loop is HOST-BOUND on this rig (Triton python launcher ~66–70 µs/call floor) — it measures launcher throughput and masks every kernel shorter than the floor (all 6 are shorter; nw1's 100.4 µs median was segment noise: [100.4, 122.0, 76.9]). Preserved as `p13_r002_sweep_result_run1_launchbound.json` + log. Timing methodology replaced, sweep re-run | n/a (probe-only) |
| 2 | p13 run 2 (graph-assisted kernel-only CUDA-event timing + r001 control arm) | 0 | none — clean data, sub-1% variance, control arm validates against Verifier D_cand=28.203 µs (23.492–23.665 same class) | n/a (probe-only) |
| 2 | p13 run 2 re-run (rows.append bug fix) | 0 | PROBE-SIDE: run 2 first executed the ALL-FAIL selection branch because the success path never appended rows (printed per-config data was correct); fixed one line, re-run — identical timing numbers, correct selection trace | n/a (probe-only) |
| 3 | author candidate `_e2_002.py` (r001 verbatim + `num_warps=2`) | — | none | `cc98318b…` (after docstring fix) |
| 3 | gates: ast.parse + 18-token DANGER scan + counts | 0 | none — all-zero; 4 dot sites; 6 widen casts; 1 num_warps site value 2 | `cc98318b…` |
| 3 | p14 candidate gates | 0 | none — first-attempt PASS (all suites, poisoned orderings, stateless, config audit; ResourceWarning on an unclosed file handle fixed probe-side, re-run PASS) | `cc98318b…` |
| 3 | harness smoke (real AST loader) | 0 | none — `PASS accuracy` | `cc98318b…` |
| 4 | docstring fix: removed the single occurrence of the word "autotune" (scan token) from the ModelNew docstring; re-ran all gates on final bytes | 0 | none | `cc98318bbfed0d9cdbf7a99769c4a44077dc15590496a4e8410ba9eda99f3078` (FINAL) |

Zero same-round candidate repairs consumed (the in-round repair budget is untouched — all fixes were probe-side or docstring-only, before classification).

## Deviations

- **D1 (observation, no code change)**: harness smoke at `--warmup 5 --repeat 10` printed `PASS accuracy; v0=0.149523 ms, v1=0.239993 ms, speedup=0.623x` (exit 0). Harness-printed observation only, NOT a coder measurement claim. Consistent with the pre-declared host-bound wall: the ~8 µs kernel-only gain (23.5→15.3 µs/call) is largely masked by the ~85 µs python launcher tax, so the primary_metric 5% bar is expected to FAIL honestly — the round's product is the capability qualification + best-config deliverable (decision expected_wall_improvement_pct 0.0). The Verifier owns the authoritative wall/D_cand/D_base attribution.
- **D2 (conformance note, timing instrumentation)**: run 1 of the p13 sweep timed configs with a plain python launch loop bracketed by CUDA events; on this rig that loop is host-bound (~66–70 µs/call launcher floor) and masks all six kernel durations — inadequate for the "fastest exactness-passing" rule. Run 2 replaced it with graph-assisted kernel-only CUDA-event timing (100 launches captured once into a graph, 10 back-to-back replays per segment, median; r001 control arm cross-validates against the Verifier's D_cand=28.203 µs). Graphs appear ONLY in the probe instrumentation — the candidate keeps the no-graph binding (DANGER scan all-zero). Both runs preserved under `log/probes/`.
- **D3 (conformance note, capability-negative datapoints)**: fp16-operand dots at (32,32) COMPILE on this rig at all warp counts but fail the exactness gate on the fp16-extreme suite (max_abs=1459.0, vendor-class one-hot tie-flip signature — the vendor base itself diverges 1457 on the same suite) — the decision's "expected error class ~1e-3" did NOT materialize; the failure class is argmax tie-flipping on exactly-tied one-hot rows under fp16-operand score math, not accumulation noise. Recorded per the dispatch's pre-declared expectation ("fp16-operand dot compile failure is an EXPECTED possible outcome" — here it compiled but failed exactness, the other expected negative branch). The fp32-widened path passes the same suite at 3.052e-05, so the shipped config is unaffected.
- **D4 (conformance note)**: num_warps=4 vs 2 showed no kernel-only gain (15.441 vs 15.317 µs, within noise) — the decision's projection band (nw4+fp32: 14–18 µs) is met by BOTH nw2 and nw4; the selection rule's tie-band + fewer-capabilities + lower-nw ordering resolves to nw2 exactly as pre-declared.
- **D5 (conformance note)**: GQA (`num_kv_heads != num_heads`), non-fp16 dtypes, non-cuda devices remain outside decision scope (r001 D3 precedent, unchanged boundary).

## Evidence for Verifier

- Candidate: `triton_mm_encoder_attention_e2_002.py` @`cc98318bbfed0d9cdbf7a99769c4a44077dc15590496a4e8410ba9eda99f3078`.
- Probe logs under `log/probes/` (hashes below). Canonical measurement route: unchanged harness, default stream, seed 42, warmup 50 / repeat 100 interleaved pairs.

### Artifact hash ledger

```text
cc98318bbfed0d9cdbf7a99769c4a44077dc15590496a4e8410ba9eda99f3078  triton_mm_encoder_attention_e2_002.py
20b360ac936bf4d9d41afadac90c40578f0a758e628ec40af2d3c759eb22d3fb  rounds/decision_002.md
c16b1528b25ae1a3bbfc72b3e459462505d940677e62b30a0585e3b41b46e9e9  rounds/sketch_002.json
4171de8dcb1df6478942b0861756d660ef7955c5741e60072f03476a5fab2fc2  triton_mm_encoder_attention_e2_001.py (r001, canonical base for this round)
86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  /root/CodeBuddy/20260818191200/kernelswift/auto_bench.py
ee7f8325c514d53f4ddb3d10aacb8bf66ca9fe2290a556d51df3b5964ea47426  log/probes/p13_r002_config_sweep.py
28088b8da80c9255a1e55b031626adc34b4a988d8612e8a2cab15192a2c8fa64  log/probes/p13_r002_config_sweep.log
9472cd8bf7fc17fda24155f56474c8dbf386b233f033e4ebac5c3f2e7dd58c4f  log/probes/p13_r002_sweep_result.json
48a64734967f221319c9285e26a4682d9559a1f99d8293dde5f9e9ebb1fedb34  log/probes/p13_r002_sweep_result_run1_launchbound.json (superseded run-1 evidence)
2178aa9aa05795d7da1f2f6fc6ff7bea1598f142fa3ae8a23be77665a429bdb9  log/probes/p13_r002_config_sweep_run1_launchbound.log (superseded run-1 evidence)
a4948c79d40c7a2c8388daaa2c24e8da14b3c8c5676e4a1cce8f14497d831ad6  log/probes/p14_r002_candidate_gates.py
942ae314a5c1ac12fe2b3bc0d0641ca7c9d272f4c9af9c942fb5c08394bca044  log/probes/p14_r002_candidate_gates.log
f137d87f31ced2ca64c5485abb3772a8d6f28ff2cec5d559521e2389eac52721  log/probes/p14_r002_gates_result.json
322d932902fb161d7f4529576db972f1a858b2a731ff31a4f8b9f2ca1bde3863  log/probes/binding_statement_report_002.json
```

## Exact Commands (all with `cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable;` prefix; interpreter `/usr/local/bin/python3`; device cuda:0)

```bash
# pre-adoption capability sweep (selection rule applied inside; machine table p13_r002_sweep_result.json)
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/p13_r002_config_sweep.py

# candidate gates (compile-smoke, unchanged-comparator suites, poisoned run_out x2 orderings, stateless, config audit)
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/p14_r002_candidate_gates.py

# gates + real-harness smoke (exit 0, PASS accuracy)
/usr/local/bin/python3 -c "import ast; ast.parse(open('kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_002.py').read()); print('AST_PARSE_OK')"
/usr/local/bin/python3 auto_bench.py \
  --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
  --v1_file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_002.py \
  --warmup 5 --repeat 10 --full-traceback
# 18-token DANGER scan (all-zero) + counts: 4 tl.dot sites; 6 .to(tl.float32) widen casts; 1 num_warps site value 2
```

Coder claims no measurement and no verdict; classification is candidate-ready. Orchestrator owns the verification dispatch.
