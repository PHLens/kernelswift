# Round Status 002 — Candidate Verification (mm_encoder_attention bi150 epoch2)

## Status: END — verification complete, classification `no-improvement` delivered

- role: verifier
- phase: verifying (Round 002) → complete (Orchestrator owns transitions)
- measurement_exclusive: honored throughout; no other commands ran on the device during pairs
- candidate: `triton_mm_encoder_attention_e2_002.py` @`cc98318bbfed0d9cdbf7a99769c4a44077dc15590496a4e8410ba9eda99f3078` (unchanged before AND after all runs; zero repairs consumed; diff-verified only functional delta vs r001 is num_warps 1→2)
- interpreter: `/usr/local/bin/python3`; device: `cuda:0` (Iluvatar BI-V150); CoreX bootstrap in every shell

## Identity checks (all confirmed live)

| Artifact | Declared | Live | Verdict |
|---|---|---|---|
| candidate sha256 | `cc98318b…3078` | same, re-verified post-run | match |
| decision_002.md | `20b360ac…d3fb` | same, re-verified post-run | match |
| sketch_002.json | `c16b1528…9e9e` | same | match |
| report_000.md | `20b21646…7ffc` | same | match |
| report_001.md | `13adafe9…449c` | same | match |
| baseline_adapter.py | `c3980a2c…1c9f` | same | match |
| base.py | `86ac5703…6ed2` | same, re-verified post-run | match |
| auto_bench.py | `71fb3ad0…fe29` | same, re-verified post-run | match |
| triton_mm_encoder_attention_e2_001.py | `4171de8d…fc2` | same | match |
| binding_statement_report_002.json | (not dispatch-declared) | `322d9329…3863` | recorded |
| measurement fingerprint | `0c4c7d66…966e` | recomputed live = same | match |
| r001→r002 delta | num_warps 1→2 only | diff-verified: `num_warps=1,`→`num_warps=2,` at the single launch site; kernel arithmetic textually identical; other deltas docstrings/comments | match |

## Completed commands

1. Hash ledger (10 artifacts) + measurement-fingerprint recompute + r001→r002 diff verification — PASS.
2. Verifier correctness probe — ALL PASS first attempt (no probe-side fixes needed): seed42 max_abs 4.883e-04; fp16-extreme vs fp32 ground truth 3.052e-05 (r001-identical); B1S41/B2S96/B2S82 PASS; run_out poisoned ×2 bitwise both orderings with data_ptr preserved; forward bitwise-stable; stateless; 9 DANGER tokens zero; 4 dot sites (32,32) fp32; single num_warps=2 site; 6 widen casts; r002 outputs BITWISE-equal to r001 on every suite (log/probes/verifier_r002_result.json).
3. Kernel-mode profile attempt — standing D1 verbatim: `TypeError: ModelNew.run_out() missing 2 required positional arguments: 'value' and 'out'`, exit 1 (log/r002_kernel_mode_attempt.txt).
4. Authoritative timing pair 1/3 — PASS accuracy; v0=0.146358 ms, v1=0.232103 ms (log/r002_pair_001_timing.txt).
5. Authoritative timing pair 2/3 — PASS accuracy; v0=0.144984 ms, v1=0.231037 ms (log/r002_pair_002_timing.txt).
6. Authoritative timing pair 3/3 — PASS accuracy; v0=0.144069 ms, v1=0.231689 ms (log/r002_pair_003_timing.txt).
7. Dual-scope forward-mode profile pw=20/pi=100 — harness exit 0; extra correctness PASS (v0=0.145125, v1=0.231593); trace log/r002_forward_100iter.pt.trace.json sha256 `1eb734f9…50e4` (log/r002_profile_run.txt).
8. Canonical summarize per scope — reference OK (log/r002_summary_reference.json: 15.3582 µs/call, 0.88/call); candidate scope standing D2 (exit 2, overlap) → census substitution log/probes/verifier_r002_scope_census.py → log/diagnostic_scope_census_round002.json.
9. Census + decomposition — AUTHORITATIVE D_cand(nw2) = 19.5550 µs/call (GPU-span projection 100/100; whole-trace identical; vendor 17.4212 whole-trace / 15.3582 attributed); T_launcher net +84.5712 µs (whole-trace; r001 +84.7651, −0.19 µs; +80–90 invariance band PASS); host invariance: 1.00 aten::empty, 1.00 cuLaunchKernel (cuda_driver), 0 memcpys/graphs/syncs — all identical to r001; F2 projection net_F2 = 19.555 − 16.455 = +3.10 µs/call worse than base (sub-parity; parity band 16.5 short by 3.06; win band 9.2 short by 10.36).
10. Post-measurement hash re-verification — all frozen artifacts unchanged.
11. rounds/report_002.md written (8-observable contract mirror + vNext fact pack); rounds/verdict_002.json written, fact-pack hash re-pinned (`d15b1a81…faa8`), validated with the skill's own `validate_verdict` (valid: True, classification none, terminal no-improvement); state/verifier_context.md updated.

## Raw samples

- reference_raw_samples_ms: [0.146358, 0.144984, 0.144069] → median 0.144984
- candidate_raw_samples_ms: [0.232103, 0.231037, 0.231689] → median 0.231689
- improvement_pct: −59.803151 (decisively below the +5.0% bar; wall INSIDE the pre-declared honest band 0.222–0.241 ms)
- session drift: v0 −3.4399% vs r000 / −0.2690% vs r001; paired same-session basis absorbs; ABAB not needed

## Terminal classification

`no-improvement` (streak 2/3; canonical unchanged) — report at rounds/report_002.md @`bb46dee71b12e8fb5289fbe3a7419e18cbd26e8f4bee5de3dc01b84f6354e1d5`; verdict at rounds/verdict_002.json @`a86c2e8cd059b1a71439664c296eb5b660b56e8d8b4aa1bea93927480190d9d9` (file sha; canonical fact-pack hash `d15b1a81…faa8` inside).

Key numbers for the r003 gate: AUTHORITATIVE D_cand(nw2) = 19.5550 µs/call (the dispatch's ~15–16 parity-plus branch did NOT materialize on the attributed basis; method delta probe-vs-attributed consistent at +4.2–4.7 µs); F2 composition projection = +3.10 µs/call WORSE than base (sub-parity 0.94–0.96x composed class); T_launcher invariance PASS (+84.57 µs, +80–90 band); host census fully invariant vs r001; deliverable ledger: banked deliverable improves to @cc98318b at 0.6258x (was 0.6033x), outputs bitwise-equal to r001.

## Next safe action

Orchestrator validates report_002.md + verdict_002.json gates, applies no-improvement transitions (performance_miss_streak 2/3, canonical unchanged, deliverable ledger → nw2 config), and dispatches round 003 (F2 composition per the staged plan — or honest close-out, Designer's call with the sub-parity projection as input). Verifier idle until next dispatch.
