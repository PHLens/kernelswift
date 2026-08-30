# Round Status 001 — Candidate Verification (mm_encoder_attention bi150 epoch2)

## Status: END — verification complete, classification `no-improvement` delivered

- role: verifier
- phase: verifying (Round 001) → complete (Orchestrator owns transitions)
- measurement_exclusive: honored throughout; no other commands ran on the device during pairs
- candidate: `triton_mm_encoder_attention_e2_001.py` @`4171de8dcb1df6478942b0861756d660ef7955c5741e60072f03476a5fab2fc2` (unchanged before AND after all runs; zero repairs consumed)
- interpreter: `/usr/local/bin/python3`; device: `cuda:0` (Iluvatar BI-V150); CoreX bootstrap in every shell

## Identity checks (all confirmed live)

| Artifact | Declared | Live | Verdict |
|---|---|---|---|
| candidate sha256 | `4171de8d…fc2` | same, re-verified post-run | match |
| decision_001.md | `67b96739…6c78` | same | match |
| sketch_001.json | `a1c27dba…2363` | same | match |
| report_000.md | `20b21646…7ffc` | same | match |
| baseline_adapter.py | `c3980a2c…1c9f` | same | match |
| base.py | `86ac5703…6ed2` | same, re-verified post-run | match |
| auto_bench.py | `71fb3ad0…fe29` | same, re-verified post-run | match |
| binding_statement_report_001.json | `623783fd…0b2f` | same | match |
| measurement fingerprint | `0c4c7d66…966e` | recomputed live = same | match |

## Completed commands

1. Hash ledger (9 artifacts) + measurement-fingerprint recompute — PASS.
2. Verifier correctness probe — run 1 had two PROBE-SIDE expected-value defects (widen-cast threshold 8→6; extreme-suite basis base→fp32 ground truth after diagnosing the vendor kernel's own fp16 saturation at |score| ≤ 2.4e7 as the divergence source); fixed probe-side; re-run ALL PASS: seed42 max_abs 4.883e-04, fp16-extreme candidate-vs-ground-truth max_abs 3.052e-05 (vendor diverges 1457 from ground truth — vendor precision limit, not candidate defect), B1S41/B2S96 PASS, run_out poisoned ×2 bitwise both orderings with data_ptr preserved, forward bitwise-stable, stateless attr delta empty, 9 DANGER tokens zero, 4 dot sites (32,32) fp32, single num_warps=1 site (log/probes/verifier_r001_result.json).
3. Kernel-mode profile attempt — failed as structurally predicted (standing D1): `TypeError: ModelNew.run_out() missing 2 required positional arguments: 'value' and 'out'`, exit 1 (log/r001_kernel_mode_attempt.txt).
4. Authoritative timing pair 1/3 — PASS accuracy; v0=0.145375 ms, v1=0.238943 ms (log/r001_pair_001_timing.txt).
5. Authoritative timing pair 2/3 — PASS accuracy; v0=0.145466 ms, v1=0.242051 ms (log/r001_pair_002_timing.txt).
6. Authoritative timing pair 3/3 — PASS accuracy; v0=0.145368 ms, v1=0.240953 ms (log/r001_pair_003_timing.txt).
7. Dual-scope forward-mode profile pw=20/pi=100 — harness exit 0; extra correctness PASS (v0=0.144873, v1=0.265107); trace log/r001_forward_100iter.pt.trace.json sha256 `3ce1e0d8…4803` (log/r001_profile_run.txt).
8. Canonical summarize per scope — reference OK (log/r001_summary_reference.json: 15.6853 µs/call, 0.90/call); candidate scope fails `overlapping scope events` (kineto dual-span shape, standing D2 class) → census substitution log/probes/verifier_r001_scope_census.py → log/diagnostic_scope_census_round001.json.
9. Host census + dual-gate decomposition — aten 33 → 1.00/call; 1.00 cuLaunchKernel/call (cuda_driver); 0 memcpys/graphs/syncs; D_cand = 28.2030 µs/call (100/100 GPU-projection attributed, whole-trace identical); D_ref = 17.3901 (whole-trace) / 15.6853 (attributed); T_launcher = +84.7651 µs/call net (Δwall 95.578 − Δdevice 10.813; attributed-basis +83.0603 corroborates).
10. Post-measurement hash re-verification — all frozen artifacts unchanged.
11. rounds/report_001.md written (with vNext Fact Pack); rounds/verdict_001.json written and validated with the skill's own `validate_verdict` (valid: True, classification none, terminal no-improvement); state/verifier_context.md updated.

## Raw samples

- reference_raw_samples_ms: [0.145375, 0.145466, 0.145368] → median 0.145375
- candidate_raw_samples_ms: [0.238943, 0.242051, 0.240953] → median 0.240953
- improvement_pct: −65.745830 (decisively below the +5.0% bar; wall inside the decision's pre-declared 0.235–0.29 launcher-tax-transfers band)
- session drift: v0 −3.1795% vs r000 (0.145375 vs 0.150149); paired same-session basis absorbs; ABAB not needed (delta ~13× the bar, ~20× the drift)

## Terminal classification

`no-improvement` (streak 1/3; canonical unchanged) — report at rounds/report_001.md; verdict at rounds/verdict_001.json @`b6e62fdbf3757370449ca016742d202a2da6cd62f51f55de90d2dd9f23746822` (file hash; canonical fact-pack hash `b7eb0b45…bb17` inside); deliverable banked per project.md DELIVERABLE RULE: correctness-PASS Triton candidate with forward + 4-arg run_out surfaces at 0.603x.

Dual-gate highlights: T_launcher +84.77 µs/call at bsz=2 (sibling +86–89 at bsz=1 transfers in full — falsification target CONFIRMED); D_cand 28.20 µs/call (band (b) exactly); F2 gate OPEN on both conditions (≥50, ≤35) but parity band (~18) NOT reached (+10.2) and win band (~10) NOT reached (+18.2); graph-family measured-number projection +11.75 µs/call worse than base.

## Next safe action

Orchestrator validates report_001.md + verdict_001.json gates, applies no-improvement transitions (performance_miss_streak 1/3, canonical unchanged), and dispatches the next round. Verifier idle until next dispatch.
