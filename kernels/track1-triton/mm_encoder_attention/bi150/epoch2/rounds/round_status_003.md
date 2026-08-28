# Round Status 003 — FINAL Candidate Verification (mm_encoder_attention bi150 epoch2)

## Status: END — verification complete, classification `accepted` delivered (BOUNDARY-CLASS)

- role: verifier
- phase: verifying (Round 003 — FINAL ROUND) → complete (Orchestrator owns transitions: streak reset, canonical pointer, campaign close-vs-continue)
- measurement_exclusive: honored throughout; no other commands ran on the device during pairs
- candidate: `triton_mm_encoder_attention_e2_003.py` @`d503e845d1d95ad033e616730d78d67d0775df3a473b17a3c37843d463654d81` (unchanged before AND after all runs; zero repairs consumed; kernel segment byte-identical to r002, verifier-verified 4168/4168 chars)
- interpreter: `/usr/local/bin/python3`; device: `cuda:0` (Iluvatar BI-V150); CoreX bootstrap in every shell

## Identity checks (all confirmed live)

| Artifact | Declared | Live | Verdict |
|---|---|---|---|
| candidate sha256 | `d503e845…4d81` | same, re-verified post-run | match |
| decision_003.md | `0a678da8…8413` | same, re-verified post-run | match |
| sketch_003.json | `bdf42355…4e64` | same | match |
| report_000.md | `20b21646…7ffc` | same | match |
| baseline_adapter.py | `c3980a2c…1c9f` | same | match |
| base.py | `86ac5703…6ed2` | same, re-verified post-run | match |
| auto_bench.py | `71fb3ad0…fe29` | same, re-verified post-run | match |
| triton_mm_encoder_attention_e2_002.py | `cc98318b…3078` | same | match |
| binding_statement_report_003.json | `4b3985a8…7c9b6` | same | recorded |
| kernel byte-identity vs r002 | 4168/4168 chars | verifier-verified extraction-diff: KERNEL_BYTE_IDENTICAL | match |
| measurement fingerprint | `0c4c7d66…966e` | recomputed = same | match |

## Completed commands

1. Hash ledger (9 artifacts) + measurement-fingerprint recompute + kernel byte-identity machine-verification — PASS.
2. Verifier correctness probe — ALL PASS first attempt: seed42 max_abs 4.883e-04; fp16-extreme vs fp32 GT 3.052e-05; boundary suite; non-target shapes through tier-3 with zero graph artifacts; **6-way bitwise** (tier-1 ×2 / tier-2 / r002 twin / run_out poisoned ×2) on every target-regime suite; stale-address impossibility both directions; bounded-state audit exact; 9 forbidden tokens zero; 4 dots / 6 widens / single num_warps=2 site (log/probes/verifier_r003_result.json).
3. Kernel-mode profile attempt — standing D1 verbatim: `TypeError: ModelNew.run_out() missing 2 required positional arguments: 'value' and 'out'`, exit 1 (log/r003_kernel_mode_attempt.txt).
4. Authoritative timing pairs 1–3 (prescribed): PASS accuracy ×3; v0=[0.147793, 0.149939, 0.194862 (ref transient)], v1=[0.140942, 0.142966, 0.142327] → protocol medians 0.149939/0.142327 (log/r003_pair_00{1,2,3}_timing.txt).
5. Boundary-rigor extra pairs 4–8: v0=[0.144358, 0.144063, 0.149326, 0.158449, 0.151585], v1=[0.137093, 0.137414, 0.143134, 0.137038, 0.143989] — candidate faster in ALL 8 invocations (log/r003_pair_00{4..8}_timing.txt).
6. Dual-scope forward-mode profile pw=20/pi=100 — harness exit 0; extra correctness PASS (v0=0.149817, v1=0.144134); trace log/r003_forward_100iter.pt.trace.json sha256 `1c47a5f9…9df47` (log/r003_profile_run.txt).
7. Canonical summarize per scope — reference OK (15.8471 µs/call, 0.93/call); candidate scope SUCCEEDED but attribution-coarsened (0.07/call vendor leakage; graph-interior kernels emit NO kernel events on this build — D2′ new form) → census substitution log/probes/verifier_r003_scope_census.py → log/diagnostic_scope_census_round003.json.
8. Dedicated in-graph probe log/probes/verifier_r003_kernel_in_graph.py — single-launch graph replay round-trip 64.4673 µs/call event-timed (direct-launch control 18.3654; idle sync overhead 15.2122).
9. Four closing censuses — (a) tier-1 100% engaged (1.00 cudaGraphLaunch + 1.00 memcpyAsync + 0.99 DtoD, 0 cuLaunchKernel serving, 3.00 aten, 0 copy-ins, 0 timed recaptures); (b) R-term(bsz=2) = 65.76 µs/call API-sum vs sibling 69.02 → TRANSFERS within 3.42 µs; (c) in-graph round-trip 64.47 µs vs direct 18.37/19.555 — frontend dominates, p13 amortization resolved; (d) SUBMISSION = r003 composed (1.0486–1.0565x dominates r002's 0.6258x on every estimator).
10. Post-measurement hash re-verification — all frozen artifacts unchanged.
11. rounds/report_003.md written (8-observable mirror + fact pack + campaign physics closure + loud marginality/falsification); rounds/verdict_003.json written, fact-pack hash re-pinned (`b5e497b2…56d2`), validated with the skill's own validate_verdict (valid: True, **terminal_result accepted**); state/verifier_context.md updated.

## Raw samples

- reference_raw_samples_ms (8): [0.147793, 0.149939, 0.194862, 0.144358, 0.144063, 0.149326, 0.158449, 0.151585]
- candidate_raw_samples_ms (8): [0.140942, 0.142966, 0.142327, 0.137093, 0.137414, 0.143134, 0.137038, 0.143989]
- improvement_pct (protocol, 3 prescribed pairs): **+5.076731** ≥ 5.0 bar (cleared by 0.077 pp — BOUNDARY-CLASS)
- estimator table: 8-pair median +5.3451 (PASS) / 5-pair +4.6355 (FAIL) / clean per-pair mean +4.679 (FAIL) / win rate 8/8
- net: −7.612 µs/call (protocol) / −7.998 µs (8-pair); speedup 1.0535x (protocol) / 1.0565x (8-pair)

## Terminal classification

**`accepted` (BOUNDARY-CLASS)** — report at rounds/report_003.md @`1c23f13e3285f99a5f5e4ebda8bc0597b498b7bf28d950101f6b22d72c680399`; verdict at rounds/verdict_003.json @`29f1f8e909cd6cfd5cb570c9542ba8323bd1f0f9298c05d1a767fcbece2f283d` (file sha; canonical fact-pack hash `b5e497b2…56d2` inside). The decision's win branch (pre-declared reading (a)) FIRED: wall ≥ +5% with tier-1 hit-rate 100%. The priced identity was falsified in the favorable direction (predicted 0.978–1.007x; measured 1.0508–1.0565x). SUBMISSION = r003 composed @d503e845 (0.6258x → ~1.05x). Streak resets; canonical-pointer transition and campaign close-vs-continue belong to the Orchestrator (boundary-class caveat flagged).

## Next safe action

Orchestrator validates report_003.md + verdict_003.json gates and applies the accepted transitions (performance_miss_streak reset, last_accepted pointer decision weighing the boundary-class marginality, submission snapshot of the composed deliverable per the DELIVERABLE RULE), then decides campaign close-at-high-point vs continuation on the ~35 µs allocation-boundary lever. Verifier idle until next dispatch.
