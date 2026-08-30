# Coder Context

- role_contract_sha256: `26c40a94bacbbe5ac4cf12b330516b0439a823e7ca8fd648bdace3fdfcce9cba` (re-verified live, unchanged)
- context_epoch: `4`
- last_completed_round: `003` (FINAL ROUND of the campaign; classification: candidate-ready)
- accepted_kernel: `null` (canonical pointer remains baseline_adapter.py @c3980a2c… — Orchestrator-owned)
- accepted_report: `rounds/report_000.md`
- selected_profile: `triton_cuda` (profile_snapshot/triton_cuda.yaml @dc8fa4c0…; capability_claim aeba3a87…)
- runtime_fingerprint: `torch 2.7.1 / triton 3.1.0 (corex-4.4.0) / Iluvatar BI-V150 sm71 mp16 / cuda:0 / COREX_VERSION=4.4.0 bootstrap` — matches project.md
- candidate_001: `triton_mm_encoder_attention_e2_001.py` @`4171de8d…fc2` — dispatch-collapse single-kernel deliverable (banked 0.603x wall; correctness all-green)
- candidate_002: `triton_mm_encoder_attention_e2_002.py` @`cc98318b…78` — nw2 best-config deliverable (banked 0.6258x wall, D_cand=19.555 us authoritative, outputs bitwise-equal to r001; fp16-operand dots capability-NEGATIVE on exactness)
- candidate_003: `triton_mm_encoder_attention_e2_003.py` @`d503e845d1d95ad033e616730d78d67d0775df3a473b17a3c37843d463654d81` — FINAL: three-tier graph-replay composition (family graph-replayed-triton-direct-address) around the kernel BYTE-IDENTICAL to r002 (machine-verified extraction-diff 4168/4168): tier-1 manual CUDAGraph direct-address replay (caller-pointer-bound, zero copy-ins, 3x data_ptr guard + ONE replay + one ~166KB copy-out per call; bounded recapture <=4 lifetime, first-seen sets only), tier-2 copy-in replay, tier-3 r002 eager; permanence monotone; zero model-code sync/query; results never graph-resident
- r003_probe_state: p13 (harness-sequence reproduction: initial binding budget-free on correctness call -> exactly ONE warmup recapture -> timed segment 10/10 replay-served, ZERO launcher executions) PASS; p14 (capture-fired f1-f6 all True incl. zero sync/query across 10 serves) PASS; p15 (recapture ledger exact, overflow->tier-2, same-set revisit never re-binds) PASS; p16 (all three tier edges permanent-once with same-call bitwise correctness, vectors A/A2/B/C) PASS; p17 (run_out poisoned x2 orderings bitwise on all 3 tiers, fresh-buffer proof, cross-tier retention) PASS; p18 (12 interleaved serves, 4 instances x 3 rounds, no contamination) PASS; p19 (6-way bitwise tier-1/2/3+run_out x2+twin on seed42/boundary/extreme; max_abs 4.883e-04/4.883e-04/3.052e-05 r001/r002-identical; non-target B1S41/B2S82/B2S96 tier-3 zero-artifacts) PASS — ALL SEVEN first-attempt PASS
- r003_harness_smoke_observation: PASS accuracy exit 0, v1=0.180555 ms (0.863x vs baseline 0.155737; r002 smoke was 0.623x) — composition ENGAGED at smoke scale; below the decision's pre-declared 0.94-1.01x paired-median band; NOT extrapolated by coder (Verifier's warmup-50/repeat-100 paired medians + census decide; expected honest verdict = no-improvement #3 => campaign auto-termination with the composed DELIVERABLE-RULE submission banked)
- recent_three_round_evidence: `(r001) 0.603x dispatch-collapse banked; (r002) 0.6258x nw2 best-config banked, fp16 dots capability-NEGATIVE (exactness tie-flip 1459), nw exhausted; (r003) composed graph-replay candidate all-green — the campaign's PRIMARY contractual product per the DELIVERABLE RULE (correctness-PASS Triton submission at the composed boundary), sibling-precedent architecture (flexattention e2 r003) replicated at bsz=2`
- open_hypotheses: `none from coder seat; the round's named observables (tier1_hit_rate_in_timed_regime, submission/sync census, rterm_transfer_at_bsz2, kernel-in-graph device regime 15.3-19.6 us adjudication, paired wall vs the 0.94-1.01x band) are Verifier-owned`
- artifact_read_hashes: `decision_003 0a678da8…; sketch_003 bdf42355…; decision_002 20b360ac…; baseline_adapter c3980a2c…; base 86ac5703… (immutable); harness 71fb3ad0… (unchanged); sibling template triton_flexattention_e2_003.py @6ffb0c94… (read-only)`

## Current Bottleneck

- `Campaign-closing composition: the direct family's wall arithmetic was closed at r002 (T_launcher=+84.57 us invariance-band PASS; win needs D_cand <= -75.7 us — unreachable); r003 spends the final round composing the two proven mechanisms (byte-identical r002 kernel + manual graph replay) to bank the best Triton submission (0.94-1.01x class expected, smoke read 0.863x) and close the graph-family physics map at bsz=2 (R-term transfer, boundary terms, kernel-in-graph regime). Verifier adjudicates; expected honest verdict = no-improvement #3 => clean auto-termination.`

## Recent Three-round Evidence

- `r001 (banked 0.603x): dispatch-collapse single-kernel, correctness all-green.`
- `r002 (banked 0.6258x): nw2 kernel-only 15.3-19.6 us, bitwise-equal outputs, capability matrix closed (fp16 dots NEGATIVE, nw exhausted).`
- `r003 (candidate-ready): composed three-tier graph-replay @d503e845 — kernel byte-identical to r002, all 7 behavioral probes first-attempt PASS, 6-way bitwise retention through every tier, harness smoke PASS accuracy 0.863x with the composition engaged.`

## Open Hypotheses or Checks

- `(one same-round repair budget available for r003 if Verifier requests; zero consumed — all probes and gates were first-attempt PASS)`
