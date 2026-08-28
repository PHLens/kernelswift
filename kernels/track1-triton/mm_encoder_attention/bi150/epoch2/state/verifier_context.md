# Verifier Context

> Durable compact context per `references/role-context-template.md`. Updated by Verifier only.

- role_contract_sha256: `62f10a0940ca3665260226a7891f5d34e1b571e70937862bb02ad68aa2bbc82f` (verifier.md loaded in-session rounds 000–003)
- context_epoch: `5`
- last_completed_round: `003` (accepted BOUNDARY-CLASS — the campaign's declared FINAL ROUND; report + verdict emitted; Orchestrator owns transitions: streak reset, canonical pointer, close-vs-continue)
- accepted_kernel: `baseline_adapter.py` @`c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f` (pending Orchestrator decision on moving the pointer to r003 given the boundary-class pass)
- accepted_report: `rounds/report_000.md` @`20b21646d9c3ba3abe086d8133799d23a39981dcb4e1cb547e1a3f65b0bf7ffc`
- banked_deliverable (SUBMISSION per project.md DELIVERABLE RULE): `triton_mm_encoder_attention_e2_003.py` @`d503e845d1d95ad033e616730d78d67d0775df3a473b17a3c37843d463654d81` — composed three-tier graph-replay Triton submission at **1.0486–1.0565x vs base** (first candidate to beat base; up from r001 0.6033x → r002 0.6258x), correctness-PASS with 6-way bitwise retention and the full r001/r002 numerics pedigree
- recent_three_round_evidence: `r001 no-improvement (direct-Triton: −65.75%, T_launcher +84.77 µs/call, D_cand 28.203) → r002 no-improvement (nw2 config: −59.80%, D_cand 19.555 attributed, host fully invariant, T_launcher +84.57) → r003 ACCEPTED boundary-class (F2 composition: +5.0767% protocol / 1.0535x, 8/8 win rate; tier-1 100% engaged; R-term 65.76 TRANSFERS; in-graph round-trip 64.47 µs event-timed; priced identity falsified favorably by ~7 µs/call)`
- open_hypotheses: `campaign terminal pending Orchestrator: close at the ~1.05x high point (physics map complete) vs continue on the ~35 µs allocation-boundary lever (aten empty_like+empty_strided = 34.7 µs/call of the 142.3 µs composed wall) — successor rounds must not assume cushion over the 5% bar (0.077 pp margin)`
- artifact_read_hashes: see table

## Current Bottleneck (final campaign physics map — all measured this campaign)

- **Direct family (closed)**: Triton python launcher tax +84.6 µs/call (invariant across rounds; measured r001/r002) dominates the direct wall — arithmetically closed for the ≥5% bar (needs D ≤ −75.7 µs).
- **Device floor**: 28.203 (nw1) → 19.555 (nw2, attributed) µs/call direct; kernel math ~18.4 µs event-timed at the proven (32,32) fp32 envelope — the in-policy floor. fp16-operand dots exactness-NEGATIVE; nw4 no gain.
- **Composed family (r003, measured)**: wall 142.3 µs = aten 55.4 (alloc 34.7 + copy_ 20.7) + graphLaunch API 6.1 + sync 65.6 (R-term: 15.2 idle overhead + wait absorbing the 64.5 µs single-launch graph round-trip) + memcpyAsync 4.9 + DtoD 5.4 + glue. The graph frontend (~46 µs over the kernel math) and the R-term sync are build-intrinsic; the ~35 µs fresh-destination allocation is the only remaining attributable host lever.
- **R-term at bsz=2**: 65.76 µs/call API-sum — TRANSFERS from sibling bsz=1 (69.02) within 3.42 µs.
- **Kineto**: graph-interior kernels emit NO kernel events on this build (D2′) — attribution via API census + CUDA events.

## Recent Three-round Evidence

- Round 001 (no-improvement): F1 direct-Triton deliverable banked @4171de8d (0.6033x); T_launcher +84.7651 µs/call canonical; D_cand(nw1) 28.2030 µs/call. Evidence: rounds/report_001.md.
- Round 002 (no-improvement): F3 kernel-config nw1→nw2; D_cand(nw2) 19.5550 µs/call attributed (−30.66%); host fully invariant; T_launcher +84.5712 (band PASS); capability matrix closed (fp16 dots negative, nw4 no gain); deliverable improved to @cc98318b (0.6258x). Evidence: rounds/report_002.md.
- Round 003 (ACCEPTED, boundary-class): F2 composition; protocol statistic +5.0767% (3 prescribed pairs, unrounded median — the contract's rule) with 8/8 win rate and estimator straddle documented (8-pair +5.345 PASS / 5-pair +4.636 FAIL / clean-mean +4.679 FAIL); four closing censuses: tier-1 100% engaged (0 launcher executions), R-term 65.76 TRANSFERS, in-graph round-trip 64.47 µs (frontend dominates; p13 15.317 was 100-launch amortized), SUBMISSION = r003 composed (1.049–1.057x); priced identity falsified favorably (predicted 0.978–1.007x; measured 1.051–1.056x). Evidence: rounds/report_003.md, rounds/verdict_003.json @29f1f8e9….

## Open Hypotheses or Checks

- Orchestrator decisions pending: canonical pointer (boundary-class acceptance), campaign close-vs-continue, submission snapshot of the composed deliverable.
- If continued: the allocation-boundary lever (~34.7 µs/call empty_like+empty_strided) is the largest remaining attributable host term; the successor decision must treat the 5% bar margin as ~0 (this round's cushion was 0.077 pp on the protocol statistic).
- Standing build facts (final): D1 kernel-mode arity (4-arg run_out vs harness 2-arg call); D2′ kineto graph-interior blindness (census substitution via API census + CUDA events — the branch-B extreme); reference-scope attribution margin 0.88–0.93/call; single-window host transients occur on EITHER side (~1 in 8 invocations this session — pair-3 ref, pair-7 ref elevated); session drift ±3% on the v0 side across the day.
- Measurement-method lessons (canonical): single-launch graph replay round-trip ≠ amortized per-kernel time (64.47 vs 15.317 — the p13 class of probe measures multi-launch amortization, not the serving regime); event-timed direct ≈ attributed (18.37 vs 19.555); idle-sync overhead 15.21 µs on this build.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| triton_mm_encoder_attention_e2_003.py | d503e845d1d95ad033e616730d78d67d0775df3a473b17a3c37843d463654d81 | 003 |
| triton_mm_encoder_attention_e2_002.py | cc98318bbfed0d9cdbf7a99769c4a44077dc15590496a4e8410ba9eda99f3078 | 003 |
| triton_mm_encoder_attention_e2_001.py | 4171de8dcb1df6478942b0861756d660ef7955c5741e60072f03476a5fab2fc2 | 002 |
| baseline_adapter.py | c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f | 003 |
| ../../base.py | 86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2 | 003 |
| auto_bench.py | 71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29 | 003 |
| rounds/decision_003.md | 0a678da87a877b9c521b6c280eb3518b20f98e352786e9df129435e2cc918413 | 003 |
| rounds/sketch_003.json | bdf423556e7c80369ae38d4980529a739a52a3d18033e572927354b23e0a4e64 | 003 |
| rounds/report_000.md | 20b21646d9c3ba3abe086d8133799d23a39981dcb4e1cb547e1a3f65b0bf7ffc | 003 |
| rounds/report_001.md | 13adafe951df94bb7bb74294e195cfffc6992057d36e958801b293ab292f449c | 002 |
| rounds/report_002.md | bb46dee71b12e8fb5289fbe3a7419e18cbd26e8f4bee5de3dc01b84f6354e1d5 | 002 |
| rounds/report_003.md | 1c23f13e3285f99a5f5e4ebda8bc0597b498b7bf28d950101f6b22d72c680399 | 003 |
| rounds/verdict_003.json | 29f1f8e909cd6cfd5cb570c9542ba8323bd1f0f9298c05d1a767fcbece2f283d | 003 |
| log/probes/binding_statement_report_003.json | 4b3985a81b134cc947ae2cbaf1436e67885365a9be54fda8aef6961e5779c9b6 | 003 |
| log/r003_forward_100iter.pt.trace.json | 1c47a5f91b7eb5f433ec3884b2a04511219b2af794ca3ead2dcbcf2919e9df47 | 003 |
| log/diagnostic_scope_census_round003.json | (this round's census; see file) | 003 |
| log/probes/verifier_r003_kernel_in_graph.json | (this round's in-graph probe; see file) | 003 |
| skills/kernel-opt-loop/scripts/summarize_trace.py | f625276c05a4539f86f272f68d78dd84ba0c94bb4ce153613dc4437df432148c | 003 |
