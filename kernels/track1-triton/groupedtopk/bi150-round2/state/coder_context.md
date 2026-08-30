# Coder Context

- role_contract_sha256: `26c40a94bacbbe5ac4cf12b330516b0439a823e7ca8fd648bdace3fdfcce9cba`
- context_epoch: `6`
- last_completed_round: `005` (coding phase; result returned, canonical pointers unchanged pending Verifier)
- accepted_kernel: `triton_grouped_topk_r2_004.py` @`c02d956c6bb5c27c229623b01b99b85f5962db79b5ead09df6fbca7a52e721eb`
- accepted_report: `rounds/report_004.md` (r004 accepted: paired 0.474386→0.196909 = +58.50% protocol basis, +42.54% direct vs r002; manual graph capture FIRED; host census ~3 aten::copy_ trips/call — the lever r005 coalesces)
- recent_three_round_evidence: `r002 default-compile compression; r003 inductor replay refused (retired); r004 manual-workspace replay accepted with single-submission boundary; r005 merges the two output copy-out dispatches into one batched trip`
- open_hypotheses: `H-005 boundary-dispatch-coalescing implemented in candidate triton_grouped_topk_r2_005.py (sha256 cf68ed7713269416af5b49e901e040c7dcb97da9ec4f6eb4cc9bc5d70d288e9c); expected 6%; awaiting Verifier measurement under fingerprint 8deb1b01...`
- artifact_read_hashes: decision_005=4a549653…6021; sketch_005=21d13b98…92de; r004_source=c02d956c…721eb; profile_snapshot/triton_cuda.yaml=dc8fa4c0…b7ae

## Current Bottleneck

- `Verifier-backed facts carried through rounds 000-004: after r004 wall ≈196.9 µs/call with ~93 µs host share and exactly ~3 aten::copy_ boundary trips/call; vendor pair ~87 µs remains tie-gate-locked.`

## Recent Three-round Evidence

- `historical ../bi150/rounds/report_000..009 epoch-1 prior; rounds 001-002 chain compressions; r003 retired; r004 manual replay accepted — round 005 trims its remaining per-call host residue`

## Round 005 Coding State

- classification written: `candidate-ready` → `rounds/coder_result_005.md` (no major deviation; probe-side repair disclosed: strategy-parity emulation corrected to genuine construction-time anomaly injection)
- candidate deltas on byte-frozen r004 architecture: (a) construction-time torch._foreach_copy_ bind w/ mixed int64→int32 parity exercise + real-pair byte-parity probes; ANY anomaly records error artifact and pins LEGACY two-copy path permanently while tier-1 SURVIVES (proven end-to-end); binding fixed at construction, never revisited at runtime; (b) non_blocking=True on all three boundary copies (no added syncs anywhere); (c) hot callable bound post-capture over pre-resolved handles (copy-in → ONE replay → batched copy-out), stale-invalidation through a SINGLE failure handler clearing hot+graph+workspace together on EVERY tier transition; (d) guard micro-trim via constructor-derived tuples; selectivity EQUIVALENCE machine-proven against the inherited fast∧regime/fast/eager composition
- machine proofs under log/probes/ (no timing/profiler usage beyond bounded sanity):
  - coder_smoke_result.json @1b7adf6d346622be58f6a2ba2e6823e25dd10f9d4dbcd55d48edf2575014537e — 24/24 PASS incl. capability branch A recorded; bitwise-vs-r004 sweep seed42+4 tie suites+new-bytes ALL through active tier; three edge exercises (construction-failure / hot-swap invalidation / stacked cascade) each permanent-once w/ poisons hit exactly once; T=41 zero artifacts then recovery; run_out poisoned BOTH orderings in-place bitwise==r004 + leak-trap; legacy-strategy genuine anomaly exercise bitwise==r004; alternation clean
  - boundary_trip_census.json @e289a5911011e33f32d8cd43631da6aceedf4315a469a1cdf1eb6be1d161e15c — branch A trips=2/call (1 input copy-in + 1 batched copy-out) vs branch B=3; static foreach/non_blocking counts recorded
  - binding_statement_report.json @b28abf7200c1a904fb0bf56233e1b4ba2f4a1c315e1369ab8d43c9b624f0535e — seven frozen segments byte+AST identical; named-delta isolation; reduce-overhead ×0; allowlist all-zero; one compile site {default,dynamic False}
- open local checks: none outstanding; same-round repair budget untouched (0 verifier-requested)

## Open Hypotheses or Checks

- H-005 awaiting authoritative Verifier wall comparison (expected ≥5% unrounded paired median improvement vs r004 on both bases; expected-gain prior 6%; attribution scoping contract carried: intra-replay unattributability is the mechanism signature)
