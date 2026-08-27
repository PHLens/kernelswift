# Coder Context

- role_contract_sha256: `26c40a94bacbbe5ac4cf12b330516b0439a823e7ca8fd648bdace3fdfcce9cba`
- context_epoch: `5`
- last_completed_round: `004` (coding phase; result returned, canonical pointers unchanged pending Verifier)
- accepted_kernel: `triton_grouped_topk_r2_002.py` @`ad703266eb727f7725c8fa61ceaedcffc269e94291def703cb34279e5275ab12`
- accepted_report: `rounds/report_002.md` (r002 accepted +28.67% prescribed / +18.22% same-session; r003 RETIRED as no-improvement evidence — pointers unchanged)
- recent_three_round_evidence: `report_000 identity v0=0.483530ms host-dominated; report_001 staged-Triton accepted; report_002 default-compile accepted (reduce-overhead lever named unused); r003 inductor replay refused by framework → family closed; round 004 takes the remaining legal mechanism`
- open_hypotheses: `H-004 manual-cuda-graph-workspace-replay implemented in candidate triton_grouped_topk_r2_004.py (sha256 c02d956c6bb5c27c229623b01b99b85f5962db79b5ead09df6fbca7a52e721eb); expected 15%; awaiting Verifier measurement under fingerprint 8deb1b01...`
- artifact_read_hashes: decision_004=e5465d7d…3be1; sketch_004=ccf277f4…e59; r002_source=ad703266…5ab12; profile_snapshot/triton_cuda.yaml=dc8fa4c0…b7ae

## Current Bottleneck

- `Verifier-backed facts carried through rounds 000-002: wall dominated by fixed host overhead; after r002 ~0.235 ms/call remains outside kernels at 6.90 kernels/call; manual single-submission replay attacks the per-launch submission residue directly.`

## Recent Three-round Evidence

- `historical ../bi150/rounds/report_000..009 epoch-1 prior; r1-002 accepted; r1-003 retired no-improvement (inductor refuses mutated inputs) — manual capture NOT subject to that heuristic, fired and served correctly this round`

## Round 004 Coding State

- classification written: `candidate-ready` → `rounds/coder_result_004.md` (no major deviation; probe-side tooling repairs disclosed)
- candidate = ONE manual torch.cuda.CUDAGraph over instance-owned static workspace (copy-in at boundary per call, full-overwrite placeholders, copy-out OUTSIDE boundary every call), three-tier chain manual-replay→compiled-default(lazy)→staged, both upper tiers permanent-on-failure with lazy construction inside the failing call
- machine proofs under log/probes/ (no timing/profiler usage beyond bounded sanity):
  - coder_smoke_result.json @54d14d896e7de0e6e7c6357a7d92ad724f91800324523aab9bc1db4b886e638f — 21/21 PASS: cold-capture smoke (0.139 s sanity) + warm-replay bit-stable; CAPTURE-FIRED proof via handle-alive + lower-tiers-absent + stale-trap + detectable-separation quadruple; bitwise-vs-r002 sweep seed42+4 tie suites+new-bytes ALL through tier 1 (active at sweep end, compiled-default NEVER constructed); Edge A true construction-failure → lazy default permanent-once; Edge A2 live-handle replay() poison hit once → permanent; Edge B stacked cascade 1/1 poisons; T=41 zero artifacts then graph build recovery; run_out poisoned BOTH orderings in-place bitwise==r002 + fresh-copyout leak-trap; cross-instance alternation clean
  - binding_statement_report.json @1e6b44a5d6db200d91a7686dea39069046e7e184c38de83eb54444a693ddf9bc — six frozen segments byte+AST identical (+forward verbatim); run_out AST-identical w/ disclosed 3-line comment delta; DANGER rule 'reduce-overhead' total==0; exactly one torch.compile site {default,dynamic False}; carryover allowlist all-zero incl. lowercase cudagraph prose; workspace/replay contract spans machine-checked
- hazard-resolution record: two gate FAILs were my probe-setup artifacts (unrealistic injected handles), candidate never misbehaved; final bytes frozen before gates ran green
- open local checks: none outstanding; same-round repair budget untouched (0 verifier-requested)

## Open Hypotheses or Checks

- H-004 awaiting authoritative Verifier wall comparison (expected ≥5% unrounded paired median improvement; expected-gain prior 15%; TWO-BRANCH kernel-count rule with branch-A collapse toward ≤2 attributable launches/call expected from copy-in+replay+copy-out structure; cold warmup+capture cost absorbed outside timed medians)
