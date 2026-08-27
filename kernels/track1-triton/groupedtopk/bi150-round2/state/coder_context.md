# Coder Context

- role_contract_sha256: `26c40a94bacbbe5ac4cf12b330516b0439a823e7ca8fd648bdace3fdfcce9cba`
- context_epoch: `4`
- last_completed_round: `003` (coding phase; result returned, canonical pointers unchanged pending Verifier)
- accepted_kernel: `triton_grouped_topk_r2_002.py` @`ad703266eb727f7725c8fa61ceaedcffc269e94291def703cb34279e5275ab12`
- accepted_report: `rounds/report_002.md` (r002 accepted: +28.67% prescribed basis, +18.22% same-session vs r001; 6.90 kernels/call; device 103.985 µs flat)
- recent_three_round_evidence: `report_000 identity v0=0.483530ms device_ratio 0.372 host-dominated; report_001 staged-Triton+kernel-side narrowing accepted; report_002 default-compile compression accepted, reduce-overhead named as unused lever -> taken by round 003`
- open_hypotheses: `H-003 compile-graph-replay-reduce-overhead implemented in candidate triton_grouped_topk_r2_003.py (sha256 62f8883a2c6d1bdf65d84b29beb71d95500b40b8d6acaf484eb09fccdcf97d38); expected 15%; awaiting Verifier measurement under fingerprint 8deb1b01...`
- artifact_read_hashes: decision_003=e214c29a…a403; sketch_003=4a909a11…a782; r002_source=ad703266…5ab12; report_002=bd0932b9…36ce; profile_snapshot/triton_cuda.yaml=dc8fa4c0…b7ae; capability_claim.json=2e6ee49d…7f67

## Current Bottleneck

- `Verifier-backed facts carried from historical epoch 1 and rounds 000-002: wall time is dominated by fixed host overhead; after r002's dispatch compression ~0.235 ms/call remains outside kernels while device stays ~104 µs/call (vendor top-k pair = 85.51 µs of it).`

## Recent Three-round Evidence

- `historical ../bi150/rounds/report_000..009 — epoch-1 accepted a reduce-overhead compile path (with per-invocation output cloning against pool overwrite); round-003 candidate instead eliminates pool ownership structurally via externally-owned output buffers`

## Round 003 Coding State

- classification written: `candidate-ready` → `rounds/coder_result_003.md` (no major deviation; one disclosed design repair caught by own gate)
- candidate = mode-only escalation to graph-replay tier with THREE-TIER permanent chain replayed→compiled-default→staged; monotonic downward flags; strict [83,256]+fixed-config gating constructs NO compiler off-regime
- machine proofs under log/probes/ (no timing/profiler usage beyond bounded sanity):
  - coder_smoke_result.json @e6414ad0364a0e701c1000a273f4dc132bc3fe3362bce2fa014f47907095a366 — 18/18 PASS: cold-capture smoke (0.562 s sanity) + warm-replay repeat bit-stable (0.001 s sanity); bitwise-vs-r002 sweep on seed42 + four tie suites + new-bytes independence ALL through the REPLAYED route (tier active at sweep end); both fallback edges exercised permanently-once each with poison invoked exactly once; non-target regime never constructs any compiler then re-enters replay tier; run_out poisoned-buffer BOTH orderings with pointer preservation bitwise==r002; cross-instance alternation clean
  - binding_statement_report.json @b32eb677d43b7d2ad51cb4ec140aae4661495a1ce027098c2ff77301adafe1c7 — SEVEN frozen segments byte+AST identical to r002 incl. run_out; exactly two torch.compile sites {graph-replay mode + dynamic False} / {default mode + dynamic False}; whole-file quoted-mode counts 1/1 and mode=/dynamic=False counts 2/2; carryover forbidden-token scan all-zero
- hazard resolution record: first-write design returned pool-backed replay outputs; warm-replay consumer read raised framework overwrite protection (hazard ii concrete) → repaired by routing forward through fresh externally-owned output buffers (run_out unchanged); net-zero kernel delta vs r002; framework then logs "skipping cudagraphs due to mutated inputs" on buffer paths — graceful, correctness unconditional, recorded for Verifier attribution scoping (branch A or B both covered by decision rule)
- open local checks: none outstanding; same-round repair budget untouched (0 verifier-requested repairs)

## Open Hypotheses or Checks

- H-003 awaiting authoritative Verifier wall comparison (expected ≥5% unrounded paired median improvement; expected-gain prior 15%; two-branch kernel-count pass rule; capture-cost absorbed outside timed medians)
