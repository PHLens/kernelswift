# Coder Context

- role_contract_sha256: `26c40a94bacbbe5ac4cf12b330516b0439a823e7ca8fd648bdace3fdfcce9cba`
- context_epoch: `3`
- last_completed_round: `002` (coding phase; result returned, canonical pointers unchanged pending Verifier)
- accepted_kernel: `triton_grouped_topk_r2_001.py` @`4ae64cad913267f2198fec735e08f1b9490cafa1139d3a48ee11400aacb80de3` (round-001 candidate-ready, accepted per report_001)
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `report_000: wall v0=0.483530ms device_ratio 0.372 host-dominated, 14.94 kernels/call; report_001: staged-Triton pipeline + kernel-side int64->int32 narrowing, H-001 measured by Verifier (see report_001 for verdict)`
- open_hypotheses: `H-002 compile-graph-default implemented in candidate triton_grouped_topk_r2_002.py (sha256 ad703266eb727f7725c8fa61ceaedcffc269e94291def703cb34279e5275ab12); expected 10%; awaiting Verifier measurement under fingerprint 8deb1b01...`
- artifact_read_hashes: decision_002=31c972fb…ff37; sketch_002=0ccbec47…4cf3; decision_001=93783baa…3532b; sketch_001=637917e0…f6985; baseline_adapter=ecce4dac…39fa5; r001_source=4ae64cad…80de3; profile_snapshot/triton_cuda.yaml=dc8fa4c0…b7ae

## Current Bottleneck

- `Verifier-backed facts carried from historical epoch 1: wall time on bi150 small-shape operators is dominated by fixed host overhead (~66 µs/device sync pair); round-001 removed the preprocessing chain but host launch/session floor remains the bottleneck.`

## Recent Three-round Evidence

- `historical ../bi150/rounds/report_000..009 — final accepted torch.compile(reduce-overhead) host-launch path; r1-002 candidate reproduces that lineage under mode='default'/dynamic=False only`

## Round 002 Coding State

- classification written: `candidate-ready` → `rounds/coder_result_002.md` (no major deviation; conformance notes only)
- candidate = byte-faithful wrap of accepted staged pipeline behind ONE shared torch.compile(mode='default', dynamic=False) callable; strict target-regime guard [83,256]+fixed config; PERMANENT at-most-once fallback to unmodified staged execution on construction OR invocation failure (non-transient per plan)
- machine proofs (log/probes/, no timing/profiler usage):
  - coder_smoke_result.json @031db1123cc563b91d8ff02bb9dbaa569601a0986f7a4eb21fc6e0f7288ecec5 — 15/15 PASS incl. cold(3.544s-sanity)/warm compile smoke, tie suites through compiled route bitwise-equal to accepted r001, non-target-regime selectivity (compiler never constructed), forced dynamo-failure permanence (poison called exactly 1x), run_out poisoned-buffer equality cold+warm with pointer preservation
  - binding_statement_report.json @9315ba1b5f6b431713e7699f6ba89515d292e9bba56edd9d5cd4e18f5093a6b6 — six inherited segments BYTE+AST identical to r001; exactly one torch.compile site {mode:'default',dynamic=False}; forbidden-token scan all-zero; shared helper routed from both forward and run_out
- open local checks: none outstanding; same-round repair budget untouched (0 of 1 used)

## Open Hypotheses or Checks

- H-002 awaiting authoritative Verifier wall comparison (expected ≥5% unrounded paired median improvement; expected-gain prior 10%)
