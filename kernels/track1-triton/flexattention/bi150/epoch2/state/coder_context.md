# Coder Context

- role_contract_sha256: `26c40a94bacbbe5ac4cf12b330516b0439a823e7ca8fd648bdace3fdfcce9cba`
- context_epoch: `3`
- last_completed_round: `003` (coder work complete; classification candidate-ready; canonical pointers untouched — Orchestrator-owned)
- accepted_kernel: `null` (baseline_adapter.py @b8ec3458… remains the canonical comparison target until Verifier/Orchestrator adopt)
- accepted_report: `rounds/report_000.md` @`a90df70d…`
- recent_three_round_evidence: `001 candidate accepted earlier today; 002 candidate-ready (dispatch collapse, launcher price measured); 003 coder: three-tier graph-replayed-triton-direct-address candidate triton_flexattention_e2_003.py @6ffb0c94…bf1e — r002 kernel BYTE-IDENTICAL, tier-1 direct-address replay (zero copy-ins) + tier-2 copy-in + tier-3 eager, bounded recapture (initial free + ≤4), all 7 probes PASS, harness smoke 1.047x at warmup5/repeat10 (observation only).`
- open_hypotheses: `H-003 expected 8% — awaits Verifier authoritative paired medians; smoke positive; R-term (replay-intrinsic sync) python-invisible, reserved for census.`
- artifact_read_hashes: `decision_003 d4f7203e…; sketch_003 4ef267b9…; r002 kernel source 570bc2be…; candidate 6ffb0c94…; binding statement f8be3a6b…`

## Current Bottleneck

- `Host-bound per report_000 (Verifier-backed). Round-003 composes the two measured prices: r002's ~85us/call Triton python launcher (neutralized by replay) and r001's boundary fat (reduced to guard+replay+copy-out, 2 submissions, 0 copy-ins on the stable timed regime). The remaining unknown is the R swing term (build-intrinsic replay sync) — census decides.`

## Recent Three-round Evidence

- `003, candidate-ready (coder), family graph-replayed-triton-direct-address — evidence: rounds/coder_result_003.md + log/probes/p13..p19 logs + binding_statement_report_003.json + p19_r003_sweep_result.json; no verdict yet.`

## Open Hypotheses or Checks

- `Verifier: paired medians warmup50/repeat100 default stream; tier-1 hit-rate expectation 100% timed / exactly one warmup recapture (p13); census for submissions (2/call) + R-term attribution. D1 budget counting and D2 bound_sets history flagged for Orchestrator adjudication.`

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| rounds/decision_003.md | d4f7203e9a032a40eb0164eeb515a8a0be31c9e5067e2a80036af4344affb203 | 003 |
| rounds/sketch_003.json | 4ef267b9bb67f8abc52889684412336785b4281612647f55efbacdc29f8dc6f0 | 003 |
| triton_flexattention_e2_002.py | 570bc2be2cb8e79a06ebb32e5e8bf4f79aa62a38d5382b9a1a5f12426f3512b1 | 003 |
| triton_flexattention_e2_003.py | 6ffb0c94bf6b126317acddcf14119bfd27fab5709c20a1f33cfdf8883d58bf1e | 003 |
| log/probes/binding_statement_report_003.json | f8be3a6b68f080e39f5a0b772b82f541fc590e37708df0aa8a2dfe04e956a7c1 | 003 |
