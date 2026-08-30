# Verifier Context

> Durable compact context per `references/role-context-template.md`. Updated by Verifier only.

- role_contract_sha256: verifier.md loaded in-session rounds 000–003
- context_epoch: 4
- last_completed_round: `003` (report+verdict emitted; campaign auto-terminates on no-improvement #3 — Orchestrator owns the stop transition)
- accepted_kernel: `baseline_adapter.py` @`b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1` — FINAL DELIVERABLE (unchanged through r001–r003)
- accepted_report: `rounds/report_000.md` @`a90df70d54e791ecf53b38913ea1165e2a47a6dd6201d68653e6a101c5882e7c`
- recent_three_round_evidence: r001 no-improvement (aten-captured replay, fat boundary, +2.6 µs wall) → r002 no-improvement (direct Triton launch, launcher tax ≈+86–89 µs/call, wall −60.34%) → r003 no-improvement (lean direct-address replay: launcher neutralized + hit-rate 100% + device in-band, but build-intrinsic replay sync 69.02 µs/call absorbs the python prize → wash +0.2186%)
- open_hypotheses: none — campaign terminated; five-number decomposition complete
- artifact_read_hashes: see table

## Current Bottleneck (terminal physics — measured, census-grade)

- Wall ~0.147–0.151 ms/call = device 13.6–15.0 µs/call (single fused IXMMA kernel) + host ~134 µs/call (~38 cheap aten ops + sdpa C++ stack + fixed per-sample seed/sync floor).
- All three legal candidate host-families measured to root cause: (1) aten-captured replay: boundary ≥ prize (+2.6 µs, r001); (2) direct Triton launch: ~85 µs/call python launcher tax (−60.34%, r002); (3) lean direct-address graph replay: build-intrinsic replay-sync floor 69.02 µs/call absorbs the python prize (wash, r003) — the R-term is BUILD-INTRINSIC (branch (c) adjudicated), so any future graph family on this build starts ~69 µs/call in the hole.
- Device-side: proven-envelope Triton kernel (16.51 µs/call, byte-identical r002/r003) trails the vendor IXMMA kernel (13.61–15.0) — no device win inside the frozen capability envelope.

## Recent Three-round Evidence

- Round 001 (no-improvement): manual-cuda-graph-workspace-replay; retention/selectivity all pass; wall −1.6873% paired. Evidence: rounds/report_001.md, verdict_001.json @`c804df77…362`.
- Round 002 (no-improvement): triton-attention-dispatch-collapse; all correctness/envelope pass; wall −60.3422%; launcher tax quantified. Evidence: rounds/report_002.md, verdict_002.json @`a3b7f117…1985`.
- Round 003 (no-improvement #3, TERMINAL): graph-replayed-triton-direct-address; 5-way bitwise + harness-premise + stale-trap all pass; wash +0.2186%; R-term adjudicated build-intrinsic. Evidence: rounds/report_003.md, verdict_003.json @`e92f076c…ebcc`.

## Open Hypotheses or Checks

- None — campaign terminated. Terminal harness/build facts preserved for any future reopening: D2 arity (kernel-mode profiling cannot drive 3-input run_out); v0-format (adapter cannot serve as v0); trace-shape (gpu_user_annotation duplicate spans + zero attributed in-graph kernel events — census-substitution pattern); R-term (cudaGraphLaunch path intrinsically device-synchronizes ~69 µs/call on CoreX 4.4.0/torch 2.7.1/BI-V150); Triton python launcher overhead ~85 µs/call; D1 default-stream discipline.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| triton_flexattention_e2_003.py | 6ffb0c94bf6b126317acddcf14119bfd27fab5709c20a1f33cfdf8883d58bf1e | 003 |
| rounds/decision_003.md | d4f7203e9a032a40eb0164eeb515a8a0be31c9e5067e2a80036af4344affb203 | 003 |
| rounds/sketch_003.json | 4ef267b9bb67f8abc52889684412336785b4281612647f55efbacdc29f8dc6f0 | 003 |
| log/probes/binding_statement_report_003.json | f8be3a6b68f080e39f5a0b772b82f541fc590e37708df0aa8a2dfe04e956a7c1 | 003 |
| rounds/report_002.md | 2b93a9ed63b7d9b1e5b6a043fb202472f9afe647b60ea5b67c2333837c4a5ec8 | 003 |
| rounds/verdict_002.json | a3b7f117567fbd756356c9b10df58965665b8cd481513f47855da52db1c11985 | 003 |
| rounds/report_001.md | 8c93d473f6f3babcfd34c1cbe7bde76fbf1b1db1bbc002c61cbc04d76ab79336 | 003 |
| rounds/report_000.md | a90df70d54e791ecf53b38913ea1165e2a47a6dd6201d68653e6a101c5882e7c | 003 |
| baseline_adapter.py | b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1 | 003 |
| ../../base.py | dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0 | 003 |
| auto_bench.py | 71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29 | 003 |
| log/r003_forward_100iter.pt.trace.json | c8182c25c3f27af789fe6f8d187a35d7eaa95afc739a5aeb285904a85cd5c5f3 | 003 |
