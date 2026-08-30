# Designer Context

- role_contract_sha256: `7227706c7068ad4a20caebb95c045721f643a409473fc9768e73d828fb2e5ab5`
- context_epoch: `4` *(canonical root `flexattention/bi150/epoch2`; updated at Round 003 after decision dispatch)*
- last_completed_round: `002`
- accepted_kernel: `baseline_adapter.py` @`b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1` (UNCHANGED — r001/r002 no-improvements did not advance pointers)
- accepted_report: `rounds/report_000.md` @`a90df70d54e791ecf53b38913ea1165e2a47a6dd6201d68653e6a101c5882e7c`
- recent_three_round_evidence: `r000 baseline (0.151107 ms v0; Ixmma 13.56 µs/call; ratio 0.0897). r001 NO-IMPROVEMENT −1.6873% (replay wrapper ENGAGED, aten 34→6, but fat boundary > prize on 1-launch base; derived per-aten-op price 0.6-1.0 µs; observed per-call sync/driverGet in replay route, cause unattributed). r002 NO-IMPROVEMENT −60.34% (dispatch collapse 38→1 FULLY engaged; kernel DEVICE-HEALTHY 16.51 vs 13.61 µs/call; ~85 µs/call pure Triton python-launcher overhead = entire failure). miss 2/3 — ONE more no-improvement ANYWHERE terminates the campaign.`
- open_hypotheses: `H-A falsified (r001); H-B pre-falsified; H-C dead (views); H-D-v2 falsified-on-wall but kernel proven healthy (r002); r003 DISPATCHED: composition family graph-replayed-triton-direct-address (r001 machinery × r002 kernel, lean direct-address boundary). This is the LAST live mechanism; terminal close-out fully quantified either way.`
- artifact_read_hashes: `ledger table below, refreshed at Round 003`

## Current Bottleneck

- **TWO-TERM HOST PICTURE (canonical after r002)**: base = 1 launch + ~34 aten dispatches (~20-34 µs) + harness-fixed
  seed/sync inside timed window + 13.25-13.61 µs device (Ixmma). r002 removed ALL aten dispatch and kept submissions
  at 1 — yet wall regressed because the Triton python launcher costs ~85 µs/call on this build. r001 removed python
  via graph replay but paid a fat boundary (5 submissions + copy-ins + sync-class) exceeding the prize. THE REMAINING
  MECHANISM: replay the healthy kernel through a LEAN direct-address boundary (guard + 1 submission + copy-out) —
  launcher never runs, boundary fat structurally absent. Device delta +3.3..5.3 µs is known and priced; the one
  unattributed swing term is replay-intrinsic sync cost (r001 observation).

## Recent Three-round Evidence

- **r000 / baseline / n-a**: report_000 @a90df70d… gate PASS; kernel-mode blocked pre-run_out; fingerprint
  cross-validated.
- **r001 / NO-IMPROVEMENT / manual-cuda-graph-workspace-replay**: report_001 @8c93d473… — mechanism ENGAGED (aten
  34→6/call, 1 cudaGraphLaunch + 4 memcpy + 5 submissions, bitwise retention 150/150, selectivity/recovery PASS,
  branch-A attribution 0.14 kernels/call); wall FAIL; per-call cudaDeviceSynchronize + cudaDriverGetVersion observed
  in replay route (build vs design cause UNRESOLVED — r003 census pre-declares its adjudication); D2: harness
  kernel-mode arity incompatible with 3-input ops (forward-mode fallback canonical).
- **r002 / NO-IMPROVEMENT / triton-attention-dispatch-collapse**: report_002 + verdict_002 — dispatch collapse 38→1
  engaged exactly as designed; `_causal_attn_fwd` 16.51 µs/call (+2.9 vs Ixmma); wall −60.34% (+92.5 µs/call
  drift-corrected, ~82-86 µs = launcher); correctness incl. fp16-extreme all green; capability legality held
  (proven-envelope dots, num_warps=1).
- NONCANONICAL priors (labeled): sibling groupedtopk-e2 (+42.54% via replay of a MULTI-launch region; mutation-skip;
  foreach lesson); epoch-1 archive naive 0.61x (scalar-era device bound).

## Open Hypotheses or Checks

- **r003 DISPATCHED (final live mechanism)** — family `graph-replayed-triton-direct-address`, scope mixed:
  tier-1 direct-address manual-graph replay of the r002 kernel (captured against the CALLER'S OWN q/k/v pointers —
  source-verified address stability: `auto_bench.time_forward` lines 459-475 reuse one inputs list across all 150
  calls; correctness phase clones per call ⇒ bounded recapture ≤4 lifetime on first-seen pointer sets, one during
  warmup), static out_ws workspace, per-call = 3×data_ptr guard + ONE replay submission + one copy-out into
  invocation-owned/caller buffer; tier-2 copy-in replay (r001-proven) for pointer mismatches; tier-3 eager direct
  launch (r002 path) for non-target regimes/failures. Zero model-code sync; zero compile machinery; capability
  legality unchanged (proven-envelope dots, num_warps=1; P5 blocked). Artifacts:
  `rounds/sketch_003.json`@4ef267b9…, `rounds/decision_003.md`@d4f7203e… — BOTH validator-green (RC=0, valid:true).
  H-003 expected 8.0%; priced identity: launcher(−85) + lean boundary vs r001 fat; swing term R = replay-intrinsic
  sync (0 or 10-20 µs) pre-declared; same-session paired basis authoritative (bar ≈ ref_median − 7.556 µs).
- **Pre-declared failure readings** (Evaluation Contract): (a) win with tier-1 hit-rate 100; (b) boundary floor >
  prize ⇒ no-improvement #3 ⇒ campaign TERMINATES with decomposition complete; (c) build-intrinsic replay sync named;
  (d) hit-rate 0 ⇒ harness-premise falsified; (e) Triton-capture failure ⇒ tier-2-only wall ≈ r001+3.3 µs; (f)
  correctness/bitwise deviation ⇒ candidate-failed channel.
- Abort REJECTED again: a falsifiable ≥5% hypothesis exists (both censuses + source-verified premise); contract
  forbids abort; worst case converges to the same measured terminal state.
- Binding DANGER tokens for Coder ledger: compile strings 0; tl.dot envelope audit; num_warps=1; NO model-code
  sync/query calls; returning graph-resident/workspace tensors FAILS; unbounded recapture FAILS.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| skills/kernel-opt-loop/prompts/designer.md | 7227706c7068ad4a20caebb95c045721f643a409473fc9768e73d828fb2e5ab5 | 001 |
| skills/kernel-opt-loop/adapters/claude-code.md | 31a161224900c8e7af2c3b9175adbf64165c2f56199790aee264a0cd3d8fb597 | 000 |
| references/invariants.md | 2349247c5653db35ab5af5b22267f6ab813fad1f24076c88d6bf80207cbd8cb7 | 000 |
| references/anti-patterns.md | aebcdee623024594ad6a19905d626dd7c7ba099d68eba203315229608a40d0c4 | 003 |
| references/bottleneck-judgment.md | 664d1e622333559a08419bb39b0b19b04054507a8adb58e3e347ab308c69eae7 | 000 |
| references/decision-template.md | *(read at r001; schema-v2 governance)* | 001 |
| auto_bench.py | 71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29 (timing-loop lines 449-513 source-read at r003: time_forward reuses one inputs list; run_forward clones per call; set_seed+sync_devices inside timed window) | 003 |
| flexattention/base.py | dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0 | 000 |
| bi150/epoch2/project.md | 82b9ad3029ef3835760ae5e85643bfff1dbaf47c6136b181db7673e955ba15c6 | 000 |
| bi150/epoch2/team-state.md | ad532051b974aa2236bfc7d72a350529bbdcb86a3e86023bae2c84f7ca84357e | 001 |
| bi150/epoch2/baseline_adapter.py | b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1 (= accepted_kernel) | 001 |
| bi150/epoch2/profile_snapshot/triton_cuda.yaml | dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae | 000 |
| bi150/epoch2/profile_snapshot/capability_claim.json | 07aa5d489acb9c21717032087812d264dd5170fe79e7ea2326edb04cab657c1d | 000 |
| bi150/epoch2/rounds/report_000.md | a90df70d54e791ecf53b38913ea1165e2a47a6dd6201d68653e6a101c5882e7c (= accepted_report) | 001 |
| bi150/epoch2/rounds/decision_001.md | fa11b1152306e4cc4b33a02e31bc52d4c76de210c79385f41e02ee25c3bc7b1d | 001 |
| bi150/epoch2/rounds/sketch_001.json | 199275b85e831238c2f0c9c694d3c4c03550c6681bd7a8e87f3474642b3c1fce | 001 |
| bi150/epoch2/rounds/report_001.md | 8c93d473f6f3babcfd34c1cbe7bde76fbf1b1db1bbc002c61cbc04d76ab79336 | 002 |
| bi150/epoch2/rounds/decision_002.md | 459e8d37219b5534103a82a7a342c61ef04e147158a6851d794b73e2a44f8730 | 002 |
| bi150/epoch2/rounds/sketch_002.json | fb5bec0b957a04ffa19d20edb2f0fdb92de156c0aea6429b1c796a86b89bd87c | 002 |
| bi150/epoch2/rounds/report_002.md + verdict_002 | *(read via Orchestrator census at r003 dispatch; hash re-verify owned by Verifier ledger)* | 003 |
| bi150/epoch2/rounds/sketch_003.json | 4ef267b9bb67f8abc52889684412336785b4281612647f55efbacdc29f8dc6f0 | 003 |
| bi150/epoch2/rounds/decision_003.md | d4f7203e9a032a40eb0164eeb515a8a0be31c9e5067e2a80036af4344affb203 | 003 |
| track1-triton/groupedtopk/bi150-round2/final_summary.md *(sibling op)* | 7278f1f8d09cda4e22b2ee24e0eade6611c13b047b65f1f81866a5ab70829c4e | 000 |
| track1-triton/groupedtopk/bi150-round2/rounds/report_004.md *(sibling op)* | c79cc018f9c61ec34f084fc589b06b61d9b8e9ba634710d2ba365e3d1c34fe35 | 000 |
| track1-triton/summary_all_backends.md (§四 BI150) | f899c82a88118a22f06e2231ede4cce4545a740e283f118bbde327b0104bd2e4 | 000 |
| bi150/rounds/report_000.md + final_summary.md *(STALE/NONCANONICAL, parent archive)* | 98c648e682ce0034565fcb0691402b2f3593b5518a3902bc8c94fd66d135eea6 / *(via prior ledger)* | 002 |

Harness facts (r003 source-read): `time_forward` reuses ONE inputs list across warmup+repeat (addresses stable per
invocation — tier-1 premise); `run_forward` clones per call (correctness-phase pointer variance — drives bounded
recapture design); `set_seed` + `sync_devices` INSIDE the timed window (both sides pay; fixed); kernel-mode arity
limitation (D2) stands for 3-input ops.
