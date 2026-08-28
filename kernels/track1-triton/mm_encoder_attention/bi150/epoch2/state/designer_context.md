# Designer Context

- role_contract_sha256: `7227706c7068ad4a20caebb95c045721f643a409473fc9768e73d828fb2e5ab5`
- context_epoch: `1`
- last_completed_round: `002 (no-improvement, reading partial-band; streak 2/3; round budget 2/20; round 003 in flight — THE FINAL BULLET: miss #3 auto-terminates; accepted resets; rejected does NOT consume streak)`
- accepted_kernel: `baseline_adapter.py` @ `c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f` (canonical UNCHANGED — r001/r002 not accepted)
- accepted_report: `rounds/report_000.md` @ `20b21646d9c3ba3abe086d8133799d23a39981dcb4e1cb547e1a3f65b0bf7ffc`
- recent_three_round_evidence: `report_001 @13adafe9: wall −65.7458%, T_launcher +84.7651, D_cand(nw1) 28.2030. report_002 @bb46dee7 (canonical): wall −59.8032% (0.231689 vs 0.144984, inside declared band); AUTHORITATIVE D_cand(nw2) = 19.5550 µs/call attributed (probe-method 15.317, replay-regime bias +4.2–4.7); T_launcher +84.5712 (invariance PASS); device cut −30.66% attributed, wall −9.26 µs ≈ 1:1; outputs bitwise-equal to r001; host census fully unchanged; capability matrix CLOSED: fp16-operand dots compile-but-fail exactness (max_abs 1459, vendor-saturation signature) at every warp count = capability-NEGATIVE; nw4 no-gain over nw2; deliverable banked triton_mm_encoder_attention_e2_002.py @cc98318b at 0.6258x. F2 projection with canonical numbers: net = +3.10 µs/call WORSE (sub-parity; composed class 0.94–0.96x conservative / 0.94–1.01x honest band).`
- open_hypotheses: `H-003 (round 003 in flight, decision_003 dispatched — the terminal move): F2 graph-replayed-triton-direct-address composition of the byte-identical r002 kernel (three-tier: direct-address replay → copy-in replay → eager; lean boundary guard + 1 graph launch + 1 copy-out; bounded recapture ≤4). Expected verdict no-improvement #3 → auto-termination, spent DELIBERATELY per the DELIVERABLE RULE to bank the composed 0.94–1.01x-class submission (vs 0.6258x direct) + close the graph-family physics (R-term at bsz=2, boundary terms, kernel-in-graph regime 19.555-vs-15.317 adjudication). Win branch (≥+5%) needs kernel-in-graph ≤ 9.2 µs — 10.36 below floor — declared unreachable; expected_wall_improvement_pct 0.0.`
- artifact_read_hashes: `40 artifacts hashed through round-003 dispatch; ledger table below`

## Current Bottleneck

- Verifier-backed (report_001 + report_002, canonical): host-bound and now fully mapped — T_launcher = +84.77/+84.57 µs/call net (invariant; the Triton python launcher costs more than the whole 33-op aten stack); device floor D_cand = 19.5550 µs/call attributed (nw2; vendor Ixmma 17.42/15.36). The ≥5% adoption bar is arithmetically closed for every measured family (graph win needs ≤9.2 µs kernel-in-graph). The remaining value is DELIVERABLE-side: the composed submission class 0.94–1.01x vs banked 0.6258x.

## Campaign Physics Map (complete except the composed round's terms)

| Line | Value (canonical) | Source |
|---|---|---|
| wall_base (session basis) | 0.1450–0.1501 ms | r000/r001/r002 paired v0 |
| Base device (vendor Ixmma, Causal=0 bidirectional) | 17.39–17.42 whole-trace / 15.36–15.69 attributed µs/call | r001/r002 |
| Base host floor | ~133 µs/call, 33 aten ops | r000 census |
| T_launcher (net, bsz=2) | +84.7651 / +84.5712 µs/call | r001/r002 (invariance PASS) |
| D_cand direct kernel | 28.203 (nw1) → 19.555 (nw2) attributed; 23.49→15.32 probe-method (+4.2–4.7 replay-regime bias) | r001/r002 |
| Capability matrix | fp16 dots: compile-YES / exactness-NO (1459 vendor-class) = NEGATIVE; nw4 = no-gain; nw2 = shipped, bitwise-equal outputs | r002 p13 sweep |
| Graph family | R-term 69.02 µs (sibling bsz=1, transfer pending r003 measurement); boundary ~13 µs; projection net = D_kernel_in_graph − 16.455 µs | report_001/002 + sibling |
| Deliverable trajectory | 0.6033x (r001) → 0.6258x (r002) → composed 0.94–1.01x class (r003, in flight) | reports + r003 projection |

## Ranked Plausible Families (final state)

1. **F2 — graph-replayed-triton-direct-address: DISPATCHED as rounds/decision_003.md @ `0a678da8…` (+ sketch_003.json @ `bdf42355…`, schema-v2 GREEN).** The terminal move (sibling decision_003 architecture at bsz=2): byte-identical r002 kernel captured once per pointer-set, three-tier chain, lean boundary; honest 0.0 expectation; products = composed deliverable + R-term-at-bsz2 + regime adjudication. Pre-declared readings (a)–(f) cover win/capture-failure/harness-falsification/correctness channels.
2. **Deliverable hardening (post-r003, only if the round is REJECTED and a replacement is dispatched):** re-attempt composition or close out with the best banked deliverable; abort form otherwise.
3. **Closed by measurement:** direct-family wall rounds (T_launcher arithmetic); fp16-dot anything (capability-NEGATIVE); num_warps tuning (nw2 optimal, nw4 no-gain); in-envelope restructuring (no-headroom); math-backend split / compile machinery / harness manipulation (contract anti-patterns).

## Recent Three-round Evidence

- report_000: baseline 0.150149 ms; ONE fused vendor kernel 16.54–17.56 µs/call; device_ratio 0.110; 33 aten ops.
- report_001 (no-improvement #1): wall −65.7458% (0.240953), T_launcher +84.7651, D_cand 28.2030; deliverable banked @4171de8d; reading (b) exactly as pre-declared.
- report_002 (no-improvement #2): wall −59.8032% (0.231689), D_cand(nw2) 19.5550 AUTHORITATIVE (−30.66% cut, ~1:1 to wall), T_launcher invariance PASS, host census unchanged, capability matrix closed (fp16 NEGATIVE, nw4 no-gain); deliverable → @cc98318b 0.6258x; F2 projection +3.10 µs worse (sub-parity 0.94–0.96x class).
- (round 003 in flight: decision_003 + sketch_003 dispatched, validation GREEN.)

## Open Hypotheses or Checks

- H-003 (r003, final bullet): composed band 0.94–1.01x; expected reading (b) — no-improvement #3 with the composed deliverable banked (auto-termination, the sibling-style clean ending). Named products: rterm_transfer_at_bsz2, kernel-in-graph regime adjudication (19.555 attributed vs 15.317 graph-assisted), boundary census.
- Post-r003: if ACCEPTED (≥+5%, unexpected) the campaign continues (streak reset); if REJECTED (capture failure), a replacement round is possible without streak cost — retry composition or close out; if no-improvement #3, the campaign auto-terminates with the deliverable ledger settled (composed candidate if correctness-PASS, else the r002 direct @cc98318b).
- Standing observations carried for any successor campaign: D1 kernel-mode run_out arity; D2 kineto dual-span census-substitution; vendor fp16-saturation on extreme inputs (candidate 3.05e-05 vs vendor 1457 — deliverable narrative); verdict-file internal-hash citation convention (normalized-vs-file, non-blocking, noted by Orchestrator at r002).

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| skills/kernel-opt-loop/prompts/designer.md | `7227706c7068ad4a20caebb95c045721f643a409473fc9768e73d828fb2e5ab5` | 002 |
| skills/kernel-opt-loop/adapters/claude-code.md | `31a161224900c8e7af2c3b9175adbf64165c2f56199790aee264a0cd3d8fb597` | 000 |
| project.md | `023f8ae2ca8ccdc1f5770bd59f541e4c9635c013b2b4524206be9b9048577d67` | 000 |
| ../../base.py | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | 000 |
| profile_snapshot/triton_cuda.yaml | `dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae` | 002 |
| profile_snapshot/capability_claim.json | `aeba3a87f0494c2bb349b92fe668370c70d77fdebea29eac52824c3556b0d4d8` | 002 |
| baseline_adapter.py | `c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f` | 001 |
| team-state.md (read-only) | `5e22c0e259c43017d5ce3b3294fb7cf63417d6dd126fff558b1c00e2f50dc860` | 003 |
| auto_bench.py (harness) | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 000 |
| triton_mm_encoder_attention_e2_001.py (deliverable r001) | `4171de8dcb1df6478942b0861756d660ef7955c5741e60072f03476a5fab2fc2` | 002 |
| triton_mm_encoder_attention_e2_002.py (deliverable r002, nw2) | `cc98318bbfed0d9cdbf7a99769c4a44077dc15590496a4e8410ba9eda99f3078` | 003 |
| rounds/report_000.md | `20b21646d9c3ba3abe086d8133799d23a39981dcb4e1cb547e1a3f65b0bf7ffc` | 001 |
| rounds/report_001.md | `13adafe951df94bb7bb74294e195cfffc6992057d36e958801b293ab292f449c` | 002 |
| rounds/report_002.md | `bb46dee71b12e8fb5289fbe3a7419e18cbd26e8f4bee5de3dc01b84f6354e1d5` | 003 |
| rounds/verdict_001.json | `b6e62fdbf3757370449ca016742d202a2da6cd62f51f55de90d2dd9f23746822` | 002 |
| rounds/verdict_002.json | `a86c2e8cd059b1a71439664c296eb5b660b56e8d8b4aa1bea93927480190d9d9` | 003 |
| rounds/decision_001.md (authored) | `67b96739c35adabb713081a1f3a50649193b28eed420dc32dd512572fab26c78` | 001 |
| rounds/sketch_001.json (authored) | `a1c27dbae53b1c7a74681510a0d09ced6be58ed8501f86976ce55af1b4772363` | 001 |
| rounds/decision_002.md (authored) | `20b360ac936bf4d9d41afadac90c40578f0a758e628ec40af2d3c759eb22d3fb` | 002 |
| rounds/sketch_002.json (authored) | `c16b1528b25ae1a3bbfc72b3e459462505d940677e62b30a0585e3b41b46e9e9` | 002 |
| rounds/decision_003.md (authored r003) | `0a678da87a877b9c521b6c280eb3518b20f98e352786e9df129435e2cc918413` | 003 |
| rounds/sketch_003.json (authored r003) | `bdf423556e7c80369ae38d4980529a739a52a3d18033e572927354b23e0a4e64` | 003 |
| ../final_summary.md (epoch-1, noncanon) | `49ee0709c332a562d2b26a9e68fb607bbbb5259c4cdc35c2cfebf6986616ad55` | 000 |
| ../triton_mm_encoder_attention_001.py (epoch-1, noncanon) | `88ade697da35a51362c2a8643e054a61362a68ff3e9e2e60110bd3e45285e87e` | 000 |
| ../rounds/report_000.md (epoch-1, noncanon) | `138076deaeb430d9045d5648b15ac8e0a9a962e0eca5185d02ab2a7df5fd96da` | 000 |
| ../rounds/coder_result_deliverable.md (epoch-1, noncanon) | `acb03ff3c051e1ba9bc1cb06675745dc3d9aca26c04a9fe75d6f466e7e4e793a` | 000 |
| SIBLING final_summary.md | `5046a291230561d7d473e55923eeece4870f25554050ce289a8577fd7c8028f1` | 000 |
| SIBLING rounds/report_000.md | `a90df70d54e791ecf53b38913ea1165e2a47a6dd6201d68653e6a101c5882e7c` | 000 |
| SIBLING rounds/report_001.md | `8c93d473f6f3babcfd34c1cbe7bde76fbf1b1db1bbc002c61cbc04d76ab79336` | 000 |
| SIBLING rounds/report_002.md | `2b93a9ed63b7d9b1e5b6a043fb202472f9afe647b60ea5b67c2333837c4a5ec8` | 000 |
| SIBLING rounds/report_003.md | `9774ad2bfabe624ba71233c5d841a1c330d8a03926f2797a6e4f9f06cbe465f4` | 000 |
| SIBLING rounds/decision_002.md | `459e8d37219b5534103a82a7a342c61ef04e147158a6851d794b73e2a44f8730` | 001 |
| SIBLING rounds/sketch_002.json | `fb5bec0b957a04ffa19d20edb2f0fdb92de156c0aea6429b1c796a86b89bd87c` | 001 |
| SIBLING rounds/decision_003.md | `(read in full rounds 002-003; hash not taken)` | 003 |
| SIBLING rounds/sketch_003.json | `4ef267b9bb67f8abc52889684412336785b4281612647f55efbacdc29f8dc6f0` | 003 |
| references/invariants.md | `2349247c5653db35ab5af5b22267f6ab813fad1f24076c88d6bf80207cbd8cb7` | 000 |
| references/anti-patterns.md | `aebcdee623024594ad6a19905d626dd7c7ba099d68eba203315229608a40d0c4` | 000 |
| references/bottleneck-judgment.md | `664d1e622333559a08419bb39b0b19b04054507a8adb58e3e347ab308c69eae7` | 000 |
| references/role-context-template.md | `d3eead2d8480975a9a954b104d21c6d9d57e1713edf0fc8096184c09aafe56d2` | 000 |
| references/decision-template.md | `a081503562fa30751f8df63ba3553e1766b9707d9af663810d800f829409ffa0` | 001 |
| scripts/validate_decision.py | `c2882fd2d0875fdf64f8b0608be56d6269409e835e1ccf61e510999c984d5099` | 001 |
| scripts/validate_sketch.py | `00613adf561f44de63f90671ebddc7e571fc757bf572470610cf423ecbd6eba8` | 001 |

Sibling artifacts live under `kernels/track1-triton/flexattention/bi150/epoch2/`; epoch-1 archive `../` is relative to this project root. Authored artifacts (decision/sketch 001–003) are Designer-owned outputs.
