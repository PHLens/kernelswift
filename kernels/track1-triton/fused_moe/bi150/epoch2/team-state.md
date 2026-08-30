---
schema_version: 2
skill_version: 3.0.0
contract_version: 3
semantic_contract: typed-sketch-v1
attribution_contract: verdict-v1
runtime: claude-code
phase: stopped
workflow_status: stopped
run_epoch: 1
project_started_at: 2026-08-28T23:35:00Z
current_round: "002"
last_completed_round: "002"
last_accepted_round: "001"
last_accepted_kernel: triton_fused_moe_e2_001.py
last_accepted_report: rounds/report_001.md
last_completed_decision: rounds/decision_002.md
last_completed_sketch: rounds/sketch_002.json
last_completed_binding: rounds/binding_002.json
last_completed_verdict: rounds/verdict_002.json
last_attribution: none
last_completed_coder_result: rounds/coder_result_002.md
last_completed_report: rounds/report_002.md
last_result: no-improvement
performance_miss_streak: 1
failed_attempt_streak: 0
total_rounds: 2
max_rounds: 20
valid_no_improvement_limit: 3
adoption_threshold_pct: 5
measurement_fingerprint: fe73bc58146d8c16f524be2a00fe99b31e1b9678bca6b3702f4284a3ac0a5bef
target_mode: null
target_value: null
target_measurement_fingerprint: null
target_source: null
last_checkpoint_round: null
base_branch: kernel-opt/mmenc-attn-e2-20260828
base_commit: bb66016
run_branch: kernel-opt/fusedmoe-e2-20260828
measurement_exclusive: false
implementation_language: triton
implementation_backend: cuda
target_profile: triton_cuda
implementation_profile_snapshot_ref: profile_snapshot/triton_cuda.yaml
implementation_profile_snapshot_sha256: dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae
project_capability_claim_ref: profile_snapshot/capability_claim.json
project_capability_claim_sha256: fcba080f084be2791c43bbe45baaaff695cb2b4a72cc4053a3e070ae6912cff5
runtime_fingerprint_ref: project.md#runtime-fingerprint
blocked_incident: null
stop_reason: user-intervention
stop_timestamp: 2026-08-28T19:35:00Z
resume_eligible: always
resume_constraints: []
---

# Team State

Manifest updated only by the Orchestrator. Round artifacts back every value.

## Transition Log

| timestamp | from | to | result | evidence |
|---|---|---|---|---|
| 2026-08-28T23:35:00Z | — | initializing | - | phase0 scaffold (epoch2/ layout; epoch-1 archive preserved at ../) |
| 2026-08-28T23:55:00Z | initializing | ready | baseline | report_000 gate PASS: v0 canon 3.255288 ms (supersedes epoch-1 3.259 different fingerprint), identity -0.71%; LAUNCH CENSUS 123.95 kernels/call 21 distinct, DATA-DEPENDENT (148/134/64/64 CUDA ops vs active-expert count ~14 per expert) => graph capture over BASE is input-fragile, must target fixed-shape Triton candidate; device 967.85us/call device_ratio 0.2973 (host 70.3%); REAL TARGET = dispatch/indexing 635.3us/call 65.6% (scatter-store 127.4 + mask-gather 127.3 + DeviceSelect 125.8 + mask.any 86.5 + cub reduce 81.2), GEMMs only 118.8us/call 12.27%; fingerprint live-match + TWO positive controls reproduced |
| 2026-08-28T00:35:00Z | initializing | ready | - | designer phase-0 gate PASS (context sha logged; two-lever re-priced: LEVER A graph-as-is = WASH +8us central, LEVER B device-only = TRAP +94..+179 (eager taxes 2x85), BREAK-EVEN needs N_triton>=2 today=1; capture analysis: base NOT capturable (mask.any() = D2H sync x8 167.99us illegal in capture), candidate-002 capturable with static grid; 6 discrepancies ruled by orchestrator) |
| 2026-08-28T01:10:00Z | ready | designing | - | round-001 designer dispatch after six-ruling; F3/F1 family manual-graph-replay-fused with nw{1,2,4} folded as pre-adoption sweep |
| 2026-08-28T01:35:00Z | designing | coding | - | decision_001 gate PASS by orchestrator rerun (exit 0 "valid":true; decision sha 62820af457c7b0b84232dc28bffd07009b5bc1ee482059728da06761381fd1d5, sketch 6a46d4fd67b0cbce7a34ce41eac0c2b4cc19f00dd6e6098cf91a60e879634cb4; frozen snapshot UNTOUCHED per ruling a; pricing on 3.255288ms canon: graph-as-is ~-1.6% wash, break-even N_triton>=2, F3 band +26..+49% central +31%; branch-B observables declared unavailable-not-zero) |
| 2026-08-28T02:30:00Z | coding | verifying | candidate-ready | coder_001 gate PASS (candidate triton_fused_moe_e2_001.py sha da623fa92819185a1e20a8a7cbaca40acd9bfb4a3147f8e1e7b1e757c6b24cb7; pre-adoption sweep best_num_warps=1 BLOCK_M=16 grid(8,11) WON OUTRIGHT 92.855us vs nw2 122.253 — FR-4 sibling nw2 prior does NOT transfer, early unattributed sweep had wrongly picked nw4 (+38us trap caught by attribution); counting-sort grouped expert kernel 261.7->27.4us probe-instrumentation; capture-fired 3 tiers + budget trail 4-0 monotone + overflow->tier2 + permanent-once; 7 correctness suites incl. expert-activation variants (8/7/2/2/1 active) max_abs<=9.77e-4; fp16 dot re-qualified POSITIVE 2.44e-4 vs 1e-2 (mm_encoder negative does NOT transfer) + int64->int32 kernel narrowing now QUALIFIED; D1 RULED compliant: six tl.sum sites are int32 index tiles, softmax/topk/renorm/casts stay ATEN, waiver NEVER exercised; harness smoke 14.662x) |
| 2026-08-28T03:20:00Z | verifying | ready | accepted | report_001 @532fe3ea... verdict_001 @ac495b12... validate exit 0; wall v0 3.193262 -> v1 0.219792 ms, +93.248% vs canon 3.255288 (14.81x); correctness 12/12 incl 5 activation variants; census: tier-1 direct-address, ZERO python launcher executions in timed segment, 1 GraphLaunch + 1 copy-out = 2.0 submissions/call (was 9.82), 0 recaptures, 0 kernel events interior (branch-B confirmed); HOST WON BIG (423us replay-vs-eager vs modeled 170), DEVICE NEUTRAL not winning (282.5 vs 233.3 eager control is churn artifact; clean CUDA-event isolated 58.2 vs 55.95 = neutral) -> FR-2 fires but decision permits partial device landing; FR-4 fires nw1 wins 24.4%; binding_001.json missing -> orchestrator ruled: coder must produce canonical artifact, verdict not to be used for finalization until supplied |
| 2026-08-28T04:10:00Z | ready | designing | - | round-002 designer dispatch after G1 aliasing ruling; G1 option (ii) allocation-reuse with copy retained approved, option (i) persistent-return denied (harness retains v1_output at compare_case:731 AND reuses it as profile reference output buffer) |
| 2026-08-28T04:35:00Z | designing | coding | - | decision_002 gate PASS by orchestrator rerun (exit 0 "valid":true; decision sha dc782254a54331454290fac6791b7f583fff81d8de9699f03f5d06722fd7637e, sketch 015da3456f18582ad6114d3f5a0bfd14c5122a365bfbdd8031b1e543ecfe7ebe); family manual-graph-replay-fused scope=host, G1(ii) persistent out_dest as fixed copy-out TARGET with copy retained, expected 16.219us = 7.4% vs gate 10.99us; retention test encoded 3-level (observable retained_output_unchanged over 50 forwards byte-identical, FR-2 existence-check => failed_attempt not no-improvement, causal node feeds adoption_gate_cleared); G3 folded sweep BLOCK_M{16,32} + num_stages{1,2} exploratory; num_warps pinned 1; FR-3 host submissions must HOLD at 2.0; FR-4 device must not move >15us (eager controls must pre-bind workspaces or aten::fill_ churn fakes FR-4) |
| 2026-08-28T05:20:00Z | coding | verifying | candidate-ready | coder_002 SUCCESS-WITH-REGRESSION (candidate triton_fused_moe_e2_002.py sha ffd4dac3...; binding_002 @35a4500e... validate_binding VALID 22/23, op_alloc_dest unbound = profile has no allocation contract); RETENTION TEST PASS byte-identical over 50 forwards, 0 alias hits 3 tiers; sweep best_BLOCK_M=16 margin 9.829us, num_stages EXPLORATORY argmin default margin 0.031us INSIDE tie band => NOT adopted NOT recorded; num_warps pinned 1; FR-4 PASS device +0.058us with pre-bound control, kernels byte-identical; PREMISE FALSIFIED: shipped two-hop C2 = +5.113us vs r001 C1 4.491us; orchestrator independent re-measure: empty_like ~4.1us (NOT 16.219 as designer priced, NOT 0.005 as coder claimed) => G1 ceiling ~4us < 10.99us gate; alloc-free shape PROVABLY impossible (rotating pool fails retention at pool size, unsafe below ~150 forwards vs harness 150) |
| 2026-08-28T18:30:00Z | verifying | ready | no-improvement | report_002 @d67da1bb... verdict_002 @0a056897... validate exit 0; C3 cost-neutral as designed (+0.093us delta vs r001, signs mixed, noise class); retention test PASS independently (byte-identical over 50 changing-data forwards); FR-3 host submissions HOLD 2.0, FR-4 device +0.058us PASS; G1 premise permanently falsified (~4.13us empty_like ceiling < 10.99us gate); canonical UNCHANGED r001 @da623fa9... (14.81x); miss_streak 1/3; NOTE: verifier completion message never arrived, artifacts were authoritative and read directly |
| 2026-08-28T19:30:00Z | ready | stopped | accepted-final | G2 MEASUREMENT CLOSED THE LAST LEVER (verifier2 cold-rehydrate, rounds/pre_g2_measurement.md): four prelude numbers confirmed (topk 41.5 / sum 14.7 / div 6.8 / softmax 5.2 us); net G2 reclaim ~9-11us DEVICE but ~0us WALL (prelude already in-graph, submission holds 2.0, device off critical path under ~122us harness sync floor); GATE FLAG = softmax fold trips NOT-granted reduction.sum waiver (fp32 axis-k reduce), renorm-sum trips SAME waiver, topk frozen by tie semantics => only waiver-clean fold is fp16 cast ~1.6us; all three levers (G1 allocation / G2 prelude / device restructure) now closed -> campaign converges at r001 @da623fa9... 14.81x |
