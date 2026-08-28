---
schema_version: 2
skill_version: 3.0.0
contract_version: 3
semantic_contract: typed-sketch-v1
attribution_contract: verdict-v1
runtime: claude-code
phase: stopped
workflow_status: terminal
run_epoch: 1
project_started_at: 2026-08-28T23:20:00Z
current_round: "002"
last_completed_round: "002"
last_accepted_round: "000"
last_accepted_kernel: baseline_adapter.py
last_accepted_report: rounds/report_000.md
last_completed_decision: rounds/decision_002.md
last_completed_sketch: rounds/sketch_002.json
last_completed_binding: triton_mm_encoder_attention_e2_002.py
last_completed_verdict: rounds/verdict_002.json
last_attribution: none
last_completed_coder_result: rounds/coder_result_002.md
last_completed_report: rounds/report_002.md
last_result: no-improvement
performance_miss_streak: 2
failed_attempt_streak: 0
total_rounds: 2
max_rounds: 20
valid_no_improvement_limit: 3
adoption_threshold_pct: 5
measurement_fingerprint: c335b39cbf2eaa15e1a358be90d0aab85d0fd7e8ffd4b7b4e825df0901ad61f9
target_mode: null
target_value: null
target_measurement_fingerprint: null
target_source: null
last_checkpoint_round: 0
base_branch: dev
base_commit: 91a1a89
run_branch: kernel-opt/mm-encoder-attention-s60-e2
measurement_exclusive: false
implementation_language: triton
implementation_backend: gcu
target_profile: triton_gcu
implementation_profile_snapshot_ref: profile_snapshot/triton_gcu.yaml
implementation_profile_snapshot_sha256: 8dfabd0af59b8f6640b47179fee19bca2f5fe35b18535a3db24f60c842e42b70
project_capability_claim_ref: profile_snapshot/capability_claim.json
project_capability_claim_sha256: a175f2727b9198a92da978aca9e8f87834a74884372746699412931890d9748e
runtime_fingerprint_ref: project.md#runtime-fingerprint
blocked_incident: null
stop_reason: complete (deliverable shipped: e2_002 ~0.915x vs base, ~3.39x vs epoch-1 0.27x; further rounds would chase the vendor-library device floor, which is out of scope under the delivery standard)
stop_timestamp: 2026-08-29T00:10:00Z
resume_eligible: always
resume_constraints: []
---

# Team State

Manifest updated only by the Orchestrator. Round artifacts back every value.

## Transition Log

| timestamp | from | to | result | evidence |
|---|---|---|---|---|
| 2026-08-28T23:20:00Z | — | initializing | - | phase0 scaffold commit (s60/epoch2/ layout, epoch-1 archive preserved at ../; triton_gcu profile v1 onboarded) |
| 2026-08-28T23:22:00Z | initializing | ready | baseline | report_000 gate PASS (wall v0=0.227194 v1=0.228385 identity speedup 0.995x; census: base SDPA -> _scaled_dot_product_flash_attention 2 launches/call topsLaunchKernel 21.99us/call, 8 transpose + 8 as_strided + 4 view + 3 empty + empty_like + empty_strided + reshape aten ops/call; device-duration unavailable on GCU trace, runtime-launch evidence recorded; measurement_fingerprint c335b39c...) |
| 2026-08-28T23:31:00Z | ready | designing | - | round-001 designer dispatch; capability preflight probe (orchestrator-scoped): tl.arange power-of-2, tl.max/tl.sum no-keepdim, tl.dot same-dtype, tl.dot power-of-2 (NOT mult-of-16; 96=16x6 FAILS); direct-Triton MHA pad128 nw2 = 148.6us vs base 139.9us (-6.2%); launcher tax 17.4us/call (far below bi150 84.77us); num_warps 2 optimal |
| 2026-08-28T23:56:00Z | designing | ready | no-improvement | round-001 terminal: correctness PASS, candidate ~0.906x (paired -10.5% vs +5% bar). S60 DEVICE-BOUND: hand tl.dot (TP=128 padding 58% FLOP waste) device ~166us slower than CNNL SDPA ~158us floor; launcher tax 17.4us + host chain 11us << device deficit, graph-replay has no prize. tl.dot/tl.arange power-of-2 constraint written back to triton_gcu profile. Deliverable banked (correctness-PASS Triton, forward + 4-arg run_out) |
| 2026-08-29T00:10:00Z | ready | stopped | no-improvement | round-002 terminal: correctness 4/4 PASS, candidate ~0.915x (paired -9.32%). fp16 QK^T dot direction confirmed real (r001 -10.5% -> r002 -9.3%, ~30.6us device-side gain) but S60 remains device-bound; TP=128 power-of-2 padding 58% FLOP waste is structurally unavoidable, CNNL SDPA ~158us floor not crossed. performance_miss_streak=2, no remaining device lever (fp32->fp16 exhausted, num_warps 2->1 exhausted, grid-split measured worse in r001, graph-replay no prize at 17.4us launcher tax). CAMPAIGN TERMINAL (measurement-bound). Best deliverable = e2_002.py |
