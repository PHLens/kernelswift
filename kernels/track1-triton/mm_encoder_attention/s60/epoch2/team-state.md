---
schema_version: 2
skill_version: 3.0.0
contract_version: 3
semantic_contract: typed-sketch-v1
attribution_contract: verdict-v1
runtime: claude-code
phase: ready
workflow_status: running
run_epoch: 1
project_started_at: 2026-08-28T23:20:00Z
current_round: "000"
last_completed_round: "000"
last_accepted_round: "000"
last_accepted_kernel: baseline_adapter.py
last_accepted_report: rounds/report_000.md
last_completed_decision: null
last_completed_sketch: null
last_completed_binding: null
last_completed_verdict: null
last_attribution: none
last_completed_coder_result: null
last_completed_report: rounds/report_000.md
last_result: baseline
performance_miss_streak: 0
failed_attempt_streak: 0
total_rounds: 0
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
stop_reason: null
stop_timestamp: null
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
