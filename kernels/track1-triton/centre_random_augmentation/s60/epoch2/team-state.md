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
project_started_at: 2026-08-29T13:20:00Z
current_round: "001"
last_completed_round: "001"
last_accepted_round: "001"
last_accepted_kernel: triton_centre_random_augmentation_e2_001.py
last_accepted_report: rounds/report_001.md
last_completed_decision: rounds/decision_001.md
last_completed_sketch: rounds/sketch_001.json
last_completed_binding: triton_centre_random_augmentation_e2_001.py
last_completed_verdict: rounds/verdict_001.json
last_attribution: none
last_completed_coder_result: rounds/coder_result_001.md
last_completed_report: rounds/report_001.md
last_result: accepted
performance_miss_streak: 0
failed_attempt_streak: 0
total_rounds: 1
max_rounds: 20
valid_no_improvement_limit: 3
adoption_threshold_pct: 5
measurement_fingerprint: cra-s60-e2
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
implementation_profile_snapshot_sha256: 7cd0cdf4b01b064b91f2b8f199cff6d12b175903a2c8d24ba7153f4d6a6aa6a0
project_capability_claim_ref: profile_snapshot/capability_claim.json
runtime_fingerprint_ref: project.md#runtime-fingerprint
blocked_incident: null
stop_reason: accepted (+47.60%, launch-fusion decisive; remaining center-reduction fusion is marginal ~6% and needs cross-program reduction, not worth a round)
stop_timestamp: 2026-08-29T13:40:00Z
resume_eligible: always
resume_constraints: []
---

# Team State

Manifest updated only by the Orchestrator.

## Transition Log

| timestamp | from | to | result | evidence |
|---|---|---|---|---|
| 2026-08-29T13:20:00Z | — | initializing | - | phase0 scaffold (centre_random_augmentation/s60/epoch2, triton_gcu profile snapshot reused) |
| 2026-08-29T13:21:00Z | initializing | ready | baseline | report_000 baseline identity; base launch-bound 78 topsLaunchKernel/call |
| 2026-08-29T13:22:00Z | ready | designing | - | round-001 designer dispatch; preflight: full fusion (quaternion->R + rot_vec_mul + translation + mask into single kernel, grid=(n_sample,)) ~1.59x, correctness 4.77e-7 |
| 2026-08-29T13:40:00Z | designing | stopped | accepted | round-001 accepted: +47.60% (base 3.025ms -> candidate 1.585ms, 1.90x). Launch collapse 96->10 topsLaunchKernel/call, launch-API ~922->118us. correctness 4/4 exact-match PASS. S60 FIRST operator to beat base (fused_moe class, launch-bound). CAMPAIGN TERMINAL. Best deliverable = triton_centre_random_augmentation_e2_001.py |
