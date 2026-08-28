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
project_started_at: 2026-08-29T00:52:00Z
current_round: "001"
last_completed_round: "001"
last_accepted_round: "000"
last_accepted_kernel: baseline_adapter.py
last_accepted_report: rounds/report_000.md
last_completed_decision: rounds/decision_001.md
last_completed_sketch: rounds/sketch_001.json
last_completed_binding: triton_flexattention_e2_001.py
last_completed_verdict: rounds/verdict_001.json
last_attribution: none
last_completed_coder_result: rounds/coder_result_001.md
last_completed_report: rounds/report_001.md
last_result: no-improvement
performance_miss_streak: 1
failed_attempt_streak: 0
total_rounds: 1
max_rounds: 20
valid_no_improvement_limit: 3
adoption_threshold_pct: 5
measurement_fingerprint: flexattention-s60-e2
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
runtime_fingerprint_ref: project.md#runtime-fingerprint
blocked_incident: null
stop_reason: device-bound (causal attention hand-written fp16 tl.dot ~0.94x ceiling, vendor flash-attention already holds the device floor; 2.2x-over-epoch-1 deliverable banked)
stop_timestamp: 2026-08-29T01:05:00Z
resume_eligible: always
resume_constraints: []
---

# Team State

Manifest updated only by the Orchestrator.

## Transition Log

| timestamp | from | to | result | evidence |
|---|---|---|---|---|
| 2026-08-29T00:52:00Z | — | initializing | - | phase0 scaffold (flexattention/s60/epoch2, triton_gcu profile snapshot reused from mm_encoder e2) |
| 2026-08-29T00:53:00Z | initializing | ready | baseline | report_000 baseline: wall v0~0.252ms identity (1.00x); census 2 topsLaunchKernel/call @19.43us; causal SDPA |
| 2026-08-29T00:55:00Z | ready | designing | - | round-001 designer dispatch; preflight probe: causal fp16 QK^T + fp32 PV single-tile TP=128 nw1 correctness PASS (max_abs_diff 1.95e-3); authoritative 3-pair = 0.94x (0.2686 vs 0.2521ms) — 2.2x over epoch-1 0.42x, device-bound ceiling like mm_encoder |
| 2026-08-29T01:05:00Z | designing | stopped | no-improvement | round-001 terminal: correctness 6/6 PASS, candidate ~0.94x (5-pair paired -6.4%). Causal fp16 tl.dot single kernel, dispatch collapse at aten level, launch 1.0/call unchanged. S60 device-bound: vendor causal flash-attention holds floor, TP=128 power-of-2 padding 58% FLOP waste. 2.2x-over-epoch-1 deliverable banked (0.42x -> 0.94x). CAMPAIGN TERMINAL (device-bound). Best deliverable = triton_flexattention_e2_001.py |
