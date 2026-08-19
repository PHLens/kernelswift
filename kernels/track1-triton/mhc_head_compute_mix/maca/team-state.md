---
schema_version: 1
skill_version: 2.0.0
runtime: codex
phase: stopped
workflow_status: stopped
run_epoch: 1
project_started_at: 2026-08-19T00:10:00Z
current_round: "002"
last_completed_round: "002"
last_accepted_round: "001"
last_accepted_kernel: triton_mhcc_001.py
last_accepted_report: rounds/report_001.md
last_completed_decision: rounds/decision_002.md
last_completed_coder_result: rounds/coder_result_001.md
last_completed_report: rounds/report_001.md
last_result: aborted
performance_miss_streak: 0
failed_attempt_streak: 1
total_rounds: 2
max_rounds: 20
valid_no_improvement_limit: 3
adoption_threshold_pct: 5
measurement_fingerprint: d8f4b63bfbf09ce8a32f3bdcd4d85553f34abce7384e495ba5f66baf49bf795e
target_mode: null
target_value: null
target_measurement_fingerprint: null
target_source: null
last_checkpoint_round: null
base_branch: dev
base_commit: 138f0caf13784399abdc29507a6ac1f29e0fd947
run_branch: kernel-opt/mhc-head-compute-mix-c500-20260819
measurement_exclusive: false
implementation_language: triton
implementation_backend: maca
target_profile: triton_maca
runtime_fingerprint_ref: project.md#runtime-fingerprint
blocked_incident: "Round 002 Designer abort (measurement-bound/latency-floor): single-launch device 43.79 us at serial Sinkhorn latency floor; host 63% is harness-fixed set_seed+sync_devices. No candidate-owned >=5% path."
stop_reason: user-intervention
stop_timestamp: 2026-08-19T01:00:00Z
resume_eligible: always
resume_constraints: []
---

# Team State

Only Orchestrator updates the manifest. `workflow_status` expresses the
campaign lifecycle as `running|blocked|stopped`; `phase` expresses the active
workflow step and is exactly
`initializing|ready|designing|coding|verifying|repairing|measuring|blocked|stopped`.
Round artifacts provide the detail behind every manifest value. Terminal results
are `accepted|no-improvement|screened-out|design-rejected|candidate-failed|aborted`.

## Transition Log

| Timestamp | Phase | Round | Result | Canonical | Incident | Commit |
|---|---|---:|---|---|---|---|
| 2026-08-19T00:10:00Z | initializing | 000 | - | - | - | - |
| 2026-08-19T00:20:00Z | ready | 000 | baseline | baseline_adapter.py | - | <pending> |
| 2026-08-19T00:30:00Z | coding | 001 | - | baseline_adapter.py | - | - |
| 2026-08-19T00:40:00Z | ready | 001 | accepted | triton_mhcc_001.py | - | <pending> |
| 2026-08-19T00:50:00Z | blocked | 002 | aborted | triton_mhcc_001.py | latency-bound (await user) | <pending> |
| 2026-08-19T01:00:00Z | stopped | 002 | aborted | triton_mhcc_001.py | user-intervention (stop accepted) | <pending> |

## Policy Revisions

| Timestamp | Run epoch | Field | Old value | New value | Reason | Commit |
|---|---:|---|---|---|---|---|
