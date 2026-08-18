---
schema_version: 1
skill_version: 2.0.0
runtime: claude-code
phase: ready
workflow_status: running
run_epoch: 1
project_started_at: 2026-08-18T22:35:00Z
current_round: "000"
last_completed_round: "000"
last_accepted_round: "000"
last_accepted_kernel: baseline_adapter.py
last_accepted_report: rounds/report_000.md
last_completed_decision: null
last_completed_coder_result: null
last_completed_report: rounds/report_000.md
last_result: baseline
performance_miss_streak: 0
failed_attempt_streak: 0
total_rounds: 0
max_rounds: 20
valid_no_improvement_limit: 3
adoption_threshold_pct: 5
measurement_fingerprint: a1ee09ca54ab2210943bd030a6649c57d96b09d4c1beed863f4a98681ae425f2
target_mode: null
target_value: null
target_measurement_fingerprint: null
target_source: null
last_checkpoint_round: null
base_branch: dev
base_commit: e853319
run_branch: kernel-opt/rotary-embedding-s60
measurement_exclusive: false
implementation_language: triton
implementation_backend: gcu
target_profile: triton_gcu
runtime_fingerprint_ref: project.md#runtime-fingerprint
blocked_incident: null
stop_reason: null
stop_timestamp: null
resume_eligible: always
resume_constraints: []
---

# Team State

Only Orchestrator updates this manifest. `workflow_status` expresses the
campaign lifecycle as `running|blocked|stopped`; `phase` expresses the active
workflow step and is exactly
`initializing|ready|designing|coding|verifying|repairing|measuring|blocked|stopped`.
Round artifacts provide the detail behind every manifest value. Terminal results
are `accepted|no-improvement|screened-out|design-rejected|candidate-failed|aborted`.

## Transition Log

| Timestamp | Phase | Round | Result | Canonical | Incident | Commit |
|---|---|---:|---|---|---|---|
| 2026-08-18T22:35:00Z | initializing | 000 | - | - | - | - |
| 2026-08-18T22:35:00Z | ready | 000 | baseline | baseline_adapter.py | - | - |

## Policy Revisions

| Timestamp | Run epoch | Field | Old value | New value | Reason | Commit |
|---|---:|---|---|---|---|---|
