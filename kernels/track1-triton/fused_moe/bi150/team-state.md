---
schema_version: 1
skill_version: 2.0.0
runtime: unset
phase: stopped
workflow_status: stopped
run_epoch: 1
project_started_at: 2026-08-19T19:45:00Z
current_round: "003"
last_completed_round: "003"
last_accepted_round: "002"
last_accepted_kernel: triton_fused_moe_002.py
last_accepted_report: rounds/report_002.md
last_completed_decision: rounds/decision_003.md
last_completed_coder_result: rounds/coder_result_002.md
last_completed_report: rounds/report_002.md
last_result: aborted
performance_miss_streak: 0
failed_attempt_streak: 1
total_rounds: 3
max_rounds: 20
valid_no_improvement_limit: 3
adoption_threshold_pct: 5
measurement_fingerprint: 5c2a51ab3f3ebaab1123b9fa534d4e4b940f3334f80fac00252df780d3900150
target_mode: null
target_value: null
target_measurement_fingerprint: null
target_source: null
last_checkpoint_round: null
base_branch: dev
base_commit: 453590c
run_branch: kernel-opt/bi150-all-20260818
measurement_exclusive: false
implementation_language: triton
implementation_backend: cuda
target_profile: triton_cuda
runtime_fingerprint_ref: project.md#runtime-fingerprint
blocked_incident: null
stop_reason: user-intervention
stop_timestamp: 2026-08-19T22:40:00Z
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

The policy fields are frozen for a run epoch except that a user may append an
optional comparable target at a safe terminal boundary. `measurement_exclusive`
is true only while Verifier owns local machine measurement.

## Transition Log

This table is append-only. Append one row for every phase transition; never edit
or remove an earlier row. Use `-` when a transition has no result, canonical
change, incident, or commit yet.

| Timestamp | Phase | Round | Result | Canonical | Incident | Commit |
|---|---|---:|---|---|---|---|
| 2026-08-19T20:00:00Z | ready | 000 | baseline | baseline_adapter.py | - | - |
| 2026-08-19T20:10:00Z | designing | 001 | - | - | - | - |
| 2026-08-19T20:25:00Z | coding | 001 | - | - | - | - |
| 2026-08-19T20:45:00Z | verifying | 001 | - | - | - | - |
| 2026-08-19T21:05:00Z | ready | 001 | accepted | triton_fused_moe_001.py | - | - |
| 2026-08-19T21:15:00Z | designing | 002 | - | - | - | - |
| 2026-08-19T21:30:00Z | coding | 002 | - | - | - | - |
| 2026-08-19T21:50:00Z | verifying | 002 | - | - | - | - |
| 2026-08-19T22:10:00Z | ready | 002 | accepted | triton_fused_moe_002.py | - | - |
| 2026-08-19T22:20:00Z | designing | 003 | - | - | - | - |
| 2026-08-19T22:35:00Z | ready | 003 | aborted | - | - | - |
| 2026-08-19T22:40:00Z | stopped | 003 | - | triton_fused_moe_002.py | - | - |

## Policy Revisions

This table is append-only. During an epoch, append only user target amendments
at a safe terminal boundary; begin a new epoch for any other policy change.

| Timestamp | Run epoch | Field | Old value | New value | Reason | Commit |
|---|---:|---|---|---|---|---|
