---
schema_version: 1
skill_version: 2.0.0
runtime: unset
phase: stopped
workflow_status: stopped
run_epoch: 1
project_started_at: 2026-08-19T18:30:00Z
current_round: "001"
last_completed_round: "001"
last_accepted_round: null
last_accepted_kernel: null
last_accepted_report: null
last_completed_decision: rounds/decision_001.md
last_completed_coder_result: null
last_completed_report: rounds/report_000.md
last_result: aborted
performance_miss_streak: 0
failed_attempt_streak: 1
total_rounds: 1
max_rounds: 20
valid_no_improvement_limit: 3
adoption_threshold_pct: 5
measurement_fingerprint: 42673da1cdce1cce8b5e87c0e0b1780786eeb14cadaf6ef03d037fd7e2e336a7
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
stop_timestamp: 2026-08-19T19:30:00Z
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
| 2026-08-19T18:50:00Z | ready | 000 | baseline | baseline_adapter.py | - | - |
| 2026-08-19T19:00:00Z | designing | 001 | - | - | - | - |
| 2026-08-19T19:10:00Z | ready | 001 | aborted | - | - | - |
| 2026-08-19T19:30:00Z | stopped | 001 | - | baseline_adapter.py | - | - |

## Policy Revisions

This table is append-only. During an epoch, append only user target amendments
at a safe terminal boundary; begin a new epoch for any other policy change.

| Timestamp | Run epoch | Field | Old value | New value | Reason | Commit |
|---|---:|---|---|---|---|---|
| 2026-08-28T04:40:00Z | stopped | stopped | - | archival note by orchestrator: v3 successor campaign materialized at ./epoch2/ (contract_version 3); epoch-1 artifacts preserved intact in place; this manifest remains historical read-only |
