---
schema_version: 1
skill_version: 2.0.0
runtime: unset
phase: initializing
workflow_status: running
run_epoch: 1
project_started_at: null
current_round: "000"
last_completed_round: null
last_accepted_round: null
last_accepted_kernel: null
last_accepted_report: null
last_completed_decision: null
last_completed_coder_result: null
last_completed_report: null
last_result: null
performance_miss_streak: 0
failed_attempt_streak: 0
total_rounds: 0
max_rounds: 20
valid_no_improvement_limit: 3
adoption_threshold_pct: 5
measurement_fingerprint: null
target_mode: null
target_value: null
target_measurement_fingerprint: null
target_source: null
last_checkpoint_round: null
base_branch: null
base_commit: null
run_branch: null
measurement_exclusive: false
implementation_language: triton
implementation_backend: unset
target_profile: unset
runtime_fingerprint_ref: project.md#runtime-fingerprint
blocked_incident: null
stop_reason: null
stop_timestamp: null
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

## Policy Revisions

This table is append-only. During an epoch, append only user target amendments
at a safe terminal boundary; begin a new epoch for any other policy change.

| Timestamp | Run epoch | Field | Old value | New value | Reason | Commit |
|---|---:|---|---|---|---|---|
