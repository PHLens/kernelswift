---
schema_version: 1
skill_version: 2.0.0
runtime: unset
phase: stopped
workflow_status: stopped
run_epoch: 1
project_started_at: 2026-08-18T17:20:00Z
current_round: "002"
last_completed_round: "002"
last_accepted_round: "001"
last_accepted_kernel: triton_mhc_post_layer_mix_001.py
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
measurement_fingerprint: c17c7f45d44b1bec047c7c3a315275d33b21af3226dd34e884d483899ef039b6
target_mode: null
target_value: null
target_measurement_fingerprint: null
target_source: null
last_checkpoint_round: null
base_branch: dev
base_commit: e8533192f65ed4610a4b59859f1969ea83955f87
run_branch: kernel-opt/mhc_post_layer_mix-bi150-20260818
measurement_exclusive: false
implementation_language: triton
implementation_backend: cuda
target_profile: triton_cuda
runtime_fingerprint_ref: project.md#runtime-fingerprint
blocked_incident: null
stop_reason: user-intervention
stop_timestamp: 2026-08-18T20:25:00Z
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
| 2026-08-18T18:30:00Z | ready | 000 | baseline | baseline_adapter.py | - | - |
| 2026-08-18T18:45:00Z | designing | 001 | - | - | - | - |
| 2026-08-18T19:00:00Z | coding | 001 | - | - | - | - |
| 2026-08-18T19:20:00Z | verifying | 001 | - | - | - | - |
| 2026-08-18T19:45:00Z | ready | 001 | accepted | triton_mhc_post_layer_mix_001.py | - | - |
| 2026-08-18T20:00:00Z | designing | 002 | - | - | - | - |
| 2026-08-18T20:15:00Z | ready | 002 | aborted | - | - | - |
| 2026-08-18T20:25:00Z | stopped | 002 | - | triton_mhc_post_layer_mix_001.py | - | - |

## Policy Revisions

This table is append-only. During an epoch, append only user target amendments
at a safe terminal boundary; begin a new epoch for any other policy change.

| Timestamp | Run epoch | Field | Old value | New value | Reason | Commit |
|---|---:|---|---|---|---|---|
