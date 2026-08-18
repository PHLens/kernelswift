---
schema_version: 1
skill_version: 2.0.0
runtime: codex
phase: ready
workflow_status: running
run_epoch: 1
project_started_at: 2026-08-18T05:01:57Z
current_round: "001"
last_completed_round: "001"
last_accepted_round: "001"
last_accepted_kernel: triton_grouped_topk_001.py
last_accepted_report: rounds/report_001.md
last_completed_decision: rounds/decision_001.md
last_completed_coder_result: rounds/coder_result_001.md
last_completed_report: rounds/report_001.md
last_result: accepted
performance_miss_streak: 0
failed_attempt_streak: 0
total_rounds: 1
max_rounds: 20
valid_no_improvement_limit: 3
adoption_threshold_pct: 5
measurement_fingerprint: 3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809
target_mode: null
target_value: null
target_measurement_fingerprint: null
target_source: null
last_checkpoint_round: null
base_branch: dev
base_commit: 6a970c921dfb0c031b885190122ce1335d8d4cd7
run_branch: kernel-opt/grouptopk-c500-20260818
measurement_exclusive: false
implementation_language: triton
implementation_backend: maca
target_profile: triton_maca
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
| 2026-08-18T05:01:57Z | initializing | 000 | - | - | - | - |
| 2026-08-18T05:20:04Z | verifying | 000 | - | `baseline_adapter.py` | - | - |
| 2026-08-18T06:03:44Z | ready | 000 | baseline | `baseline_adapter.py` | - | - |
| 2026-08-18T06:06:55Z | designing | 001 | - | `baseline_adapter.py` | - | `52649df` |
| 2026-08-18T06:19:35Z | coding | 001 | - | `baseline_adapter.py` | - | `52649df` |
| 2026-08-18T06:42:00Z | verifying | 001 | - | `baseline_adapter.py` | - | `52649df` |
| 2026-08-18T07:13:53Z | ready | 001 | accepted | `triton_grouped_topk_001.py` | - | - |

## Policy Revisions

This table is append-only. During an epoch, append only user target amendments
at a safe terminal boundary; begin a new epoch for any other policy change.

| Timestamp | Run epoch | Field | Old value | New value | Reason | Commit |
|---|---:|---|---|---|---|---|
