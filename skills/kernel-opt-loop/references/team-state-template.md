---
schema_version: 1
skill_version: 2.0.0
runtime: unset
phase: initializing
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
measurement_fingerprint: null
implementation_language: triton
implementation_backend: mlu
target_profile: triton_mlu
runtime_fingerprint_ref: project.md#runtime-fingerprint
blocked_incident: null
stop_reason: null
stop_timestamp: null
resume_eligible: always
resume_constraints: []
---

# Team State

Only Orchestrator updates the manifest. The allowed manifest phases are exactly
`initializing|ready|designing|coding|verifying|repairing|measuring|blocked|stopped`.
Round artifacts provide the detail behind every manifest value.

## Transition Log

This table is append-only. Append one row for every phase transition; never edit
or remove an earlier row. Use `-` when a transition has no result, canonical
change, incident, or commit yet.

| Timestamp | Phase | Round | Result | Canonical | Incident | Commit |
|---|---|---:|---|---|---|---|
