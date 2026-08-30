---
schema_version: 2
skill_version: 3.0.0
contract_version: 3
semantic_contract: typed-sketch-v1
attribution_contract: verdict-v1
runtime: ascend910b4
phase: ready
workflow_status: running
run_epoch: 2
project_started_at: 2026-08-30T05:10:30Z
current_round: "002"
last_completed_round: "002"
last_accepted_round: "001"
last_accepted_kernel: triton_mm_encoder_attention_e2_001.py
last_accepted_report: rounds/report_001.md
last_completed_decision: rounds/decision_002.md
last_completed_sketch: null
last_completed_binding: null
last_completed_verdict: null
last_attribution: null
last_completed_coder_result: null
last_completed_report: null
last_result: aborted
performance_miss_streak: 0
failed_attempt_streak: 1
total_rounds: 2
max_rounds: 20
valid_no_improvement_limit: 3
adoption_threshold_pct: 5
measurement_fingerprint: 1b1822d7b74a8cd41411a27fcbc18a89cb50b1cfefb9fdac2585cdd520e9a79a
target_mode: null
target_value: null
target_measurement_fingerprint: null
target_source: null
last_checkpoint_round: null
base_branch: trunk
base_commit: db09613
run_branch: kernel-opt/mmenc-attn-e2-ascend-20260830
measurement_exclusive: false
implementation_language: triton
implementation_backend: ascend
target_profile: triton_ascend
implementation_profile_snapshot_ref: state/implementation_profile_snapshot/profile.yaml
implementation_profile_snapshot_sha256: a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321
project_capability_claim_ref: state/project_capability_claim.json
project_capability_claim_sha256: a46ce09de93c671865f2c1b661a335a8b7f5714db475a27bae763f9d84a56b9d
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

This is run epoch 2 on the Ascend 910B target. Epoch-1 artifacts under `../`
remain read-only history and are never migrated. The epoch-1 deliverable
`triton_attn_001.py` at 0.92x is preserved intact.

The frozen profile snapshot resolves the entire root-confined
implementation-profile closure (profile.yaml, vendored schema copies, probe
definitions and inputs, and approved evidence records) so validation still passes
if the canonical profile directory changes or disappears. fp16 `tl.dot`,
`num_warps`, and `num_stages` are `constrained` with approved evidence;
`make_block_ptr`, `async_copy`, `vectorize`, and a fast launcher remain Unknown
and may not be declared normative.

## Transition Log

This table is append-only. Append one row for every phase transition; never edit
or remove an earlier row. Use `-` when a transition has no result, canonical
change, incident, or commit yet.

| Timestamp | Phase | Round | Result | Canonical | Incident | Commit |
|---|---|---:|---|---|---|---|
| 2026-08-30T05:10:30Z | initializing -> ready | 000 | baseline | `baseline_adapter.py` | - | `230a378` |
| 2026-08-30T05:38:00Z | ready -> designing -> coding -> verifying -> ready | 001 | accepted | `triton_mm_encoder_attention_e2_001.py` | - | `cd44339` |
| 2026-08-30T06:05:00Z | ready -> designing -> ready | 002 | aborted | unchanged (`triton_mm_encoder_attention_e2_001.py`) | - | `46d1135` |

## Policy Revisions

This table is append-only. During an epoch, append only user target amendments
at a safe terminal boundary; begin a new epoch for any other policy change.

| Timestamp | Run epoch | Field | Old value | New value | Reason | Commit |
|---|---:|---|---|---|---|---|
