---
schema_version: 2
skill_version: 3.0.0
contract_version: 3
semantic_contract: typed-sketch-v1
attribution_contract: verdict-v1
runtime: ascend910b4
phase: coding
workflow_status: running
run_epoch: 2
project_started_at: 2026-08-30T05:10:30Z
current_round: "005"
last_completed_round: "004"
last_accepted_round: "003"
last_accepted_kernel: triton_mm_encoder_attention_e2_003.py
last_accepted_report: rounds/report_003.md
last_completed_decision: rounds/decision_004.md
last_completed_sketch: rounds/sketch_004.json
last_completed_binding: null
last_completed_verdict: null
last_attribution: null
last_completed_coder_result: rounds/coder_result_004.md
last_completed_report: rounds/report_004.md
last_result: no-improvement
performance_miss_streak: 1
failed_attempt_streak: 0
total_rounds: 4
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
| 2026-08-30T06:30:00Z | coding -> verifying -> ready | 003 | accepted | `triton_mm_encoder_attention_e2_003.py` | - | `6312866` |
| 2026-08-30T07:55:00Z | coding -> verifying -> ready | 004 | no-improvement | unchanged (`triton_mm_encoder_attention_e2_003.py`) | - | `pending` |

## Policy Revisions

This table is append-only. During an epoch, append only user target amendments
at a safe terminal boundary; begin a new epoch for any other policy change.

| Timestamp | Run epoch | Field | Old value | New value | Reason | Commit |
|---|---:|---|---|---|---|---|
| 2026-08-30T05:50:54Z | 2 | maintainer_constraint.host_code | host-side code out of scope; host gain permitted only indirectly via launch-count reduction | host-side code authorized; `launch-path-reduction` and `allocation-reuse` Host Plan rounds permitted | maintainer explicit instruction after the round 002 abort. The device-only ceiling is 4.0902% against a 5% threshold (device budget 13.4064 us/call vs a 16.3885 us/call budget), launch count is already 1.00, and the ~316 us/call non-device residual is the only remaining lever with more than 16 us of headroom. Counters and run epoch are unchanged: this is a scope authorization, not a change to a frozen policy field. | `de1b9b7` |
| 2026-08-30T07:45:51Z | 2 | adoption_threshold.semantics | 5% improvement measured relative to the same-turn paired base.py reference | the epoch bar is beating the epoch-1 deliverable, not +5% per round; compared as speedup (reference/candidate) because epoch 1 and epoch 2 share the measurement fingerprint 1b1822d7... while the machine drifted about +9%, so absolute wall across epochs is not comparable. Epoch-1 bar: ref 0.348605 / cand 0.339685 = speedup 1.02626. | maintainer clarification. Recorded while round 004 was mid-verification; Verifier was advised to keep its protocol and additionally report per-pair speedup against the epoch-1 bar. Orchestrator applies the threshold at classification; Verifier evidence is unaffected. No prior classification needs revision: round 001 (speedup 1.100x) and round 003 (speedup 1.21065) both clear 1.02626. | `69e62f4` |
| 2026-08-30T07:48:26Z | 2 | adoption_threshold.semantics (supersedes the row above) | (previous row) the bar is beating the epoch-1 deliverable | the bar is +5% relative to the PREVIOUS ROUND accepted candidate, not +5% relative to base.py and not merely beating epoch 1. Cross-turn absolute wall comparison is invalid because the machine drifted about +9%, so the comparison is the ratio of speedups measured inside the same turn: accept when speedup(candidate)/speedup(last_accepted_kernel) - 1 >= 5%. Recheck under this rule: round 001 +10.90% and round 003 +8.59%, both clear, so no prior classification changes. Derived bar for round 004: speedup >= 1.27113. | maintainer clarification of the adoption bar, given while round 004 was mid-verification. | `pending` |
| 2026-08-30T07:55:00Z | 2 | adoption_threshold.measurement_method | ratio of speedups taken from separate windows or turns | both candidates must be measured in strict pair-by-pair alternation inside ONE window; a speedup is only comparable to another speedup measured against the same reference draws. Verifier measured, within a single turn, base.py median moving -5.96% (0.376040 -> 0.353615) while the candidate moved only -2.26% (0.295850 -> 0.289150), swinging e2_004's speedup by 4.13% (1.270171 -> 1.217744). A cross-window gap under ~0.1% is two orders of magnitude below the method's own instability and must not be defended. | verifier finding in round 004; also recorded in state/verifier_context.md. | `pending` |
