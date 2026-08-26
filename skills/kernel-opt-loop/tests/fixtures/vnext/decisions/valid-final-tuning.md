# Decision 002

## Metadata

```json
{
  "schema_version": 2,
  "decision": "proceed",
  "decision_kind": "final-autotune",
  "artifact_index": "002",
  "reference_implementation": "accepted_candidate.py",
  "reference_report": "rounds/report_001.md",
  "language": "triton",
  "backend": "mlu",
  "runtime_fingerprint_ref": "project.md#runtime-fingerprint",
  "change_scope": "none",
  "change_family": "final-tuning",
  "sketch_ref": "rounds/sketch_001.json",
  "sketch_sha256": "__SKETCH_SHA256__",
  "implementation_profile_snapshot_ref": "state/implementation_profile_snapshot/profile.yaml",
  "implementation_profile_snapshot_sha256": "__PROFILE_SHA256__",
  "project_capability_claim_ref": "state/project_capability_claim.json",
  "project_capability_claim_sha256": "__CLAIM_SHA256__"
}
```


## Optimization Intent

```json
{
  "bottleneck_class": "configuration-tuning",
  "intervention": "tune launch and meta configuration against the accepted source",
  "allowed_changes": [],
  "invariants": [],
  "expected_wall_improvement_pct": 0
}
```


## Unified Sketch

```json
{
  "artifact": "rounds/sketch_001.json",
  "sha256": "__SKETCH_SHA256__",
  "rendering": "accepted sketch reused unchanged"
}
```


## Host Plan

```json
{"applicability": "not-applicable", "reason": "config-only finalization"}
```


## Evaluation Contract

```json
{"applicability": "not-applicable", "reason": "final tuning uses the Final Configuration Tuning contract"}
```


## Final Configuration Tuning

```json
{
  "submission_snapshot_id": "__SNAPSHOT_ID__",
  "anchors": {
    "accepted_candidate": {"ref": "accepted_candidate.py", "sha256": "__CANDIDATE_SHA256__"},
    "accepted_binding": {"ref": "rounds/binding_001.json", "sha256": "__BINDING_SHA256__"},
    "sketch": {"ref": "rounds/sketch_001.json", "sha256": "__SKETCH_SHA256__"},
    "profile": {"ref": "state/implementation_profile_snapshot/profile.yaml", "sha256": "__PROFILE_SHA256__"},
    "claim": {"ref": "state/project_capability_claim.json", "sha256": "__CLAIM_SHA256__"},
    "runtime_snapshot": {"ref": "state/runtime-snapshot.json", "sha256": "__RUNTIME_SNAPSHOT_SHA256__"},
    "measurement_fingerprint": {"sha256": "__MEASUREMENT_SHA256__"},
    "harness": {"ref": "auto_bench.py", "sha256": "__HARNESS_SHA256__"},
    "base": {"ref": "base.py", "sha256": "__BASE_SHA256__"}
  },
  "configurations": [{"num_warps": 1, "num_stages": 2}, {"num_warps": 2, "num_stages": 2}, {"num_warps": 2, "num_stages": 3}],
  "fallback_configuration": {"num_warps": 1, "num_stages": 2},
  "configuration_scope": {"shape_signature": "project-defined"},
  "max_trials": 3,
  "max_wall_seconds": 300,
  "warmup": 5,
  "repeat": 20,
  "mutation_reset": "fresh-process",
  "comparison_metric": "median-wall-time-ms",
  "tie_rule": "first-in-declared-order",
  "pin_selected_config": true
  
}
```

## Pitfalls and Anti-pattern Consultation

Final tuning changes configuration only; the accepted Sketch, algorithm, precision, effects, aliases, Host Plan, and public interface remain immutable.

## Rationale and Evidence

One offline bounded configuration-only gate per submission snapshot with exact-source confirmation and post-pin official verification.
