# Decision 001

## Metadata

```json
{
  "schema_version": 2,
  "decision": "proceed",
  "decision_kind": "optimization",
  "round": "001",
  "reference_implementation": "baseline_adapter.py",
  "reference_report": "rounds/report_000.md",
  "language": "triton",
  "backend": "mlu",
  "runtime_fingerprint_ref": "project.md#runtime-fingerprint",
  "change_scope": "kernel",
  "change_family": "kernel-fusion",
  "sketch_ref": "rounds/sketch_001.json",
  "sketch_sha256": "__WRONG_SKETCH_SHA256__",
  "implementation_profile_snapshot_ref": "state/implementation_profile_snapshot/profile.yaml",
  "implementation_profile_snapshot_sha256": "__PROFILE_SHA256__",
  "project_capability_claim_ref": "state/project_capability_claim.json",
  "project_capability_claim_sha256": "__CLAIM_SHA256__"
}
```

## Optimization Intent

```json
{
  "bottleneck_class": "launch-bound",
  "intervention": "fuse routing reduction into the target kernel",
  "allowed_changes": ["kernel fusion"],
  "invariants": ["correctness:pass"],
  "expected_wall_improvement_pct": 10
}
```

## Unified Sketch

```json
{
  "artifact": "rounds/sketch_001.json",
  "sha256": "__SKETCH_SHA256__",
  "rendering": "see rounds/sketch_001.json"
}
```

## Host Plan

```json
{"applicability": "not-applicable", "reason": "kernel-only change"}
```

## Evaluation Contract

```json
{
  "hypothesis_id": "H-001",
  "intervention": "fuse routing reduction into the target kernel",
  "causal_graph": {
    "nodes": ["m.reduce-fusion", "o.external-kernel-count", "p.wall-time"],
    "edges": [
      ["m.reduce-fusion", "o.external-kernel-count"],
      ["o.external-kernel-count", "p.wall-time"]
    ]
  }
}
```

## Pitfalls and Anti-pattern Consultation

Consult the target profile before coding; no silent fallback or unproven capability may enter the candidate.

## Rationale and Evidence

The typed Sketch, frozen implementation profile snapshot, and project capability claim are validated before dispatch.
