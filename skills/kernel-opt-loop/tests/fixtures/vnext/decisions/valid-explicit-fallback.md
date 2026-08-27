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
  "bottleneck_class": "mechanism-bound",
  "intervention": "substitute explicit sum reduction while dot remains unproven",
  "allowed_changes": ["algorithm substitution"],
  "invariants": ["correctness:pass"],
  "expected_wall_improvement_pct": 5,
  "uses_algorithm_substitution": true,
  "fallback_provenance": {
    "fallback_from": "matrix.dot",
    "primary_signature": {"lhs_dtype": "fp16", "rhs_dtype": "fp16", "accumulator_dtype": "fp32", "layout": "blocked", "m": 16, "n": 128, "k": 64},
    "fallback_signature": {"dtype": "fp32", "axis": "k"},
    "fallback_kind": "algorithm-substitution",
    "probe_policy": "before-fallback",
    "qualification_disposition_id": "s60-attention-dot-fallback-001",
    "qualification_disposition_sha256": "__DISPOSITION_SHA256__",
    "primary_remains_unknown": true,
    "expected_causal_consequence": "external kernel count decreases"
  }
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
