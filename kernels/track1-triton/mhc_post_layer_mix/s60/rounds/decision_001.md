# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"gcu","target_profile":"triton_gcu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"none","intervention":"no intervention: wall is dominated by a library einsum matmul (device-bound, launch overhead only 1.6%), and hand-written Triton tl.dot is Unknown on GCU and would not beat the CNNL matmul kernel","allowed_changes":[],"invariants":["ModelNew public contract","output dtype and shape","numerical semantics"],"expected_wall_improvement_pct":0.0}
```

## Unified Sketch

N/A: aborted

## Host Plan

```json
{"applicability":"not-applicable","reason":"aborted"}
```

## Evaluation Contract

```json
{"applicability":"not-applicable","reason":"aborted"}
```

## Pitfalls and Anti-pattern Consultation

- Launch overhead is ~67 us/call ≈ 1.6% of wall; the einsum `abmn,abmc->abnc` matmul is a large library GEMM that dominates device time. Kernel fusion has negligible headroom, and `tl.dot` is Unknown on GCU (would not beat CNNL matmul).

## Rationale and Evidence

`report_000.md` records 6 launches/call with 67 us launch overhead on a 4.27 ms wall (1.6%). The workload is device-matmul-bound, not launch-bound. No falsifiable >=5% intervention exists; abort as measurement-bound.
