# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"gcu","target_profile":"triton_gcu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"none","intervention":"no intervention: the eager F.scaled_dot_product_attention already lowers to a single fused CNNL kernel (1 launch/call), and hand-written Triton SDPA is ~100x slower on device (proven in the flexattention s60 campaign)","allowed_changes":[],"invariants":["ModelNew public contract","output dtype and shape","numerical semantics"],"expected_wall_improvement_pct":0.0}
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

- The eager non-causal SDPA is a single fused CNNL kernel (1 `topsLaunchKernel`/call), identical to the flexattention s60 baseline. The flexattention s60 campaign already established the decisive fact: a hand-written Triton causal-SDPA kernel (per-(token,head) program, `tl.dot`) is correct but ~100x slower on device than the fused CNNL flash-attention kernel, because tiny per-head dot programs cannot match a library flash-attention kernel on GCU.

## Rationale and Evidence

`report_000.md` records the eager reference issues exactly 1 GCU runtime launch per forward call. There is no kernel-count headroom to exploit (contrast fused_moe's 147 launches). The only candidate direction — a hand-written Triton attention kernel — was already measured in the flexattention s60 campaign and found ~100x slower on device than the library SDPA. No falsifiable >=5% intervention exists; the campaign aborts as measurement-bound.
