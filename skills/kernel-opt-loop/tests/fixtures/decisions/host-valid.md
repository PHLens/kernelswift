# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"002","reference_implementation":"triton_example_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"mlu","target_profile":"triton_mlu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"host","change_family":"host-allocation-reuse"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"reuse the output allocation across compatible forwards","allowed_changes":["ModelNew.forward","output cache"],"invariants":["ModelNew public contract","output dtype and shape","numerical semantics"],"expected_wall_improvement_pct":6.0}
```

## Unified Sketch

N/A: host-only change

## Host Plan

```json
{"applicability":"required","affected_scope":["ModelNew.forward","output cache"],"state_owner":"ModelNew instance","lifetime":"model lifetime","allocation_reuse":"reuse when shape, dtype, and device match","cache_key":["shape","dtype","device"],"invalidation":"replace on cache-key change","concurrency":"one model instance is not shared across concurrent forwards","device_stream_behavior":"caller-selected device and current stream are preserved","unchanged_behavior":["returned shape","returned dtype","numerical semantics"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-002","intervention":"reuse the output allocation across compatible forwards","expected_causal_chain":["output allocations per call decrease","host overhead decreases","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"output_allocations_per_call","expectation":"decrease"},{"name":"host_us_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output dtype and shape unchanged","current stream preserved"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`; cache invalidation and stream ownership are explicit in the Host Plan.

## Rationale and Evidence

Repeated forwards use compatible output shapes, dtypes, and devices while allocation overhead remains measurable.
