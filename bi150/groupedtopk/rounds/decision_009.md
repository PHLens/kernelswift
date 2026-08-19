# Decision 009

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"009","reference_implementation":"triton_grouped_topk_008.py","reference_report":"rounds/report_008.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"host","change_family":"compile-mode-reduce-overhead"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"compile the accepted fixed-shape ModelNew target forward with torch.compile mode reduce-overhead while retaining its direct Triton stages and exact library torch.topk calls unchanged","allowed_changes":["set mode=reduce-overhead on the constructor-owned compiled callable","retain compiled dispatch only for the accepted constructor/configuration and contiguous fp32 target shape","fallback to the accepted eager two-stage forward for all unsupported graph/lifecycle cases"],"invariants":["accepted two-stage Triton algorithm and exact torch.topk group/final ordering","all-equal, two-expert-tie, and structured group-tie semantics","current caller device and stream","per-forward temporary buffer ownership and no user-visible cross-call aliasing","public constructor and forward contract","non-target fallback behavior","immutable base.py and unchanged harness","measurement fingerprint unchanged"],"expected_wall_improvement_pct":6.0}
```

## Unified Sketch

N/A: host-only change

## Host Plan

```json
{"applicability":"required","affected_scope":["ModelNew.__init__","constructor-owned compiled callable","accepted two-stage target forward"],"state_owner":"the ModelNew instance; the compiler-owned reduce-overhead graph state is private to its immutable compiled callable","lifetime":"create one reduce-overhead compiled callable in constructor; use it only for matching target-shape forwards; release it with the model instance","allocation_reuse":"no user-visible output or temporary tensor cache is added; the accepted target forward retains per-forward scores/group_scores/masked_scores allocations","cache_key":["torch version","compile mode","target device type","topk","renormalize","num_expert_group","topk_group","scoring_func","routed_scaling_factor","input shape","input dtype","input stride"],"invalidation":"do not dispatch the compiled callable when any cache-key field differs, after compile failure, or for unsupported shapes/dtypes/scoring modes; use accepted eager two-stage forward instead","concurrency":"the compiled callable owns no user-visible per-forward buffers; every invocation retains distinct scores/group_scores/masked_scores and concurrent forwards must not alias outputs","device_stream_behavior":"compile and execute on the caller-selected CUDA device and current stream; do not mutate device context or stream; compilation/graph capture overhead is excluded only through unchanged warmup","unchanged_behavior":["two direct Triton stages","exact torch.topk group/final selection and tie behavior","public constructor and forward signatures","fallback semantics","renormalization and routed scaling"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-009","intervention":"compile the accepted fixed-shape ModelNew target forward with torch.compile mode reduce-overhead while retaining its direct Triton stages and exact library torch.topk calls unchanged","expected_causal_chain":["CoreX reduce-overhead mode captures the accepted stable fixed-shape graph","launch and dispatch overhead for the surrounding compiled graph decreases without changing the two-stage dataflow","scoped wall median decreases by at least five percent while exact selection and device work do not regress materially"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"wall_time","expectation":"unrounded paired median improves at least five percent versus accepted triton_grouped_topk_008.py"},{"name":"device_us_per_call","expectation":"no material regression versus accepted compiled candidate"},{"name":"kernel_count_per_call","expectation":"no material regression versus accepted compiled candidate"}],"guardrails":["correctness:pass","exact topk_ids equality for seeded/all-equal/two-expert-tie/structured-group-tie inputs","allclose weights atol=1e-2 rtol=1e-2","same device/current stream and per-forward ownership","fallback on compile/graph incompatibility","unchanged constructor and forward signatures"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Retain Round 008's two direct Triton stages and exact library `torch.topk` calls; compiler mode is the only proposed change.
- Round 002 custom selection and Round 003 one-stage post-selection masking remain prohibited mechanism families.
- The matched probe establishes only mode availability. Failure to compile the full target graph, graph capture, adversarial tie behavior, stream/device ownership, or the 5% wall threshold rejects this candidate.
- Do not include compilation or graph-capture setup in benchmark samples beyond the unchanged harness warmup.

## Rationale and Evidence

`scripts/bi150_torch_compile_reduce_overhead_probe.py` compiled and executed a
file-backed CUDA function in `reduce-overhead` mode with exact output on the
recorded CoreX Torch `2.7.1` BI150 runtime. Round 008 established the accepted
default-mode compiled candidate at `0.344360 ms`, with `111.120595703125
us/call` and `8.96 kernels/call`; its device ratio of `0.3226872915` leaves a
substantial wall component outside attributed device time. The explicit mode
switch is a distinct, falsifiable host-runtime intervention. It is rejected by
any target-graph or lifecycle failure, exact-tie mismatch, material device
regression, or an unrounded paired median gain below five percent.
