# Decision 008

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"008","reference_implementation":"triton_grouped_topk_004.py","reference_report":"rounds/report_004.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"host","change_family":"compiled-dispatch"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"compile the accepted fixed-shape ModelNew target forward through torch.compile while retaining its two direct Triton stages and exact library torch.topk calls unchanged","allowed_changes":["one constructor-owned torch.compile callable for the accepted fixed [83,256] softmax route","compiled dispatch only for the accepted constructor/configuration and contiguous fp32 target shape","fallback to the accepted eager two-stage forward for all unsupported graph/lifecycle cases"],"invariants":["accepted two-stage Triton algorithm and exact torch.topk group/final ordering","all-equal, two-expert-tie, and structured group-tie semantics","current caller device and stream","per-forward temporary buffer ownership and no aliasing","public constructor and forward contract","non-target fallback behavior","immutable base.py and unchanged harness","measurement fingerprint unchanged"],"expected_wall_improvement_pct":6.0}
```

## Unified Sketch

N/A: host-only change

## Host Plan

```json
{"applicability":"required","affected_scope":["ModelNew.__init__","ModelNew.forward","constructor-owned compiled callable","accepted two-stage target forward"],"state_owner":"the ModelNew instance; compiled callable is immutable after construction","lifetime":"create one compiled callable in constructor; use it only for matching target-shape forwards; release with the model instance","allocation_reuse":"no output or temporary tensor cache is added; each forward retains accepted per-forward allocations","cache_key":["torch version","target device type","topk","renormalize","num_expert_group","topk_group","scoring_func","routed_scaling_factor","input shape","input dtype","input stride"],"invalidation":"do not dispatch the compiled callable when any cache-key field differs, after compile failure, or for unsupported shapes/dtypes/scoring modes; use accepted eager two-stage forward instead","concurrency":"compiled callable owns no per-forward buffers; every invocation still allocates distinct scores/group_scores/masked_scores and concurrent forwards must not alias","device_stream_behavior":"compile and execute on the caller-selected CUDA device and current stream; do not mutate device context or stream; compile overhead is excluded only through unchanged warmup","unchanged_behavior":["two direct Triton stages","exact torch.topk group/final selection and tie behavior","public constructor and forward signatures","fallback semantics","renormalization and routed scaling"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-008","intervention":"compile the accepted fixed-shape ModelNew target forward through torch.compile while retaining its two direct Triton stages and exact library torch.topk calls unchanged","expected_causal_chain":["CoreX torch.compile captures a stable fixed-shape forward dispatch","eager Python and compatible framework dispatch overhead decreases without altering the two-stage dataflow","scoped wall median decreases by at least five percent while device kernel count/time do not regress materially"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"wall_time","expectation":"unrounded paired median improves at least five percent versus accepted triton_grouped_topk_004.py"},{"name":"device_us_per_call","expectation":"no material regression versus accepted candidate"},{"name":"kernel_count_per_call","expectation":"no material regression versus accepted candidate"}],"guardrails":["correctness:pass","exact topk_ids equality for seeded/all-equal/two-expert-tie/structured-group-tie inputs","allclose weights atol=1e-2 rtol=1e-2","same device/current stream and per-forward ownership","fallback on compile/graph incompatibility","unchanged constructor and forward signatures"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Round 004 remains algorithmically immutable for this decision: both direct Triton stages and exact library top-k calls must remain in the compiled callable.
- Round 002 custom selection and Round 003 one-stage masking are rejected mechanisms, not compiler fallbacks.
- The probe proves only CUDA compiler availability; graph breaks, compiler failure, altered streams, changed allocations, or failed adversarial ties reject this candidate.
- Do not expose compile overhead in measured samples beyond the unchanged harness warmup.

## Rationale and Evidence

`scripts/bi150_torch_compile_probe.py` compiled and executed a file-backed CUDA
function on the recorded CoreX Torch 2.7.1 runtime with exact output. The
accepted candidate's device ratio is `0.2945183072`, leaving a potentially
meaningful dispatch/host component in the `0.432098 ms` wall median. This is a
new mechanism class that does not change active sets, top-k ordering, or kernel
algorithm. It is falsified by any compile/graph/lifecycle failure, any
correctness mismatch, or an unrounded paired median gain below five percent.
