# Decision Contract Template

Use this contract for every `rounds/decision_NNN.md`. The JSON blocks and the
Unified Sketch are normative. The final two prose sections explain the decision
but must not add implementation requirements that are absent from Optimization
Intent, Unified Sketch, or Host Plan.

Run the validator before handing a proceeding decision to Coder:

```bash
python3 <skill-root>/scripts/validate_decision.py \
  rounds/decision_NNN.md --expected-profile triton_mlu
```

## Required Sections and Fields

Every decision has these seven H2 headings, once each and in this spelling:

1. `Metadata`
2. `Optimization Intent`
3. `Unified Sketch`
4. `Host Plan`
5. `Evaluation Contract`
6. `Pitfalls and Anti-pattern Consultation`
7. `Rationale and Evidence`

Metadata is one fenced JSON object with these fields:

| Field | Type and requirement |
|---|---|
| `schema_version` | Integer `1` |
| `decision` | `proceed` or `abort` |
| `round` | Three-digit string matching the decision title |
| `reference_implementation` | Relative path to the accepted implementation |
| `reference_report` | Relative path to the accepted report |
| `language` | Implementation language, `triton` for the v1 profile |
| `backend` | Runtime backend, `mlu` for the v1 profile |
| `target_profile` | Discovered profile, `triton_mlu` in v1 |
| `runtime_fingerprint_ref` | Relative project reference with an anchor |
| `change_scope` | `kernel`, `host`, `mixed`, or `none` |

Optimization Intent is one fenced JSON object with `bottleneck_class`, one
falsifiable `intervention`, `allowed_changes`, `invariants`, and numeric
`expected_wall_improvement_pct`. A proceeding decision has at least one allowed
change and invariant. An abort may have no allowed changes but still records its
invariants and the reason for stopping as its intervention.

Evaluation Contract is one fenced JSON object with `hypothesis_id`, the exact
same `intervention`, a nonempty `expected_causal_chain`, `primary_metric`, one or
more `mechanism_observables`, nonempty `guardrails`, and `profiling_level`. The
primary metric is always:

```json
{"name":"wall_time","expected_improvement_pct":5.0}
```

`guardrails` includes `correctness:pass`. `profiling_level` is exactly one of:

- `summary`: required scoped summary evidence only;
- `targeted`: summary evidence plus the intent-specific probes named by the
  mechanism observables;
- `deep-on-demand`: targeted evidence plus a complete trace investigation when
  conflicting or noisy results require it.

The selected profiling mode never waives mandatory evidence:

- Level 0 for every candidate: correctness and interleaved paired wall timing.
- Level 1 after correctness passes: separately scoped reference and candidate
  device time per call, kernel count per call, and top-k kernel breakdown.

## Scope Requirements

| `decision` / `change_scope` | Unified Sketch | Host Plan | Evaluation Contract |
|---|---|---|---|
| `proceed` / `kernel` | Required Sketch | `not-applicable` with a nonempty reason | Required |
| `proceed` / `host` | Exact marker `N/A: host-only change` | Required | Required |
| `proceed` / `mixed` | Required Sketch | Required | Required |
| `abort` / `none` | Exact marker `N/A: aborted` | Exact aborted object | Exact aborted object |

A required Host Plan uses `applicability: required` and all of these fields:

- `affected_scope`
- `state_owner`
- `lifetime`
- `allocation_reuse`
- `cache_key`
- `invalidation`
- `concurrency`
- `device_stream_behavior`
- `unchanged_behavior`

The list fields `affected_scope`, `cache_key`, and `unchanged_behavior` must be
nonempty. Every other required Host Plan field is a nonempty string.

## Unified Sketch Grammar

A kernel or mixed decision contains exactly one `sketch` fence. Its four headers
appear exactly once in D, O, C, H order:

```sketch
# D Declarations
tensor input shape=[N] dtype=fp32 layout=contiguous memory=global

# O Operations
load value <- input[index]

# C Control
parallel index over N
guard index < N

# H Target Hints
target=triton_mlu
num_warps=1
```

D statements start with `tensor`, `tile`, or `scalar`. O statements start with
`alloc`, `load`, `compute`, or `store`. C statements start with `parallel`,
`for`, `if`, `else`, `guard`, or `end`. H starts with exactly
`target=<target_profile>`. Every H directive occupies its own line and has one
`name=value` pair; whitespace-separated directives on one line are invalid.

## Complete Kernel Example

The referenced implementation, report, and fingerprint document must be
materialized in the project before use.

````markdown
# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"mlu","target_profile":"triton_mlu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel"}
```

## Optimization Intent

```json
{"bottleneck_class":"device-bound","intervention":"fuse the routing reduction into the target kernel","allowed_changes":["kernel dataflow"],"invariants":["ModelNew public contract","output dtype and shape"],"expected_wall_improvement_pct":8.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor scores shape=[T,E] dtype=fp32 layout=row_major memory=global
tile row shape=[BLOCK_E] dtype=fp32 memory=register

# O Operations
load row <- scores[token,0:E]
compute probs = softmax(row)
store output[token,0:K] <- topk(probs,K)

# C Control
parallel token over T
guard token < T

# H Target Hints
target=triton_mlu
num_warps=1
num_stages=2
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the routing reduction into the target kernel","expected_causal_chain":["external routing kernels disappear","device time decreases","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"external_kernel_count_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output dtype and shape unchanged"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`; no matching failure invalidates this path.

## Rationale and Evidence

The accepted trace contains separate routing kernels that are inside the candidate's change boundary.
````

## Complete Host-only Example

````markdown
# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"002","reference_implementation":"triton_example_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"mlu","target_profile":"triton_mlu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"host"}
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
````

## Complete Abort Example

````markdown
# Decision 004

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"004","reference_implementation":"triton_example_003.py","reference_report":"rounds/report_003.md","language":"triton","backend":"mlu","target_profile":"triton_mlu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none"}
```

## Optimization Intent

```json
{"bottleneck_class":"none","intervention":"no stable intervention clears the adoption threshold","allowed_changes":[],"invariants":["ModelNew public contract","benchmark semantics"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md`; all remaining paths repeat measured failures or lack a falsifiable five-percent hypothesis.

## Rationale and Evidence

The accepted report and completed round evidence do not justify another stable improvement attempt.
````
