# Decision 005

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"005","reference_implementation":"triton_grouped_topk_004.py","reference_report":"rounds/report_004.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-falsifiable-intervention"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"abort because no semantics-preserving change supported by the remaining evidence can credibly improve the accepted 0.432098 ms wall median by at least five percent","allowed_changes":[],"invariants":["accepted triton_grouped_topk_004.py remains canonical","exact torch.topk group and final expert ordering including active-set-dependent ties","current caller device and stream behavior","per-forward buffer ownership with no cross-call aliasing","non-target fallback semantics","public constructor and forward contract","immutable base.py and unchanged harness","measurement fingerprint unchanged"],"expected_wall_improvement_pct":0.0}
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

- The accepted Round 004 candidate must retain exact library `torch.topk` group and final selection. Round 002 proved that a custom final selector fails active-set-dependent BI150/PyTorch tie ordering.
- The rejected one-stage mask dependency from Round 003 must not be repeated. The accepted two-stage arrangement is required because the mask depends on library-produced `group_idx`.
- The remaining `at::native::direct_copy int32` is `8.76669921875 us/call`, while five percent of the accepted `0.432098 ms` wall median is `21.6049 us`. Even eliminating the observed copy entirely cannot support the required wall-time claim.
- The profile directly proves only narrow integer behavior. The accepted four-element int32 group-index transfer is runtime-conformant evidence for Round 004, but it does not establish a new semantics-preserving zero-copy, int64-consumption, or alternate index-transfer mechanism.
- The larger remaining kernels are exact library `gatherTopK` at `48.852978515625 us/call` and `bitonicSortKVInPlace` at `36.45123046875 us/call`; changing them would repeat the prohibited custom-selection path or require evidence for a new exact compatible selection mechanism.
- No `tl.dot`, launch hints, fast launcher, block pointers, mixed precision, host cache, context change, or stream change is established as a valid replacement mechanism.

## Rationale and Evidence

`rounds/report_004.md` establishes `triton_grouped_topk_004.py` as canonical after an unrounded paired wall median improvement of `7.455430192%`: `0.432098 ms` versus `0.466908 ms` for the accepted-reference adapter. It also records `127.260771484375 us/call` device time and `9.9` kernels/call, down from `178.991259765625 us/call` and `14.86` kernels/call.

The remaining device profile is dominated by required exact library final selection. The only independently named non-library overhead close to a removable mechanism is the int32 direct copy at `8.76669921875 us/call`. That upper bound is materially below the `21.6049 us` equivalent of the mandatory five-percent wall improvement, before accounting for the irreducible fixed harness and host portions of the current wall time. It therefore cannot justify another >=5% hypothesis.

A different final-selection implementation is disallowed by verified tie behavior, and a different index-transfer path would require unproven integer semantics. No stable, falsifiable intervention clears the adoption threshold under the current canonical evidence. Classification: `aborted`; do not dispatch Coder or Verifier. Reconsider only after a matched BI150 probe or Verifier evidence establishes a semantics-preserving reduction in the exact library selection or a new independently measurable mechanism capable of at least five-percent wall improvement.
