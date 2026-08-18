# Decision 007

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"007","reference_implementation":"triton_grouped_topk_004.py","reference_report":"rounds/report_004.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"persistent-no-falsifiable-intervention"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"abort because three consecutive design reviews retain the same evidence boundary: no semantics-preserving mechanism can credibly improve the accepted 0.432098 ms wall median by at least five percent","allowed_changes":[],"invariants":["accepted triton_grouped_topk_004.py remains canonical","exact torch.topk group and final expert ordering including active-set-dependent ties","current caller device and stream behavior","per-forward buffer ownership with no cross-call aliasing","non-target fallback semantics","public constructor and forward contract","immutable base.py and unchanged harness","measurement fingerprint unchanged"],"expected_wall_improvement_pct":0.0}
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

- Exact library `torch.topk` remains mandatory after the Round 002 structured
  group-tie failure; custom selection is not a legal remaining mechanism.
- The accepted two-stage fusion requires the post-group-topk mask stage; the
  Round 003 one-stage alternative is causally invalid.
- The only quantified non-selector candidate, an int32 direct copy at
  `8.76669921875 us/call`, remains less than the `21.6049 us` wall-time
  equivalent of the required five-percent improvement.
- The remaining dominant gather/sort kernels require new matched evidence for a
  semantics-preserving selector reduction. No such evidence exists.

## Rationale and Evidence

`rounds/report_004.md` is the latest accepted comparable evidence: it establishes
`triton_grouped_topk_004.py` at `0.432098 ms`, `127.260771484375 us/call`, and
`9.9 kernels/call`, with a `7.455430192%` improvement over the canonical
adapter. Rounds 005 and 006 independently found no qualifying next mechanism.
Round 007 confirms that the same condition persists without new target evidence
or an authorized policy change.

Classification: `aborted`; do not dispatch Coder or Verifier. The campaign
requires new matched BI150 evidence for exact selection, or a user-authorized
change to the frozen performance policy, before another candidate is justified.
