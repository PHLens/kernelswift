# Decision 006

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"006","reference_implementation":"triton_grouped_topk_004.py","reference_report":"rounds/report_004.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-falsifiable-intervention"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"abort because Round 004 remains the accepted candidate and current evidence does not support another semantics-preserving >=5% wall-time improvement","allowed_changes":[],"invariants":["accepted triton_grouped_topk_004.py remains canonical","exact torch.topk group and final expert ordering including active-set-dependent ties","current caller device and stream behavior","per-forward buffer ownership with no cross-call aliasing","non-target fallback semantics","public constructor and forward contract","immutable base.py and unchanged harness","measurement fingerprint unchanged"],"expected_wall_improvement_pct":0.0}
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

- Round 004's exact library `torch.topk` boundaries are required for BI150/PyTorch active-set-dependent ties; Round 002 custom selection cannot replace them.
- Round 005 established that the named int32 copy is `8.76669921875 us/call`, below the `21.6049 us` wall-time equivalent of the required five-percent improvement from the accepted `0.432098 ms` median.
- The larger exact `gatherTopK` and `bitonicSortKVInPlace` kernels remain the dominant device contributors; changing them without a proven compatible selector would repeat the rejected design family.

## Rationale and Evidence

`rounds/report_004.md` establishes `triton_grouped_topk_004.py` as canonical at
`0.432098 ms` median, `127.260771484375 us/call` device time, and `9.9`
kernels/call. Round 005's abort remains applicable: no independently measured
semantics-preserving mechanism currently clears the five-percent threshold.
The accepted two-stage routing fusion and its exact tie suite remain unchanged.

Classification: `aborted`; do not dispatch Coder or Verifier. Reconsider only
after matched BI150 evidence establishes a new exact-selection reduction or
another independent mechanism with a credible >=5% wall-time causal chain.
