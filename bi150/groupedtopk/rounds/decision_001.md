# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"capability-miss"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"abort because a Triton grouped-top-k replacement would normatively require unproven reductions, masked/indexed selection, and row-layout semantics on triton_cuda","allowed_changes":[],"invariants":["ModelNew public constructor and forward contract","grouped top-k semantics","PyTorch top-k ordering and tie behavior","output shapes dtypes and device","immutable base.py","unchanged harness and measurement fingerprint"],"expected_wall_improvement_pct":0.0}
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

- `triton_cuda.md` establishes only one-dimensional contiguous fp32 `tl.load`/`tl.store`, `tl.arange` at extent 256, and `tl.program_id(axis=0)` behavior.
- The required grouped top-k dataflow would normatively use `tl.max`, `tl.argmax`, `tl.sum`, `tl.where` or equivalent masking/indexing, and fp32 row-layout semantics beyond the probe. Each is Unknown or Constrained for this use, and the profile requires a capability-miss classification instead of assuming support.
- The MLU-specific historical failures are not direct BI150 CUDA evidence. They reinforce that hierarchical selection, sort networks, and dynamic compaction cannot be presumed beneficial or legal without matched target evidence.
- No host optimization is selected: the canonical baseline has device_ratio `0.3769941822`, and no Verifier-backed host decomposition identifies a separately observable >=5% host mechanism.

## Rationale and Evidence

`rounds/report_000.md` establishes `baseline_adapter.py` as canonical with a `0.474995 ms` wall median and `179.0703515625 us/call` scoped device time. The accepted-reference scope identifies `at::native::sbtopk::gatherTopK` at `48.7290625 us/call` and `at::native::bitonicSortKVInPlace` at `36.879697265625 us/call`, which makes selection fusion a potential future direction.

However, `baseline_adapter.py` requires softmax, eight group maxima over 32 experts, top-four group selection, group-mask expansion, top-eight selection from masked scores, optional sum-based renormalization, and fp32/int32 result conversion. The matched CUDA profile does not establish the reductions, masks, index semantics, layout regime, or tie behavior needed to normatively describe a replacement. A decision may not convert those Unknown capabilities into assumptions.

Classification: `capability-miss`. Do not dispatch Coder or Verifier for this decision. Reconsider only after matched BI150/CoreX probes establish every required reduction, masking/indexing, fp32 row-layout, and tie-ordering requirement; any such future intervention requires a new decision in an unused round.
