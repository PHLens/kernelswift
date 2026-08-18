# Coder Result 003

Result: design-revision-required(reason=major-deviation)

## Identity

- Round: `003`
- Decision: `rounds/decision_003.md`
- Decision SHA-256: `dfe241e2b7b6f2609a3d59185d2d067072b986e9316f0b1e857a4023d0ac5030`
- Canonical source: `baseline_adapter.py`
- Canonical source SHA-256: `689d458c7abe07323508fc054bfef609dc4bd1cd9c94e3bb706d6f2d2cd00016`
- Candidate: `null`
- Candidate SHA-256: `null`
- Language: `triton`
- Backend: `cuda`
- Target profile: `triton_cuda`
- Runtime fingerprint: `project.md#runtime-fingerprint`

## Result Classification

`design-revision-required(reason=major-deviation)`

The immutable H-003 sketch requires one direct Triton kernel to compute
softmax, group maxima, and final group masking while retaining the exact
library `torch.topk(group_scores, 4)` selection boundary. The mask is a
function of the library-produced `group_idx`, so it cannot be materialized in
the one preprocessing kernel before that host/library operation completes.

A conforming implementation needs either a second direct Triton mask kernel
launched after `torch.topk`, or a host `masked_fill`/equivalent after the
library selection. The former changes the approved one-kernel intervention and
the latter leaves group masking outside the stated fused kernel. Neither is a
non-semantic accommodation. No candidate was created or compiled.

## Profile Conformance Review

- The first preprocessing kernel is individually expressible using proven
  `triton_cuda` primitives.
- The required post-selection data dependency, not a missing target primitive,
  prevents conformance to the immutable design.
- No unsupported primitive, remote command, compile, benchmark, or profile
  change was attempted.

## Attempt Ledger

| Attempt | Candidate SHA-256 | Command | Defect | Result |
|---|---|---|---|---|
| 1 | `null` | Design-to-implementation dependency review | `masked_scores` requires `group_idx` returned by exact library `torch.topk`; the decision permits only one direct kernel before that boundary. | major deviation required |

## Stable Reason Code

`post-selection-mask-requires-second-stage`

## Handoff

Return to Designer. A future decision may explicitly authorize a two-stage
preprocess/mask arrangement or a host-side mask while retaining exact library
selection, with a complete mixed Host Plan and a falsifiable >=5% wall-time
mechanism. Keep `baseline_adapter.py` canonical.
