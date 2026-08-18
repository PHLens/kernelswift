# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: `3`
- last_completed_round: `002`
- accepted_kernel: `baseline_adapter.py`
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `Round 001 was a profile capability gate. Round 002 direct full selection kernel failed BI150/PyTorch active-set-dependent tie ordering. Round 003 was design-rejected before a candidate because exact library group topk produces group_idx only after the proposed one kernel would need it to materialize the mask.`
- open_hypotheses: `Do not implement H-003 as written. A future decision must explicitly authorize a post-group-topk mask stage or a host-side mask with a complete Host Plan and measurable >=5% mechanism.`
- artifact_read_hashes: `baseline_adapter.py, team-state.md, project.md, decision_003.md, coder_result_002.md, triton_grouped_topk_002.py, triton_cuda.md, invariants.md, and coder.md read for Round 003.`

## Current Bottleneck

- `The baseline's top-k gather and bitonic sort remain dominant. Round 002 excludes custom final selection without a compatible tie-order mechanism; Round 003 exposes the unavoidable group-selection-to-mask dependency.`

## Recent Three-round Evidence

- Round 001, `rounds/decision_001.md`: capability-miss under the earlier profile; no candidate existed.
- Round 002, `rounds/coder_result_002.md`: full direct selection kernel passed seeded smoke but failed structured group-tie integer IDs.
- Round 003, `rounds/coder_result_003.md`: no candidate; one-kernel preprocessing/masking decision requires `group_idx` from library topk after that kernel has finished.

## Open Hypotheses or Checks

- Preserve `baseline_adapter.py` as canonical.
- Do not reintroduce a custom group/final top-k selection network.
- A new decision may use an explicit two-stage dataflow only when it states allocation, device, stream, concurrency, and cache behavior and predicts >=5% wall improvement.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `baseline_adapter.py` | `689d458c7abe07323508fc054bfef609dc4bd1cd9c94e3bb706d6f2d2cd00016` | 003 |
| `rounds/decision_003.md` | `dfe241e2b7b6f2609a3d59185d2d067072b986e9316f0b1e857a4023d0ac5030` | 003 |
| `rounds/coder_result_003.md` | `6955ae27e32b4ad10dfe2e824c629b620ca493f88f52964076ee3e1321d9f437` | 003 |
