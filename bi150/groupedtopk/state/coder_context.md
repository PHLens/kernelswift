# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: `2`
- last_completed_round: `001`
- accepted_kernel: `baseline_adapter.py`
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `Round 001 was capability-miss under the earlier profile. Round 002 direct fixed-shape Triton candidate compiled and passed seeded harness smoke, but failed exact PyTorch IDs on a structured group-tie case after the bounded repair budget; canonical baseline remains unchanged.`
- open_hypotheses: `A new design is required to reproduce BI150/PyTorch active-set-dependent top-k tie ordering. Do not reuse triton_grouped_topk_002.py as a source baseline or claim Verifier timing.`
- artifact_read_hashes: `base.py, baseline_adapter.py, project.md, team-state.md, decision_002.md, triton_cuda.md, invariants.md, and coder.md read for Round 002.`

## Current Bottleneck

- `Verifier-backed baseline profiler reports top-k gather at 48.7290625 us/call and bitonic sort at 36.879697265625 us/call; the failed candidate cannot be compared because exact integer correctness is not established.`

## Recent Three-round Evidence

- Round 000, baseline, `rounds/report_000.md`: baseline adapter is canonical; reference wall median `0.474612 ms`, adapter wall median `0.474995 ms`.
- Round 001, `rounds/decision_001.md`: capability-miss under the earlier profile; no candidate was created.
- Round 002, `rounds/coder_result_002.md`: candidate compiled and passed seeded smoke but failed structured group-tie correctness after one tie-order repair.

## Open Hypotheses or Checks

- Do not route `triton_grouped_topk_002.py` to Verifier; it is not correctness-eligible.
- Preserve `baseline_adapter.py` as canonical.
- Any future candidate must either reproduce the active-set-dependent PyTorch tie ordering or explicitly revise the decision with a validated semantics-preserving mechanism.
- Do not use the failed candidate as the source baseline.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `base.py` | `d57ace7d9196e2e44bdcfd17d1738482e7fd1bbb2d86fc6c9449c43938953eb5` | 000 |
| `baseline_adapter.py` | `689d458c7abe07323508fc054bfef609dc4bd1cd9c94e3bb706d6f2d2cd00016` | 002 |
| `rounds/decision_002.md` | `d3c0f316945706acaad5c6f68ae0d93e9bbf3c848ca735b20c2356b304107d37` | 002 |
| `rounds/coder_result_002.md` | `0da68154d0713fc2944d41336750028cecc335fcd00fe9e23293d9a548a877ae` | 002 |
