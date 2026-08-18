# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: `4`
- last_completed_round: `004`
- accepted_kernel: `triton_grouped_topk_001.py`
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `Round 002 allocation coalescing and Round 003 combined reduction were no-improvement; Round 004 host dispatch specialization is candidate-ready after source gates and the true compile/correctness smoke.`
- open_hypotheses: `Verifier must establish newly admitted hidden-metadata fast-path parity, retained fallback behavior, authoritative correctness/wall timing, and the contracted host/device observables.`
- artifact_read_hashes: `Round 004 decision, accepted reports/canonical/reference adapter, candidate, project, team-state, contexts, and profile hashes are recorded below.`

## Current Bottleneck

- Round 001 remains canonical at 68.280 us/call wall, one
  10.7442822265625 us device kernel per call, and a 41.58952 us/call inclusive
  CPU scope. Round 004 changes only the fixed-path Python dispatch metadata
  work; kernel, launch, and the two independent allocations remain frozen.

## Recent Three-round Evidence

- Round 002, no-improvement, `rounds/report_002.md`,
  fresh-allocation-coalescing; formal wall regressed 13.711567434852972%.
- Round 003, no-improvement, `rounds/report_003.md`,
  value-index-reduction-fusion; formal wall improved only
  0.04903708987159917%.
- Round 004, candidate-ready, `rounds/coder_result_004.md`,
  fast-path-dispatch-specialization; exact source gates and the unique remote
  true-loader compile/correctness smoke passed.

## Open Hypotheses or Checks

- Prove newly admitted hidden tensors varying nonleading width, dtype,
  contiguity, and device take the fast path and match base values/exact IDs.
- Prove token assertion, retained gating/config/grad guards, and byte-equivalent
  fallback behavior, including sigmoid, unsupported scoring, and grad cases.
- Run authoritative correctness and paired wall timing against the accepted
  class-rename reference. Audit the contracted source/CPU/device/launch/allocation
  observables; exclude 1/1 smoke timing from adoption.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `skills/kernel-opt-loop/prompts/coder.md` | `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196` | 004 |
| `skills/kernel-opt-loop/adapters/codex.md` | `b77b99e78bbe9cb379ce71deda1b0879bb6c9bd5bc27233e1d53fdd9e74ff151` | 004 |
| `skills/kernel-opt-loop/references/invariants.md` | `22b53f5f900c8062c445f35be52414b4abba99f8e4893a4dfab996eb1cd8d29c` | 004 |
| `skills/kernel-opt-loop/prompts/coder_targets/triton_maca.md` | `2cfa08c2664f01e70bb43eec7bb998be836a6a719b17535268a8d6ca18c85540` | 004 |
| `project.md` | `5b97cdfd38c52600dee404fc1319befdc6790973324200345c7e16382af24651` | 004 |
| `team-state.md` | `8a8ad250d43175ed24b276646f3f9948f50e9c36272f6414836f6c8f36a289a9` | 004 |
| `rounds/report_001.md` | `f2866692d8d1c4519e9a2028c7b1d707fcb4f9f945fd856f78139f2dbe2aec4a` | 004 |
| `rounds/report_002.md` | `a5ad9cfe8ead4e1e3cf06ef990ea0817537af4c088219f1eed9a551055426365` | 004 |
| `rounds/report_003.md` | `6c645acf858745585d4f668546609dc9d3dbc3f7c1b8110a013193f6c89c2fdd` | 004 |
| `rounds/decision_004.md` | `5aab9a8df7ab5664cf6a2374d945b06a3a76d60a5fff9e4134a229a75bf0f587` | 004 |
| `triton_grouped_topk_001.py` | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | 004 |
| `reference_triton_grouped_topk_001.py` | `70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9` | 004 |
| `triton_grouped_topk_004.py` | `50da7d1b6483822273b0dda404b33fb79fd96fbd95e3f0a9a48d8c8a3e315683` | 004 |
