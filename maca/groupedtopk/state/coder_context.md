# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: `3`
- last_completed_round: `003`
- accepted_kernel: `triton_grouped_topk_001.py`
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `Round 001 accepted kernel fusion; Round 002 allocation coalescing was no-improvement; Round 003 is candidate-ready after eight combined value/index reductions passed the true compile/correctness smoke.`
- open_hypotheses: `Verifier must establish targeted tie-ID parity, authoritative correctness and wall timing, then reduction count/device time if the wall gate passes.`
- artifact_read_hashes: `Round 003 decision, accepted reports/canonical/reference adapter, candidate, project, team-state, and profile hashes are recorded below.`

## Current Bottleneck

- Round 001 authoritatively reports the accepted candidate at 68.280 us/call,
  one 10.7442822265625 us device kernel per call, and a 41.58952 us/call
  inclusive CPU scope. The accepted kernel performs eight expert argmax
  reductions plus eight full-width selected-value sum reductions.

## Recent Three-round Evidence

- Round 001, accepted, `rounds/report_001.md`, kernel-fusion; correctness,
  targeted tie parity, 1.0 kernel/call, and 69.59021613749428% authoritative
  wall improvement passed.
- Round 002, no-improvement, `rounds/report_002.md`,
  fresh-allocation-coalescing; capability/safety passed but formal wall
  regressed 13.711567434852972%, so canonical remained Round 001.
- Round 003, candidate-ready, `rounds/coder_result_003.md`,
  value-index-reduction-fusion; pinned MACA combined reduction
  compile/execution and fixed-seed smoke correctness passed.

## Open Hypotheses or Checks

- Verify exact IDs and allowed weights on targeted group-cutoff and expert-cutoff
  ties while proving the fixed fast path executes.
- Verify exactly eight combined expert value/index reductions, zero expert
  selected-value `tl.sum` reductions, one kernel/launch per call, and unchanged
  host allocations/guard/fallback.
- Run authoritative correctness and paired wall timing against the accepted
  class-rename reference, then targeted device profiling only if required by the
  contract. Exclude 1/1 smoke timing from adoption.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `skills/kernel-opt-loop/prompts/coder.md` | `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196` | 003 |
| `skills/kernel-opt-loop/prompts/coder_targets/triton_maca.md` | `2cfa08c2664f01e70bb43eec7bb998be836a6a719b17535268a8d6ca18c85540` | 003 |
| `project.md` | `41f73ad526412fe37a41116701a3257cb7f90bffbae88611f69a99a4e2bb7750` | 003 |
| `team-state.md` | `35b2f3bec6a069e2ea3364c64eebe1f3b38653bd7f38f950a7ba2ba59f07c52b` | 003 |
| `rounds/report_001.md` | `f2866692d8d1c4519e9a2028c7b1d707fcb4f9f945fd856f78139f2dbe2aec4a` | 003 |
| `rounds/report_002.md` | `a5ad9cfe8ead4e1e3cf06ef990ea0817537af4c088219f1eed9a551055426365` | 003 |
| `rounds/decision_003.md` | `cfcee8a61b91536da0aa302504b8bc4119c9c2deac5150878b6371870791f6b7` | 003 |
| `triton_grouped_topk_001.py` | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | 003 |
| `reference_triton_grouped_topk_001.py` | `70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9` | 003 |
| `triton_grouped_topk_003.py` | `9409bd85da798b083e785774525a076ec781b6df13cd1129843fe7e9c9ead9f6` | 003 |
