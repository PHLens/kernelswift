# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: `1`
- last_completed_round: `001`
- accepted_kernel: `baseline_adapter.py`
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `Round 000 baseline remains canonical; Round 001 candidate-ready after real AST-loader, Triton-MACA compile, execution, and fixed-seed smoke correctness.`
- open_hypotheses: `Verifier must establish targeted group/expert tie ID parity, authoritative correctness, paired wall timing, and targeted profiler observables before any adoption decision.`
- artifact_read_hashes: `Round 001 decision, canonical source, candidate, project, team-state, profile, and baseline report hashes are recorded below.`

## Current Bottleneck

- Round 000 reports `baseline_adapter.py` at 0.231739 ms median wall time,
  147.7526708984375 device us/call, and 15.0 kernels/call. Four
  gatherTopK/bitonicSort launches account for 89.6741943359375 us/call.

## Recent Three-round Evidence

- Round 000, baseline, `rounds/report_000.md`, canonical PyTorch adapter.
- Round 001, candidate-ready, `rounds/coder_result_001.md`, kernel-fusion;
  exact-regime 1/1 smoke passed but supplies no authoritative performance
  evidence and does not advance the canonical pointer.

## Open Hypotheses or Checks

- Verify exact IDs for fixed seeded input and targeted equal-logit cases at both
  the group cutoff and expert cutoff.
- Run authoritative Level 0 correctness and paired benchmark against
  `baseline_adapter.py`, then targeted Level 1 kernel-count/device-time
  evidence if correctness passes.
- Confirm one fused Triton kernel per call and disappearance of baseline
  gatherTopK/bitonicSort kernels. The 1/1 smoke timing is excluded from adoption.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `skills/kernel-opt-loop/prompts/coder.md` | `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196` | 001 |
| `skills/kernel-opt-loop/prompts/coder_targets/triton_maca.md` | `2cfa08c2664f01e70bb43eec7bb998be836a6a719b17535268a8d6ca18c85540` | 001 |
| `project.md` | `6721db4a009b0a539ab70040ab86151ed0cea8990d6d88236ef07abeca0506d3` | 001 |
| `team-state.md` | `760d627f2bbd6869430c8d7561a67faf096f5d41ec42ab78b3df1da78cbe5c77` | 001 |
| `rounds/report_000.md` | `9b8374ee96d72fa8eed02415440eb778867d9ee0f3d0e8914608695a0c299f00` | 001 |
| `rounds/decision_001.md` | `c71c970e3bcf6d7984272611627d711ce64b6f3c18d1a057b2aab440c50c173f` | 001 |
| `baseline_adapter.py` | `d92a1be10bf9ad036287ed0f769b195262132893fa78d88fa30f992fbe757827` | 001 |
| `triton_grouped_topk_001.py` | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | 001 |
