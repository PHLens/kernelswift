# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: `2`
- last_completed_round: `004`
- accepted_kernel: `triton_grouped_topk_003.py`
- accepted_report: `rounds/report_003.md`
- recent_three_round_evidence: `Round 004 valid no-improvement: candidate-owned stream snapshot reduced one controllable lookup but wall median improvement was 2.058982586436897%; canonical remains Round 003.`
- open_hypotheses: `Two valid no-improvement rounds remain before the configured stop threshold. Any next host hypothesis must preserve both caches and output lifetime guardrails.`
- artifact_read_hashes: `decision_004.md, candidate source, reference adapter, coder_result_004.md, project.md, and team-state.md recorded in the Round 004 ledger.`

## Current Bottleneck

- The accepted candidate has safe output-buffer reuse, exact-key metadata cache,
  and one direct Triton-GCU launch per call. GCU device duration remains
  unavailable. The candidate-owned stream lookup optimization did not clear 5%.

## Recent Three-round Evidence

- Round 002, accepted, `rounds/report_002.md`: `9.02136875254568%` wall improvement.
- Round 003, accepted, `rounds/report_003.md`: `6.464721724746064%` wall improvement.
- Round 004, no-improvement, `rounds/report_004.md`: `2.058982586436897%` wall improvement; correctness and guardrails PASS.

## Open Hypotheses or Checks

- Continue only with a bounded host hypothesis or a matched GCU microbenchmark.
- Do not change canonical after a valid below-threshold round.
- Kernel dataflow changes require attributable device evidence unavailable in the
  current GCU exporter.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `rounds/decision_004.md` | `a126c9abc86da11734be828bc6c5900e0b1107ba07ecbfa079fc4f74d1416713` | 004 |
| `reference_triton_grouped_topk_003.py` | `9977aaf9ec96c851be33f2582e6284451fd41686a1acc4607deb4e104dca5ea7` | 004 |
| `triton_grouped_topk_004.py` | `ea9be7896731f7f371f9ba087c8d01daca6556c66a3e50b2c6146fe6de118bb0` | 004 |
| `rounds/coder_result_004.md` | 5ded926 | 004 |
