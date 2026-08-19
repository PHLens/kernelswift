# Verifier Context State

- role_contract_sha256: `<computed-at-first-use>`
- context_epoch: 1
- last_completed_round: `000`
- accepted_kernel: `baseline_adapter.py` (pending Orchestrator canonical pointer)
- accepted_report: `rounds/report_000.md` (pending Orchestrator canonical pointer)
- recent_three_round_evidence: `<Phase 0 only: baseline established>`
- open_hypotheses: `<none: Phase 0>`
- artifact_read_hashes: `<see table below>`

## Current Bottleneck

- Baseline (reference `base.py`) device time `148.0188 us/call`, `8.66` kernels/call.
  Dominant compute: FlashAttentionScore transpose + fused attention (~47 us/call),
  causal `aclnnTriu` (~26.2 us/call), FlashAttentionScore core (~24.9 us/call).
  `EVENT_WAIT_SQE` (~31.5 us/call) indicates host/launch synchronisation wait.

## Recent Three-round Evidence

- `<round 000, result baseline, report rounds/report_000.md, change family: baseline>`

## Open Hypotheses or Checks

- `<none: Phase 0>`

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` | `12f8a77b8f52b50d513800907b6b21ff9c98709647b306793241f6f8da3cb105` | 000 |
| `baseline_adapter.py` | `31c4e9acea7d94ddd97740dbd3d33e6b505cbc3a118ed891b28f9e1ac5c0696f` | 000 |
| `../../auto_bench.py` (harness) | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 000 |
