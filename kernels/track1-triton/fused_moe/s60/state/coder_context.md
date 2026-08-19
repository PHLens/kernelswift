# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: 1
- last_completed_round: `000`
- accepted_kernel: `baseline_adapter.py`
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `Phase 0 baseline established only`
- open_hypotheses: `<to-fill in Round 001>`
- artifact_read_hashes: `<to-fill>`

## Current Bottleneck

- Eager baseline 147 launches/call; fusion target is the per-token/per-expert
  FFN + routing. State-dict keys must be exactly `w1`/`w2` to survive
  `load_state_dict`.

## Recent Three-round Evidence

- `000`, baseline, `rounds/report_000.md`, not-applicable (Phase 0)

## Open Hypotheses or Checks

- `<to-fill in Round 001>`

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` | `21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d` | `000` |
| `baseline_adapter.py` | `b939d91f0f85e299a1102bfceb00da0e38c484a81c8d23ec78777fce68a3ee6f` | `000` |
