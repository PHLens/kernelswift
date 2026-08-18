# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: 2
- last_completed_round: `001`
- accepted_kernel: null
- accepted_report: null
- recent_three_round_evidence: `<round 001 candidate-ready, change family triton-attention-rewrite>`
- open_hypotheses: `<H-001 pending Verifier measurement>`
- artifact_read_hashes: `<see table below>`

## Current Bottleneck

- `<Verifier-backed fact only — not yet available>`

## Recent Three-round Evidence

- Round 001: `candidate-ready`; single Triton attention kernel replaces
  `F.scaled_dot_product_attention` + view/transpose/reshape; candidate
  `triton_attn_001.py` (SHA `61eeb336...`). Diagnostic timing ~1.039x; awaits
  Verifier authoritative measurement.

## Open Hypotheses or Checks

- `<H-001: Triton attention rewrite eliminates 4 layout kernels and reduces device/wall time — pending Verifier>`

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | 000 |
| `baseline_adapter.py` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | 000 |
| `triton_attn_001.py` | `61eeb3367619684e6f61ea3a908c1fc78a575834b4a84c032748277d0e76be74` | 001 |
| `rounds/decision_001.md` | `fa6ffd3d2a08dd78d2f3ad958890d0419a0115b898c68b6bbf4ef88105d43eca` | 001 |
| `project.md` | `<orchestrator-owned>` | 000 |
