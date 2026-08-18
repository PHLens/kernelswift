# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: 1
- last_completed_round: null
- accepted_kernel: null
- accepted_report: null
- recent_three_round_evidence: `<none: Phase 0 not yet complete>`
- open_hypotheses: `<pending Phase 0>`
- artifact_read_hashes: `<see table below>`

## Phase 0 Baseline Adapter Confirmation

- baseline_adapter.py is FAITHFUL:
  - exactly one `Model` -> `ModelNew` rename (`class Model` -> `class ModelNew`); no stray `Model` class remains.
  - helper functions `random_rotation_matrices`, `rot_vec_mul`, `centre_random_augmentation` preserved as top-level defs.
  - `get_inputs` / `get_init_inputs` preserved.
  - `forward` returns `Tensor[4,256,3]` fp32 (n_sample=4, N_ATOM=256, 3 coords).
- no candidate kernel in Phase 0.

## Current Bottleneck

- `<Verifier-backed fact only — not yet available>`

## Recent Three-round Evidence

- `<none>`

## Open Hypotheses or Checks

- `<Phase 0 baseline adapter confirmation complete>`

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` | `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553` | 000 |
| `baseline_adapter.py` | `7d4a79ae96328fc03a4489710f68b7f639ddea9cbd5c0f7bb45e1cec5472061b` | 000 |
| `project.md` | `<orchestrator-owned>` | 000 |
