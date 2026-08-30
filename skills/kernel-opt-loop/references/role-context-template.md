# <Role> Context

> Naming contract: the durable role context file is exactly
> `state/designer_context.md`, `state/coder_context.md`, or
> `state/verifier_context.md` — one `*_context.md` per role. No `*_state.md`
> alias exists and no compatibility alias may be created.

- role_contract_sha256: `<sha256>`
- context_epoch: `<integer>`
- last_completed_round: `<NNN-or-null>`
- accepted_kernel: `<relative path-or-null>`
- accepted_report: `<relative path-or-null>`
- recent_three_round_evidence: `<bounded evidence summary>`
- open_hypotheses: `<bounded next work item summary>`
- artifact_read_hashes: `<artifact hash ledger summary>`

## Current Bottleneck

- `<Verifier-backed fact only>`

## Recent Three-round Evidence

- `<round, result, evidence pointer, and change family>`

## Open Hypotheses or Checks

- `<bounded next work item>`

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `<relative path>` | `<sha256>` | `<NNN>` |
