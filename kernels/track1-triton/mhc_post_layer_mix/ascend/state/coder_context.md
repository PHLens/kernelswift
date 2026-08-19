# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: 2
- last_completed_round: `002` (candidate-ready; Verifier verdict pending)
- accepted_kernel: `candidate_001.py` (last accepted; R002 candidate_002.py pending)
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `R001 candidate_001.py kernel-fusion 3.64x accepted; R002 candidate_002.py kernel-tuning (256/4->1280/2) ~0.5-1% faster, below 5% threshold`
- open_hypotheses: `R002 sweep shows BLOCK_C/num_warps yield ~1% only; remaining ~0.26ms is harness-fixed host/sync gap (device_ratio 0.70). Likely no-improvement.`
- artifact_read_hashes: `<see table below>`

## Current Bottleneck

- `<Verifier-backed fact only — not yet available>`

## Recent Three-round Evidence

- `<none>`

## Open Hypotheses or Checks

- Phase 0 baseline adapter confirmation: COMPLETE (see "Phase 0 Baseline Adapter Confirmation" below).
- No candidate kernel produced in Phase 0 (per contract).

## Phase 0 Baseline Adapter Confirmation

Faithfulness of `baseline_adapter.py` vs `base.py`:

- Class rename: exactly one top-level `Model` -> `ModelNew`. PASS.
- `get_inputs` / `get_init_inputs`: both preserved verbatim (signatures + return values). PASS.
  - `get_init_inputs()` returns `[]` in both. PASS.
- No stray top-level `Model` class in adapter (only `ModelNew`). PASS.
- `forward` returns `(x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()`
  with `term2 = einsum('abmn,abmc->abnc', comb_res_mix, residual.float())`.
  Output `Tensor[2,4096,4,1280]` bf16, matches project.md Semantics. PASS.
- Module constants `n0=2, n1=4096, h=1280, mhc_mult=4` preserved. PASS.
- `generate_mhc_post_test_data`: logic identical; only `device="cuda"` -> `device='cuda'`
  (quote-style change, cosmetic). PASS.
- No `__main__` block in either file. CONFIRMED ABSENT.

### Resolved defect (fixed via adapter regeneration — Coder did NOT edit the adapter)

1. `__init__` stale reference: previously `baseline_adapter.py` line 7 read
   `super(Model, self).__init__()`. Root cause was `make_baseline_adapter.py`
   only renaming `ClassDef.name` while leaving the old class name in the
   `super(Model, self)` call inside the class body (and the worktree held an
   outdated script). Orchestrator fixed the generator and regenerated the adapter:
   line 7 now reads `super(ModelNew, self).__init__()`. RESOLVED.

Note: the harness AST loader rewrites device strings ("cuda" -> "npu"); the adapter's
single-quoted `'cuda'` string is still a plain string literal and is expected to be
rewritten identically. Flag for Verifier awareness only.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---:|---:|
| `../base.py` | `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3` | 001 |
| `baseline_adapter.py` | `a4f0aa8ac2d59c57059223b1710d20718af1b0f892cd7c373174e531c927133e` | 001 |
| `baseline_adapter.py` (previous, superseded) | `9c7e660ea393d0e28b75126524ab7baac623d6b520f098991ede2fdff07b6ae3` | 000 |
| `rounds/decision_001.md` | `6c9bf2b10c30b3a1205fe3c94f3ba4b6dc8abe1f2753b41536bdf7e29acb32ad` | 001 |
| `candidate_001.py` | `b74e407348d424c9265ddf831b245cda90297a48bdbaa576fa7e6b57b5d121f9` | 002 |
| `rounds/decision_002.md` | `0539d245c659369917660581165e8a332e00a65ca9d56128f7a0fe4fbf4d2a21` | 002 |
| `candidate_002.py` | `6a66f302b3cbf2316b99c9d207e32161cb2bc05e4ea327279ce7be3d8955357c` | 002 |
| `project.md` | `<orchestrator-owned>` | 000 |
