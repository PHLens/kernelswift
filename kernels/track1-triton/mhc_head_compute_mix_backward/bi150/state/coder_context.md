# Coder Context

## Contract

- contract_hash: `not-recorded` (coder.md prompt read and followed)
- last_completed_round: `001`

## Selected Profile / Fingerprint Facts

- target_profile: `triton_cuda`
- runtime: torch 2.7.1, triton 3.1.0 (CoreX), Iluvatar BI-V150, capability (7,1)
- measurement_fingerprint: `a03823074048c8cb5e8199b593c8c19aa3b259180969321015e5a1679461b71a`

## Artifact Read Hashes

- base.py: `28d4d213638e5adc7f6bc52928c8f7596cd8446e3e0201beab364b5c3c7988fc`
- baseline_adapter.py: `98cf1e1e5f493ae4ae08c9391f00db83b3cd783a1b194d9c0d63851091133a5d`
- decision_001.md: `dc0a4837cc8a5aeb867e9d71f8c1e4bc1930ee57d431a279f761329271e5371a`
- candidate (round 001): `5d419f5d2e920abf3cf583a22f155e76047f9e5bc3a5cc36baca5477fae94349`

## Open Local Checks

- None. Round 001 candidate written, passed py_compile + harness smoke
  (`PASS accuracy`, speedup 1.752x indicative), result `candidate-ready`.

## Notes

- Single-kernel fusion confirmed viable on BI150: `tl.sigmoid`, `tl.sum(axis=0)`
  over `(BLOCK,4)`, full scalar `tl.sum`, and `tl.atomic_add` all lower and match
  reference (local probe err < 1e-5). No fallback to two-program needed.
- Candidate result artifact: `rounds/coder_result_001.md`.
