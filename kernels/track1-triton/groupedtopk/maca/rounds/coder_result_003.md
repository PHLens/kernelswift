# Coder Result 003

Result: `candidate-ready`

- reason_code: `all-required-coder-gates-passed`
- round: `003`
- source_canonical: `triton_grouped_topk_001.py`
- source_canonical_sha256: `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384`
- source_report: `rounds/report_001.md`
- source_report_sha256: `f2866692d8d1c4519e9a2028c7b1d707fcb4f9f945fd856f78139f2dbe2aec4a`
- reference_adapter: `reference_triton_grouped_topk_001.py`
- reference_adapter_sha256: `70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9`
- decision: `rounds/decision_003.md`
- decision_sha256: `cfcee8a61b91536da0aa302504b8bc4119c9c2deac5150878b6371870791f6b7`
- candidate: `triton_grouped_topk_003.py`
- candidate_sha256: `9409bd85da798b083e785774525a076ec781b6df13cd1129843fe7e9c9ead9f6`
- selected_profile: `triton_maca`
- runtime_fingerprint: `project.md#runtime-fingerprint`
- measurement_fingerprint: `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`

## Gate Summary

| Gate | Return code | Evidence |
|---|---:|---|
| Decision validation | 0 | `validate_decision.py --expected-profile triton_maca rounds/decision_003.md` returned `"valid": true`. |
| Final `ast.parse` | 0 | Parsed the complete 292-line candidate at the recorded SHA. |
| Exact source delta | 0 | Candidate bytes equal canonical Round 001 after exactly eight authorized expert argmax/value-sum pair replacements. |
| Frozen-region equivalence | 0 | Group path, host two-output allocations, guard, fallback, launch/grid/arguments/T/BLOCK_E/num_warps, constructor/signature, and public entrypoints are byte-equivalent to Round 001. |
| Reduction counts | 0 | Expert `tl.argmax=0`, expert `tl.sum=0`, combined `tl.max(... return_indices=True, return_indices_tie_break_left=True)=8`; four group `tl.argmax` calls remain. |
| Actual remote loader / combined-reduction compile-smoke | 0 | Remote candidate SHA matched; true loader, pinned MACA frontend/backend lowering, unchanged launch, execution, and fixed-seed correctness passed. |

The unique remote 1/1 smoke reported
`PASS accuracy; v0=0.116930 ms, v1=0.089022 ms, speedup=1.313x` and
`1 passed, 0 failed`. These values are smoke timing only, not authoritative
performance evidence or an adoption claim.

## Primitive and Hint Conformance

- Each unrolled expert rank uses the decision-authorized standard spelling:
  `expert_logit_i, expert_id_i = tl.max(expert_remaining_i, axis=0,
  return_indices=True, return_indices_tie_break_left=True)`.
- The matched target profile did not previously prove 256-lane max-with-index.
  This was the explicit capability gate; the exact spelling compiled and
  executed through the real pinned Triton-MACA harness without substitution.
- Eight separate 256-lane selected-value `tl.sum(tl.where(...))` reductions
  and eight expert `tl.argmax` reductions are absent. Eight combined
  value/index reductions remain, while the Round 001 group selection and final
  eight-scalar normalization are unchanged.
- Explicit left tie breaking is present in all eight combined reductions. The
  fixed-seed harness passed; targeted group-cutoff and expert-cutoff tie parity
  remains authoritative Verifier work.
- The candidate starts from accepted Round 001, not rejected Round 002. It
  retains two independent fresh `torch.empty` outputs and introduces no shared
  backing, dtype view, cache, pool, reuse, mutable state, synchronization,
  device switch, stream switch, or launcher change.
- Direct launch remains grid `(83,)`, `T=83`, `BLOCK_E=256`, and
  `num_warps=1`. No unsupported scheduling or memory primitive was added.

## Attempt Ledger

| Attempt | Command | Exit status | Defect / evidence | Candidate before SHA256 | Candidate after SHA256 |
|---:|---|---:|---|---|---|
| 1 | `validate_decision.py --expected-profile triton_maca rounds/decision_003.md` | 0 | None; immutable decision valid. | not-applicable | not-applicable |
| 2 | Scoped canonical-copy patch with eight expert-pair replacements | 0 | Candidate materialized without defect. | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | `9409bd85da798b083e785774525a076ec781b6df13cd1129843fe7e9c9ead9f6` |
| 3 | Final `ast.parse`, exact replacement, frozen-region, and reduction-count gates | 0 | None. | `9409bd85da798b083e785774525a076ec781b6df13cd1129843fe7e9c9ead9f6` | same |
| 4 | Local actual `auto_bench.load_ks_module(candidate)` | 1 | Local WSL Python lacked `torch`; failure occurred importing `auto_bench.py` before candidate loading. Environment note, not candidate defect. | `9409bd85da798b083e785774525a076ec781b6df13cd1129843fe7e9c9ead9f6` | same |
| 5 | Actual loader AST-filter precheck with minimal local import stubs | 0 | AST-filter/compile/exec precheck only; not a substitute for the true runtime loader. | `9409bd85da798b083e785774525a076ec781b6df13cd1129843fe7e9c9ead9f6` | same |
| 6 | Orchestrator-assisted remote SHA check and unique `auto_bench.py --warmup 1 --repeat 1 --full-traceback` smoke | 0 | True loader, combined max-with-index/left-tie compile, unchanged launch, execution, and fixed-seed correctness passed. | `9409bd85da798b083e785774525a076ec781b6df13cd1129843fe7e9c9ead9f6` | same |

No repair attempt was required. The candidate was not uploaded directly by the
Coder; Orchestrator performed the authorized sync and unique remote gate.
