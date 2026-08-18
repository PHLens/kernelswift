# Coder Result 004

Result: `candidate-ready`

- reason_code: `all-required-coder-gates-passed`
- round: `004`
- source_canonical: `triton_grouped_topk_001.py`
- source_canonical_sha256: `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384`
- source_report: `rounds/report_001.md`
- source_report_sha256: `f2866692d8d1c4519e9a2028c7b1d707fcb4f9f945fd856f78139f2dbe2aec4a`
- reference_adapter: `reference_triton_grouped_topk_001.py`
- reference_adapter_sha256: `70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9`
- decision: `rounds/decision_004.md`
- decision_sha256: `5aab9a8df7ab5664cf6a2374d945b06a3a76d60a5fff9e4134a229a75bf0f587`
- candidate: `triton_grouped_topk_004.py`
- candidate_sha256: `50da7d1b6483822273b0dda404b33fb79fd96fbd95e3f0a9a48d8c8a3e315683`
- selected_profile: `triton_maca`
- runtime_fingerprint: `project.md#runtime-fingerprint`
- measurement_fingerprint: `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`

## Gate Summary

| Gate | Return code | Evidence |
|---|---:|---|
| Decision validation | 0 | `validate_decision.py --expected-profile triton_maca rounds/decision_004.md` returned `"valid": true`. |
| Final `ast.parse` | 0 | Parsed the complete frozen candidate at the recorded SHA. |
| Exact source delta | 0 | Candidate bytes equal canonical Round 001 after only the authorized forward guard/device substitutions. |
| Frozen-region equivalence | 0 | Complete kernel, launch/grid/arguments/T/BLOCK_E/num_warps, fallback, constructor/signature, grad predicate, and public entrypoints are byte-equivalent to Round 001. |
| Dispatch source observables | 0 | Shape tuple materializations are 0; hidden metadata eligibility queries are 0; `gating_output.device` property reads are 1; two independent `torch.empty` calls remain. |
| Reference class rename | 0 | Reference SHA matched and exact byte comparison proved only `ModelNew` to `Model` changed from canonical. |
| Local actual loader | 1 | Local WSL Python lacks `torch`; import failed in `auto_bench.py` before candidate loading. This is an environment note, not a candidate defect. |
| Actual loader AST-filter precheck | 0 | The real `auto_bench.load_ks_module` path executed with minimal import stubs and retained `ModelNew`, `get_inputs`, and `get_init_inputs`; this does not substitute for the true runtime gate. |
| Orchestrator-assisted remote true-loader smoke | 0 | Remote candidate SHA matched; pinned C500 loader, compile, launch, execution, and fixed-seed correctness passed. |

The unique remote 1/1 smoke reported
`PASS accuracy; v0=0.102861 ms, v1=0.085493 ms, speedup=1.203x` and
`1 passed, 0 failed`. These values are smoke timing only, not authoritative
performance evidence or an adoption claim.

## Host and Hint Conformance

- The first statement remains the exact token-count assertion. The sole new
  invocation local is `gating_device = gating_output.device`.
- The fixed guard directly tests `gating_output.shape == (83, 256)`; it
  retains gating dtype, contiguity, CUDA-device, all mutable constructor
  comparisons, and the exact conditional grad predicate.
- Hidden width, dtype, contiguity, and device-equality eligibility reads are
  absent. The conditional hidden `requires_grad` read and leading token
  assertion remain.
- Both Round 001 output allocations remain separate and fresh, and both use the
  invocation-local gating device. No shared backing, dtype view, cache, pool,
  reuse, aliasing, model/global mutable state, synchronization, device switch,
  or stream switch was introduced.
- The full Round 001 Triton kernel remains unchanged, including its separate
  expert argmax/value-sum reductions. The Round 002 and Round 003 changes are
  absent. Direct launch remains grid `(83,)`, `T=83`, `BLOCK_E=256`, and
  `num_warps=1`.
- Canonical fallback and public entrypoints are byte-equivalent. Newly admitted
  hidden-metadata parity, retained fallback cases, authoritative correctness,
  wall timing, and targeted profiling remain Verifier work.

## Attempt Ledger

| Attempt | Command | Exit status | Defect / evidence | Candidate before SHA256 | Candidate after SHA256 |
|---:|---|---:|---|---|---|
| 1 | `validate_decision.py --expected-profile triton_maca rounds/decision_004.md` | 0 | None; immutable decision valid and profile matched. | not-applicable | not-applicable |
| 2 | Scoped canonical-copy patch with only authorized forward substitutions | 0 | Candidate materialized without defect. | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | `50da7d1b6483822273b0dda404b33fb79fd96fbd95e3f0a9a48d8c8a3e315683` |
| 3 | Final `ast.parse`, exact replacement, frozen-region, and source-count gates | 0 | None. | `50da7d1b6483822273b0dda404b33fb79fd96fbd95e3f0a9a48d8c8a3e315683` | same |
| 4 | Local actual `auto_bench.load_ks_module(candidate)` | 1 | Local WSL Python lacked `torch`; failure occurred importing `auto_bench.py` before candidate loading. | `50da7d1b6483822273b0dda404b33fb79fd96fbd95e3f0a9a48d8c8a3e315683` | same |
| 5 | Actual loader AST-filter precheck with minimal local import stubs | 0 | AST-filter/compile/exec precheck only; not a substitute for the true runtime loader. | `50da7d1b6483822273b0dda404b33fb79fd96fbd95e3f0a9a48d8c8a3e315683` | same |
| 6 | Orchestrator-assisted remote SHA check and unique `auto_bench.py --warmup 1 --repeat 1 --full-traceback` smoke | 0 | True loader, pinned frontend/backend compile, unchanged launch, execution, and fixed-seed correctness passed. | `50da7d1b6483822273b0dda404b33fb79fd96fbd95e3f0a9a48d8c8a3e315683` | same |

No candidate repair attempt was required. The candidate was not uploaded
directly by the Coder; Orchestrator performed the authorized sync and unique
remote gate.
