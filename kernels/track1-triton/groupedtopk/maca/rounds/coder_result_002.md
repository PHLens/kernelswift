# Coder Result 002

Result: `candidate-ready`

- reason_code: `all-required-coder-gates-passed`
- round: `002`
- source_canonical: `triton_grouped_topk_001.py`
- source_canonical_sha256: `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384`
- source_report: `rounds/report_001.md`
- source_report_sha256: `f2866692d8d1c4519e9a2028c7b1d707fcb4f9f945fd856f78139f2dbe2aec4a`
- reference_adapter: `reference_triton_grouped_topk_001.py`
- reference_adapter_sha256: `70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9`
- decision: `rounds/decision_002.md`
- decision_sha256: `96b175002ab35ebbdeab2e647e1f0acfb150d08ca30792db1c6657a3afea7c55`
- candidate: `triton_grouped_topk_002.py`
- candidate_sha256: `1cbfddc1fd91ef4d73e388758467962cb471fc2a5f508c0af0749dcce53080d1`
- selected_profile: `triton_maca`
- runtime_fingerprint: `project.md#runtime-fingerprint`
- measurement_fingerprint: `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`

## Gate Summary

| Gate | Return code | Evidence |
|---|---:|---|
| Decision validation | 0 | `validate_decision.py --expected-profile triton_maca rounds/decision_002.md` returned `"valid": true`. |
| Final `ast.parse` | 0 | Parsed the complete 283-line candidate at the recorded SHA. |
| Canonical text equivalence | 0 | Candidate bytes equal canonical Round 001 after exactly one replacement of the two-output allocation block with the decision-specified backing/view block. |
| Kernel/launch/guard/fallback/entrypoint equivalence | 0 | Kernel definition, direct-launch block, full fast guard, fallback, constructor/signature, `get_inputs`, and `get_init_inputs` are byte-equivalent to Round 001. |
| Reference adapter identity | 0 | SHA matched and bytes equal canonical after exactly one `class ModelNew` to `class Model` replacement. |
| Actual remote loader / dtype-view / compile-smoke | 0 | Remote reference and candidate hashes matched; true loader, MACA dtype-view construction, unchanged launch, execution, and correctness passed. |

The unique remote 1/1 smoke reported
`PASS accuracy; v0=0.110046 ms, v1=0.098838 ms, speedup=1.113x` and
`1 passed, 0 failed`. These values are smoke timing only, not authoritative
performance evidence or an adoption claim.

## Host Plan and Capability Conformance

- The fixed path now makes exactly one local `torch.empty` call for a fresh
  flat 1328-element int32 backing on `gating_output.device`.
- `topk_weights` is
  `backing[:83 * 8].view(torch.float32).view(83, 8)`; `topk_ids` is
  `backing[83 * 8:].view(83, 8)`. The equal-width dtypes give disjoint byte
  spans `[0,2656)` and `[2656,5312)`.
- The backing is a forward-local variable. No cache, pool, reuse, model/module
  state, global mutable state, output recycling, synchronization, device
  context, or stream switch was introduced.
- Each fast-path call unconditionally creates a new backing. Both returned
  views retain that allocation while live; their element ranges do not overlap.
- MACA CUDA-tensor dtype-view was the explicit capability gate. It executed
  through the true remote harness without a copy fallback, second allocation,
  extra device kernel in candidate code, or algorithm change.
- The complete Triton kernel and launch are frozen from Round 001. No Triton
  primitive or hint changed.
- The fixed seeded harness passed correctness. Authoritative storage-identity,
  non-overlap, retained-lifetime, mutation-isolation, allocation-count,
  profiling, and wall gates remain Verifier-owned.

## Attempt Ledger

| Attempt | Command | Exit status | Defect / evidence | Candidate before SHA256 | Candidate after SHA256 |
|---:|---|---:|---|---|---|
| 1 | `validate_decision.py --expected-profile triton_maca rounds/decision_002.md` | 0 | None; immutable decision valid. | not-applicable | not-applicable |
| 2 | Reference adapter SHA and class-rename-only byte check | 0 | None. | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | same |
| 3 | Scoped canonical-copy patch `git apply --check --recount` | 0 | Patch preflight passed. | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | not-applicable |
| 4 | Scoped canonical-copy patch `git apply --recount` | 0 | Created allocation-only candidate; no defect. | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | `1cbfddc1fd91ef4d73e388758467962cb471fc2a5f508c0af0749dcce53080d1` |
| 5 | Final `ast.parse` plus exact canonical replacement check | 0 | None; all frozen regions byte-equivalent. | `1cbfddc1fd91ef4d73e388758467962cb471fc2a5f508c0af0749dcce53080d1` | same |
| 6 | Local actual `auto_bench.load_ks_module(candidate)` | 1 | Local WSL Python lacked `torch`; failure occurred importing `auto_bench.py` before candidate loading. Environment note, not candidate defect. | `1cbfddc1fd91ef4d73e388758467962cb471fc2a5f508c0af0749dcce53080d1` | same |
| 7 | Actual loader AST-filter precheck with minimal local import stubs | 0 | AST-filter/compile/exec precheck only; not a substitute for the true runtime loader. | `1cbfddc1fd91ef4d73e388758467962cb471fc2a5f508c0af0749dcce53080d1` | same |
| 8 | Orchestrator-assisted remote SHA check and unique `auto_bench.py --warmup 1 --repeat 1 --full-traceback` smoke | 0 | True loader, MACA dtype-view, unchanged Triton launch, execution, and fixed-seed correctness passed. | `1cbfddc1fd91ef4d73e388758467962cb471fc2a5f508c0af0749dcce53080d1` | same |

No repair attempt was required. The candidate was not uploaded directly by the
Coder; Orchestrator performed the authorized sync and unique remote gate.
