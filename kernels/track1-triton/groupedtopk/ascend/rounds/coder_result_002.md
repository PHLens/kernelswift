# Coder Result 002

Result: `candidate-ready`

- round: `002`
- source_canonical: `triton_grouped_topk_001.py`
- source_canonical_sha256: `b7b47d1fec7eaed59eba784dd3300393df12bdc94cab164b9e9d238afb39357a`
- decision: `rounds/decision_002.md`
- decision_sha256: `a3b8aebf92a887ec07def2f9a3f804726db620b37f9b6e9f7bb7bbaba6aebf78`
- candidate: `triton_grouped_topk_002.py`
- candidate_sha256: `9315412c0dd7e2c56a6ce16924c74d7fbf0c4872edab454dd51ce2f62e91413f`
- selected_profile: `triton_ascend`
- runtime_fingerprint: `project.md#runtime-fingerprint`
- measurement_fingerprint: `d2dc2d5a61930039371da06149b3156c4911a136c6c5df859f50d68ea0e3b871`

## Decision Validation

`validate_decision.py rounds/decision_002.md --expected-profile triton_ascend`
returned `valid: true`. Metadata is `language=triton`, `backend=ascend`,
`target_profile=triton_ascend`, `change_scope=host`,
`change_family=allocation-reuse`, matching the manifest profile and the recorded
Ascend runtime fingerprint.

## Implementation

The candidate is a byte-for-byte copy of `triton_grouped_topk_001.py` in every
kernel, grid, launcher, and numerical-semantics region. The only change is the
host-side output-allocation path in `ModelNew`, per the Host Plan:

- `ModelNew.__init__` now initializes three instance attributes:
  `_output_cache_key = None`, `_weights = None`, `_ids = None` (state owner is the
  `ModelNew` instance; no global or class-level cache).
- `ModelNew.forward` builds the cache key `(tokens, self.topk, torch.float32,
  torch.int32, gating_output.device)`. When it equals `_output_cache_key`, the
  cached `_weights`/`_ids` buffers are reused; otherwise two fresh
  `torch.empty` tensors are allocated on `gating_output.device` and replace the
  cache. This satisfies the Host Plan cache key (tokens, topk, weights dtype,
  ids dtype, device), invalidation (replace on any component change), lifetime
  (model lifetime), and concurrency (one instance, no cross-instance sharing)
  requirements.
- No stream or device context is created; the buffers live on the
  caller-selected NPU device and the current stream is preserved.
- The Triton kernel `_grouped_topk_kernel`, its launch grid `[(tokens,)]`,
  `num_warps=1`, all `tl.*` primitives, and the returned `weights`/`ids` shapes
  and dtypes (`[tokens, topk]` fp32 / int32) are unchanged from round 001.

## Primitive and Hint Conformance

- No new primitive is introduced. The kernel uses exactly the same `tl.load`,
  `tl.store`, `tl.arange`, `tl.program_id`, `tl.zeros`, `tl.reshape`,
  `tl.max`, `tl.argmax`, `tl.sum`, `tl.exp`, `tl.where`, `tl.broadcast_to`,
  `tl.full`, and `tl.static_range` as the accepted round 001 candidate, all of
  which match the Ascend profile evidence.
- `num_warps=1` and direct Triton launch (the proven Ascend launcher path) are
  unchanged.
- The Host Plan conformance note from `prompts/coder_targets/triton_ascend.md`
  ("reusing output buffers requires an explicit Host Plan with cache keys,
  invalidation, device/stream behavior, and concurrency assumptions") is
  satisfied by decision 002's Host Plan, which this implementation follows.

## Attempt Ledger

| Attempt | Command | Exit status | Defect | Candidate before SHA256 | Candidate after SHA256 |
|---:|---|---:|---|---|---|
| 1 | `python3 -m py_compile kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_002.py` | 0 | none | not-applicable | `9315412c0dd7e2c56a6ce16924c74d7fbf0c4872edab454dd51ce2f62e91413f` |
| 2 | `python3 -c 'import ast; ast.parse(...)'` | 0 | none | `9315412c0dd7e2c56a6ce16924c74d7fbf0c4872edab454dd51ce2f62e91413f` | `9315412c0dd7e2c56a6ce16924c74d7fbf0c4872edab454dd51ce2f62e91413f` |
| 3 | `python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_002.py --warmup 5 --repeat 10 --full-traceback` | 0 | none; correctness PASS and compile smoke PASS | `9315412c0dd7e2c56a6ce16924c74d7fbf0c4872edab454dd51ce2f62e91413f` | `9315412c0dd7e2c56a6ce16924c74d7fbf0c4872edab454dd51ce2f62e91413f` |

Compile-smoke evidence: `py_compile` exit 0, `ast.parse` exit 0, and the real
harness AST loader (`auto_bench.py`) loaded the candidate, launched it on the
Ascend NPU, and reported `PASS accuracy; ... speedup=2.816x` (smoke run,
warmup 5 / repeat 10). No semantic repair was required.
