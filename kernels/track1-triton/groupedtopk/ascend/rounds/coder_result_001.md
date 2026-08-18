# Coder Result 001

Result: `candidate-ready`

- round: `001`
- source_canonical: `baseline_adapter.py`
- source_canonical_sha256: `3eda2738d12ed93f4718bf67eca276e1bbc09eb4e3f8fac6b724b5c9e4981134`
- decision: `rounds/decision_001.md`
- decision_sha256: `e57e3fb560f7d8b39ec1b1a90be80a144a59564415813d3a783758c1351ea344`
- candidate: `triton_grouped_topk_001.py`
- candidate_sha256: `b7b47d1fec7eaed59eba784dd3300393df12bdc94cab164b9e9d238afb39357a`
- selected_profile: `triton_ascend`
- runtime_fingerprint: `project.md#runtime-fingerprint`
- measurement_fingerprint: `d2dc2d5a61930039371da06149b3156c4911a136c6c5df859f50d68ea0e3b871`

## Primitive and Hint Conformance

- `tl.load`, `tl.store`, `tl.arange`, `tl.program_id`, `tl.zeros`, `tl.reshape`,
  `tl.max`, `tl.argmax`, `tl.sum`, `tl.exp`, `tl.where`, `tl.broadcast_to`,
  `tl.full`, and `tl.static_range` match the Ascend profile evidence.
- `num_warps=1` is the only selected launch hint and is proven on the recorded
  Ascend runtime.
- The candidate uses direct Triton launch (the proven Ascend launcher path).

## Attempt Ledger

| Attempt | Command | Exit status | Defect | Candidate before SHA256 | Candidate after SHA256 |
|---:|---|---:|---|---|---|
| 1 | `python3 -m py_compile kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_001.py` | 0 | none | not-applicable | `b7b47d1fec7eaed59eba784dd3300393df12bdc94cab164b9e9d238afb39357a` |
| 2 | `python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_001.py --warmup 1 --repeat 3 --full-traceback` | 0 | none; correctness PASS and compile smoke PASS | `b7b47d1fec7eaed59eba784dd3300393df12bdc94cab164b9e9d238afb39357a` | `b7b47d1fec7eaed59eba784dd3300393df12bdc94cab164b9e9d238afb39357a` |

No semantic repair was required. The real harness AST loader, Ascend NPU
runtime, and candidate compile/execution smoke all passed.
