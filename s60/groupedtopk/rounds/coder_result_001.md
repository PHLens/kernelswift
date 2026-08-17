# Coder Result 001

Result: `candidate-ready`

- round: `001`
- source_canonical: `baseline_adapter.py`
- source_canonical_sha256: `6713aa567c945e98628f5b3c58d2bf5d71c3df85af8ad19438c00a447890fdd1`
- decision: `rounds/decision_001.md`
- decision_sha256: `f49d72923a1e274a5ae00725947db509665c9ef899f0113c2db07a4d7336f6af`
- candidate: `triton_grouped_topk_001.py`
- candidate_sha256: `f42ff6b47b28996199bbe9b8df0a181db2834be99473453f3eea35df51df693e`
- selected_profile: `triton_gcu`
- runtime_fingerprint: `project.md#runtime-fingerprint`
- measurement_fingerprint: `3942e25aebbe7690a55cf27768a3bc3fd552cc8106f6bd2dd7416cea2d274bf3`

## Primitive and Hint Conformance

- `tl.load`, `tl.store`, `tl.arange`, `tl.program_id`, `tl.zeros`, `tl.reshape`,
  `tl.max`, and `tl.argmax` match the GCU profile evidence.
- Round 001 smoke extended the exact-regime evidence to `tl.sum`, `tl.exp`,
  `tl.where`, `tl.broadcast_to`, `tl.full`, and `tl.static_range`.
- `num_warps=1` is the only selected launch hint and is proven on the recorded
  GCU runtime.
- The candidate uses direct Triton launch because both observed fast-libentry
  import paths are unavailable on this runtime.

## Attempt Ledger

| Attempt | Command | Exit status | Defect | Candidate before SHA256 | Candidate after SHA256 |
|---:|---|---:|---|---|---|
| 1 | `python3 -m py_compile s60/groupedtopk/triton_grouped_topk_001.py` | 0 | none | not-applicable | `f42ff6b47b28996199bbe9b8df0a181db2834be99473453f3eea35df51df693e` |
| 2 | `cd /root/kernelswift-s60 && python3 auto_bench.py --v0_file base.py --v1_file triton_grouped_topk_001.py --warmup 1 --repeat 3 --full-traceback` | 0 | none; correctness PASS and compile smoke PASS | `f42ff6b47b28996199bbe9b8df0a181db2834be99473453f3eea35df51df693e` | `f42ff6b47b28996199bbe9b8df0a181db2834be99473453f3eea35df51df693e` |

No semantic repair was required. The real harness AST loader, GCU runtime, and
candidate compile/execution smoke all passed.
