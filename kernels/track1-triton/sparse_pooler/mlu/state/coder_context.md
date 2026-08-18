# Coder Context State

- role_contract_sha256: `f0b93280b9a8854a6d29e4fcd3ea57244d67b87969df57410531fe07dff29cc7`
- context_epoch: 2
- last_completed_round: `002`
- accepted_kernel: `triton_sparse_pooler_001.py`
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `001 candidate-ready (fused kernel BLOCK_V=1024); 002 candidate-ready (BLOCK_V=1024->2048, single-line constexpr change) -> no-improvement, falsified`
- open_hypotheses: `await Designer decision_003 for next canonical change; Coder will copy triton_sparse_pooler_001.py as the starting point`
- artifact_read_hashes: `see ledger below`

## Current Bottleneck

- (Verifier-backed) Candidate wall time 0.606758 ms with device_ratio 0.346 (mixed class). The fused `_sparse_pooler_max_kernel` at 98.73 us/call is the dominant device kernel. Round 002 BLOCK_V=2048 was falsified (fused kernel 99.71->102.42 us/call, +2.71). BLOCK_V=1024 remains the best-known tiling for the current reduction strategy.

## Recent Three-round Evidence

- Round 001 (candidate-ready -> accepted): fused kernel `_sparse_pooler_max_kernel`, BLOCK_V=1024, num_warps=1, grid (4,30), on-device seq_offset via bounded prefix scan, pooling=="sum" Python fallback. Candidate SHA `182f2ebb...`. No repair needed. Evidence: `rounds/coder_result_001.md`, `rounds/report_001.md`.
- Round 002 (candidate-ready -> no-improvement): byte-identical copy of 001 with `BLOCK_V = 1024` -> `BLOCK_V = 2048`. Grid (4,30) -> (4,15). Candidate SHA `62dc853d...`. No repair needed; BLOCK_V=2048 compiled and ran correctly on first attempt. Evidence: `rounds/coder_result_002.md`, `rounds/report_002.md`.

## Open Hypotheses or Checks

- Awaiting Designer's decision_003. Coder's starting point is `triton_sparse_pooler_001.py` (the accepted canonical), not the rejected 002.
- Proven primitives on this runtime: `tl.maximum`, `tl.log`, `tl.where`, `tl.full((BLOCK_V,), -inf, dtype=fp32)`, `tl.load` with `other=-inf` mask, `num_warps=1`. These do not need re-probing.
- `num_warps=2` is known to fail; do not use.
- `fast_libentry` is importable but not required unless the Host Plan demands launcher reduction.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `triton_sparse_pooler_001.py` | `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd` | 002 |
| `triton_sparse_pooler_002.py` | `62dc853db5423cb5d99ad53433f3fb35919abe901a64d6e3acb3d815ac678248` | 002 |
| `rounds/decision_002.md` | `0d39de9e280f6ffa2cc3d1d3322d393fa400eb8f405b7e7ee3ceb3ef845b3dd4` | 002 |
| `rounds/decision_001.md` | `0816c943dfcfd157c9c4268196f4779b9804b9107de5fff0ba135d66f4f5bc75` | 001 |
| `baseline_adapter.py` | `d7e69ed4b66a4193a4475cc307fc0929cef807f875785652df6cc36fb2c487e5` | 000 |
| `base.py` | `ccccbbefadf1d697341451b542f17392acc8a2b9e4a3a41e50b2f9d58dbf61de` | 000 |
