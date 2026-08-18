# Coder Result 003

Result: `candidate-ready`

- round: `003`
- source_canonical: `triton_grouped_topk_002.py`
- source_canonical_sha256: `90d7b09569d1d155c8e44e1626f2c0f3b3f41e0919a8a9e5b76719e874b17ce3`
- reference_adapter: `reference_triton_grouped_topk_002.py`
- reference_adapter_sha256: `9d3a368e93afc557d18eba6241df83757ec4c7478686809e90c7b8f1945fa8cd`
- decision: `rounds/decision_003.md`
- decision_sha256: `2f90569b0cbf786f217cd45fac38c51990d7b5c041dc1f9a5ac6e5ac38129594`
- candidate: `triton_grouped_topk_003.py`
- candidate_sha256: `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37`
- selected_profile: `triton_gcu`
- runtime_fingerprint: `project.md#runtime-fingerprint`
- measurement_fingerprint: `3942e25aebbe7690a55cf27768a3bc3fd552cc8106f6bd2dd7416cea2d274bf3`

## Implementation Conformance

- Added only a ModelNew-instance-owned metadata cache. Entries contain host
  `block_e`, `epg`, grid, and `num_warps` values.
- Cache keys include exact gating shape/dtype, output requirements, routing
  configuration, GCU device, current stream id, and launch configuration.
- Cache lookup/insertion uses the existing model-local output-pool lock;
  separate ModelNew instances have disjoint metadata dictionaries.
- Cache misses compute and insert immutable metadata; stream identity failures
  use an uncached safe miss path.
- The Triton kernel body, grid `(tokens,)`, constexpr values, direct launch,
  output pool, and `num_warps=1` remain unchanged.

## Guardrail Evidence

| Check | Result | Evidence |
|---|---|---|
| exact-key hit | PASS | S60 metadata command: cache size stayed 1 across repeated 256-expert calls |
| exact-key invalidation | PASS | 128-expert call increased cache size to 2; returning to 256 hit original entry |
| separate instance ownership | PASS | Separate model cache size 1 and dictionary identity differed |
| retained output lifetime | PASS | retained output remained stable and received distinct storage |
| concurrent output-pool safety | PASS | same-instance concurrent calls received distinct output storage |
| correctness against accepted reference | PASS | S60 smoke and three formal paired runs |

## Attempt Ledger

| Attempt | Command / change | Exit status | Defect | Candidate before SHA256 | Candidate after SHA256 |
|---:|---|---:|---|---|---|
| 1 | `python3 -m py_compile s60/groupedtopk/triton_grouped_topk_003.py` | 0 | none | `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37` | `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37` |
| 2 | S60 metadata hit/miss/invalidation and lifecycle checks | 0 | none; all guardrails PASS | `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37` | `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37` |
| 3 | S60 auto_bench smoke, three paired benchmarks, and paired profile | 0 | none; correctness PASS | `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37` | `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37` |
