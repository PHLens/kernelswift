# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: `2`
- last_completed_round: `003`
- accepted_kernel: `triton_grouped_topk_003.py`
- accepted_report: `rounds/report_003.md`
- recent_three_round_evidence: `Round 003 accepted; exact-key host metadata cache passed hit/miss/invalidation and lifecycle checks; wall median 0.292588 ms -> 0.273673 ms.`
- open_hypotheses: `Launcher/context specialization is next, but no host-time attribution exists. Kernel dataflow changes require matched GCU device evidence or a same-runtime microbenchmark.`
- artifact_read_hashes: `decision_003.md, candidate source, reference adapter, coder_result_003.md, project.md, and team-state.md recorded in the Round 003 ledger.`

## Current Bottleneck

- The accepted candidate has safe output-buffer reuse, exact-key host metadata
  cache, and one direct Triton-GCU launch per call. GCU device duration remains
  unavailable from the recorded profiler exporter.

## Recent Three-round Evidence

- Round 001, accepted, `rounds/report_001.md`, `kernel-fusion`: `39.08693002628853%`
  wall improvement and runtime launches `12.0 -> 1.0` per call.
- Round 002, accepted, `rounds/report_002.md`, `allocation-reuse`:
  `9.02136875254568%` wall improvement; output lifetime and concurrency PASS.
- Round 003, accepted, `rounds/report_003.md`, `host-metadata-specialization`:
  `6.464721724746064%` wall improvement; metadata hit/miss and lifecycle PASS.

## Open Hypotheses or Checks

- Launcher/context specialization requires a targeted host decomposition before
  coding and must not change stream, device, synchronization, or launch count.
- Any future GCU kernel change must keep device-time claims unavailable unless a
  matched exporter provides attributable device durations.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `rounds/decision_003.md` | `2f90569b0cbf786f217cd45fac38c51990d7b5c041dc1f9a5ac6e5ac38129594` | 003 |
| `reference_triton_grouped_topk_002.py` | `9d3a368e93afc557d18eba6241df83757ec4c7478686809e90c7b8f1945fa8cd` | 003 |
| `triton_grouped_topk_003.py` | `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37` | 003 |
| `rounds/coder_result_003.md` | 71861c3 | 003 |
