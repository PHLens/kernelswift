# Verifier Context

- role_contract_sha256: `62f10a0940ca3665260226a7891f5d34e1b571e70937862bb02ad68aa2bbc82f`
- context_epoch: `1`
- last_completed_round: `001`
- accepted_kernel: `triton_grouped_topk_r2_001.py` (sha256 `4ae64cad913267f2198fec735e08f1b9490cafa1139d3a48ee11400aacb80de3`) — pending Orchestrator commit
- accepted_report: `rounds/report_001.md` (Result: accepted; verdict artifact `rounds/verdict_001.json`, fact-pack pin c46c3349…59c97)
- recent_three_round_evidence:
  - `000 — baseline: wall 0.483530/0.481109 ms, 14.94 kernels/call, device 180.11/178.84 us/call, ratio ~0.372 (adapter≈base identity)`
  - `001 — preprocess-fusion-triton-stages ACCEPTED: paired wall medians 0.470655 -> 0.416933 ms (+11.41%, ≥5% bar met, H-001 8% exceeded); kernels/call 14.94 -> 6.97 kernel-mode (7.01 forward); device us/call 180.45 -> 105.31 (-41.4%); retained gatherTopK 1.99/call (49.37 us) + bitonicSortKVInPlace 1.99/call (37.29 us) unchanged; Triton stages A/B/C = 7.34/5.73/5.58 us/call`
  - `(slot for round 002)`
- open_hypotheses: `H-002 (pending Designer): residual host-dispatch dominance (device share of wall only 0.253) and top-k machinery 86.7 of 105.3 us/call device`

## Round 001 Measurement Facts (authoritative)

- Correctness: harness comparator PASS ×7 invocations + verifier probe suites (tie-heavy: all-equal, two-expert-tie-same-group, structured-group-tie-boundary, duplicate-max-pairs-cross-group) ids exact every case; run_out==forward bitwise over poisoned buffers ×2.
- Wall medians (--warmup 50 --repeat 100): reference `0.470655 ms`, candidate `0.416933 ms`; three ordered pairs; screening pairs at --warmup 10 --repeat 20 both >10% faster → not screened out.
- Profiling: canonical kernel-mode succeeded via ModelNew.run_out (report_000 constraint satisfied); forward dual-scope collected as same-session supplementary evidence; scope totals cross-agree ±0.4%.
- Tooling note: this torch build double-records record_function spans (user_annotation + gpu_user_annotation); summarize_trace.py strict-overlap rejects dual-recorded scopes (P1). Offline host-window-only salvage convention codified in log/probes/verifier_scoped_resummarize_001.py.
- Fingerprint discipline held: measurement fingerprint recomputed and regime flags byte-identical across rounds; all input/anchor/candidate hashes re-verified pre/post measurement.

## Current Bottleneck

- `Post-round-001 execution is 3 Triton stages (18.65 us/call) around retained library top-k pair (86.66 us/call combined); wall remains host-dominated — device share fell to 0.253, so launch/dispatch + torch.empty temporaries dominate the residual 310 us of each 417 us call; next lever space per facts only: fewer/fused launches or allocation strategy on the candidate side.`

## Open Checks

- `gpu_user_annotation projection asymmetry: reference scope spans were single-recorded while candidate scope was double-recorded in the same forward trace; if future rounds see asymmetric scope rejection from summarize_trace.py, apply the P1 host-window salvage and count it a named failed attempt when the strict tool errors.`

## Artifact Read Hashes (rolling ledger)

- ../base.py `12f33248…d0f58`; auto_bench.py `71fb3ad0…fe29`; baseline_adapter.py `ecce4dac…39fa`; report_000.md `320b8b03…56fdd`; decision_001.md `93783baa…532b`; sketch_001.json `637917e0…6985`; binding_statement_report.json `5fbddd0d…11d0`; triton_cuda.yaml `dc8fa4c0…b7ae`; triton_grouped_topk_r2_001.py `4ae64cad…8de3`
