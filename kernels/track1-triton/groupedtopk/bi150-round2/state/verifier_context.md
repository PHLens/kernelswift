# Verifier Context

- role_contract_sha256: `62f10a0940ca3665260226a7891f5d34e1b571e70937862bb02ad68aa2bbc82f`
- context_epoch: `1`
- last_completed_round: `004`
- accepted_kernel: `triton_grouped_topk_r2_004.py` (sha256 `c02d956c…21eb`) — pending Orchestrator commit
- accepted_report: `rounds/report_004.md` @`c79cc018…4fe35` Result: accepted, verdict `rounds/verdict_004.json` @`13340553…44a46`, fact-pack pin 19992a18…3e62
- recent_three_round_evidence:
  - `002 — compile-graph-default ACCEPTED: prescribed-pair wall 0.475034 -> 0.338824 ms (+28.67%); same-session accepted-pair +18.22%; kernels/call 6.97 -> 6.90; device flat ~104 us/call`
  - `003 — compile-graph-replay-reduce-overhead NO-IMPROVEMENT: framework skipped cudagraphs for mutated inputs EVERY invocation -> replay never fired; −8.0875% same-session vs accepted; retired from chain`
  - `004 — manual-cuda-graph-workspace-replay ACCEPTED (decisively): ACTIVE TIER manual-replay on both instances; per-call structure = guard + copy-in + ONE replay submission + 2 copy-outs (~3 aten::copy_ DtoD memcpys/call); attributed cat=kernel count ZERO inside candidate scope -> branch B taken with positive single-submission census evidence (log/diagnostic_scope_census_round004.json); bitwise==r002 True/True everywhere incl stale-trap & alternation & run_out poisoned ×2; T=41 selectivity zero-artifacts then tier-1 capture/recovery; WALL: prescribed basis ref median 0.474386 -> cand median 0.196909 ms (+58.50%); direct same-session accepted-pair r002 0.3463206812739372 -> r004 0.19897893071174622 (+42.54%); cold capture ≈144.7 ms one-time outside timed medians`
- open_hypotheses: `residual space for H-00x (Designer): remaining host residue ~93 µs/call at 0.1969 ms wall (two copy-outs fusable into one DtoD copy; allocator chatter for two fresh buffers; guard micro-costs); gate-limited device headroom behind CHECK-TIE audit for vendor top-k internals (~87 µs hidden device work/call)`

## Current Bottleneck

- `After round-004 manual workspace replay acceptance: wall 0.196909 ms carries graph-hidden device work (essentially the round-002 ~104 µs kernel band by construction) plus ~93 µs residual host/boundary time dominated by copy-in+copy-out submissions and two per-call fresh buffer allocations; biggest single controllable items are boundary-copy fusion and the CHECK-TIE-gated vendor selection internals.`

## Round Measurement Facts (authoritative, rolling)

- Round 002 (former accepted basis): wall pairs `[0.475034, 0.472995, 0.479432]` / `[0.338824, 0.338136, 0.344416]`; kernels/call 6.90 kernel-mode; device 103.985 µs/call.
- Round 004 (new accepted): wall authoritative v0 `[0.477596, 0.474386, 0.467383]` / v1 `[0.197615, 0.196909, 0.195931]`; decisive same-session accepted-pair +42.54488932633068%; candidate scope cat=kernel ZERO (branch B), host census = 3 boundary memcpys/call only.

## Open Checks

- `CANDIDATE scopes under manual-replay tiers have ZERO attributed cat=kernel events on this build ('scope has no kernel events' from canonical summarize_trace.py) — expected branch-B behavior going forward; record tool rejections as named attempts per convention but treat as diagnostic, not failure; use host-window category census for firing evidence.`
- `gpu_user_annotation double-record P1 pattern and torch.compile cache warmth observations remain standing checks from rounds 001–002.`

## Artifact Read Hashes (rolling ledger)

- ../base.py `12f33248…d0f58`; auto_bench.py `71fb3ad0…fe29`; baseline_adapter.py `ecce4dac…39fa`
- report_000.md `320b8b03…56fdd`; decision_001.md `93783baa…532b`; sketch_001.json `637917e0…6985`; triton_grouped_topk_r2_001.py `4ae64cad…8de3`
- decision_002.md `31c972fb…ff37`; sketch_002.json `0ccbec47…4cf3`; binding_statement_report.json(r2) `9315ba1b…a6b6`; triton_cuda.yaml `dc8fa4c0…b7ae`; triton_grouped_topk_r2_002.py `ad703266…ab12`
- decision_003.md `e214c29a…a403`; sketch_003.json pinned `4a909a11…a782`; binding_statement_report.json(r3) `b32eb677…e1c7`; triton_grouped_topk_r2_003.py `62f8883a…d38`; report_003.md `e00efc94…6bea`; verdict_003.json `9336749c…2134`
- decision_004.md `e5465d7d…3be1`; sketch_004.json pinned `ccf277f4…e59`; binding_statement_report.json(r4) `1e6b44a5…f9bc`; triton_grouped_topk_r2_004.py `c02d956c…21eb`
- verifier probes/results: log/probes/verifier_tie_runout_result_00{1,2,3,4}.json; paired-probe results r001v002/r002v003/r002v004; log/diagnostic_scope_census_round004.json
