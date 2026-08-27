# Verifier Context

- role_contract_sha256: `62f10a0940ca3665260226a7891f5d34e1b571e70937862bb02ad68aa2bbc82f`
- context_epoch: `1`
- last_completed_round: `005`
- accepted_kernel: `triton_grouped_topk_r2_004.py` (sha256 `c02d956c…21eb`) — REMAINS last_accepted after round 005 no-improvement (streak accounting owned by Orchestrator)
- accepted_report: `rounds/report_004.md` @`c79cc018…4fe35`; round 005 terminal: `rounds/report_005.md` @`ada9d94a…d122` Result: no-improvement, verdict `rounds/verdict_005.json` @`cd0b3016…03a7`, fact-pack pin 0c25e5ad…e96e
- recent_three_round_evidence:
  - `003 — compile-graph-replay-reduce-overhead NO-IMPROVEMENT: inductor skipped cudagraphs for mutated inputs every invocation → replay never fired; −8.0875% same-session`
  - `004 — manual-cuda-graph-workspace-replay ACCEPTED: per-call guard+copy-in+ONE replay+2 copy-outs; wall 0.474386→0.196909 protocol basis (+58.50%), direct accepted-pair +42.54%; candidate scope cat=kernel ZERO (branch B) with host census evidence`
  - `005 — boundary-dispatch-coalescing NO-IMPROVEMENT: strategy branch A-batched bound; python-dispatcher trips 3→2/call independently confirmed by verifier census (aten::_foreach_copy_ ×1/call) BUT gpu_memcpy still ~3/call and submissions ~7/call unchanged → wall noise-band (−0.04%/+2.34% direct pair; −0.56% cross-anchor); adoption bar ≈9.85 µs NOT cleared; streak now 1/3`
- open_hypotheses: `remaining levers all documented for Orchestrator/Designer economics: CHECK-TIE vendor-entry audit (~87 µs hidden device work/call inside replay), persistent-result-buffer ownership redesign (violates current Decision lines), operator-scope-exceeding multi-graph batching`

## Current Bottleneck

- `Unchanged from r004 acceptance at canonical level: wall 0.196909 ms = hidden device band (~104 µs by construction, tie-gate-locked vendor pair dominant) + ~93 µs residual host time whose composition is NOW PRECISELY KNOWN from round-005 census: ~7 cudaMemcpyAsync-class submissions + 2 fresh buffer allocs + guard/attr residue per call; python dispatch is exhausted as a lever (round-005 negative result).`

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
- decision_004.md `e5465d7d…3be1`; sketch_004.json pinned `ccf277f4…e59`; binding_statement_report.json(r4) `1e6b44a5…f9bc`; triton_grouped_topk_r2_004.py `c02d956c…21eb`
- decision_005.md `4a549653…6021`; sketch_005.json pinned `21d13b98…92de`; binding_statement_report.json(r5) `b28abf72…535e`; boundary_trip_census.json(coder) `e289a591…e15c`; triton_grouped_topk_r2_005.py `cf68ed77…8e9c`
- verifier probes/results: log/probes/verifier_tie_runout_result_00{1,2,3,4,5}.json; paired-probe results r001v002/r002v003/r002v004/r004v005; diagnostic censuses round 004/005 under log/
