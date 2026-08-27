# Verifier Context

- role_contract_sha256: `62f10a0940ca3665260226a7891f5d34e1b571e70937862bb02ad68aa2bbc82f`
- context_epoch: `1`
- last_completed_round: `003`
- accepted_kernel: `triton_grouped_topk_r2_002.py` (sha256 `ad703266…ab12`) — REMAINS last_accepted after round 003 no-improvement (pointer transition owned by Orchestrator)
- accepted_report: `rounds/report_002.md` (accepted); round 003 terminal: `rounds/report_003.md` @`e00efc94…6bea` Result: no-improvement, verdict `rounds/verdict_003.json` @`9336749c…2134`, fact-pack pin e12c939c…be31
- recent_three_round_evidence:
  - `001 — preprocess-fusion-triton-stages ACCEPTED: paired wall 0.470655 -> 0.416933 ms (+11.41%); kernels/call 14.94 -> 6.97; device us/call 180.45 -> 105.31`
  - `002 — compile-graph-default ACCEPTED: prescribed-pair wall 0.475034 -> 0.338824 ms (+28.67%); same-session accepted-pair +18.22%; kernels/call 6.97 -> 6.90; device flat ~104 us/call in declared band`
  - `003 — compile-graph-replay-reduce-overhead NO-IMPROVEMENT: correctness/bitwise==r002/selectivity ALL PASS through active tier 'replayed', but framework emitted 'skipping cudagraphs due to mutated inputs (2 instances)' EVERY invocation -> replay never fired; reduce-overhead machinery = pure overhead; decisive same-session pair r002 0.3451220691204071 -> r003 0.373033806681633 = -8.087497166542533%; device flat 105.01 vs 103.99 confirms host-side regression only; kernel-count 6.94/call letter-trips the 6.90 ceiling via exactly 3 span-edge events (composition byte-identical) — documented mechanism-neutral in report_003`
- open_hypotheses: `open levers for H-00x (Designer's selection): CHECK-TIE on-device audit gate to enter vendor top-k sites (~85.5 of ~104 µs/call device); capture-compatible buffership redesign so the mutated-input exclusion stops blocking cudagraph capture on this build; anything else per Design authority`

## Current Bottleneck

- `Unchanged from round-002 acceptance: best-known configuration is compiled-default staged pipeline @wall 0.338824 ms with device ~104 µs/call (ratio ~0.307) — residual host time outside kernels (~235 µs/call) dominates; two entry paths identified and both currently blocked/documented (cudagraph capture defeated by mutated-input policy under buffer-carrying invocations; vendor top-k internals gated by CHECK-TIE audit).`

## Round Measurement Facts (authoritative, rolling)

- Round 002 (accepted basis): wall pairs `[0.475034, 0.472995, 0.479432]` / `[0.338824, 0.338136, 0.344416]`; kernels/call 6.90 kernel-mode; device 103.985 µs/call.
- Round 003: wall authoritative v0 `[0.481532, 0.478203, 0.452523]` / v1 `[0.374760, 0.374314, 0.353708]`; decisive same-session accepted-pair −8.087497166542533%; framework mutation-skip notice every invocation; attributed launches 6.94/call kernel-mode / 7.00 forward vs r002 basis 6.90.

## Open Checks

- `gpu_user_annotation double-record scope rejection (round-001 P1) has NOT recurred since round 001 (rounds 002–003 scopes all canonical-tool clean); keep the salvage convention available if strict-overlap rc=2 reappears and count a named failed attempt whenever it fires.`
- `torch.compile/inductor caches persist across processes under CoreX — cold-sanity numbers vary by prior-session warmth (coder smoke 562 ms capture vs verifier probe 313 ms); always labeled observation-only.`
- `If any future candidate relies on graph capture of buffer-writing regions on this build, first design for non-mutated captured inputs or expect systematic 'skipping cudagraphs due to mutated inputs' demotion (round-003 root cause).`

## Artifact Read Hashes (rolling ledger)

- ../base.py `12f33248…d0f58`; auto_bench.py `71fb3ad0…fe29`; baseline_adapter.py `ecce4dac…39fa`
- report_000.md `320b8b03…56fdd`; decision_001.md `93783baa…532b`; sketch_001.json `637917e0…6985`; triton_grouped_topk_r2_001.py `4ae64cad…8de3`
- decision_002.md `31c972fb…ff37`; sketch_002.json `0ccbec47…4cf3`; binding_statement_report.json(r2) `9315ba1b…a6b6`; triton_cuda.yaml `dc8fa4c0…b7ae`; triton_grouped_topk_r2_002.py `ad703266…ab12`
- decision_003.md `e214c29a…a403`; sketch_003.json pinned `4a909a11…a782`; binding_statement_report.json(r3) `b32eb677…e1c7`; triton_grouped_topk_r2_003.py `62f8883a…d38`; report_003.md `e00efc94…6bea`; verdict_003.json `9336749c…2134`
- verifier probes/results: log/probes/verifier_tie_runout_result_00{1,2,3}.json; paired probes r001v002/r002v003 result JSONs under log/probes
