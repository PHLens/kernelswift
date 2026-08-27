# Verifier Context

- role_contract_sha256: `62f10a0940ca3665260226a7891f5d34e1b571e70937862bb02ad68aa2bbc82f`
- context_epoch: `1`
- last_completed_round: `002`
- accepted_kernel: `triton_grouped_topk_r2_002.py` (sha256 `ad703266eb727f7725c8fa61ceaedcffc269e94291def703cb34279e5275ab12`) — pending Orchestrator commit
- accepted_report: `rounds/report_002.md` (Result: accepted @`bd0932b9…a36ce`; verdict artifact `rounds/verdict_002.json` @`db173df8…de5f7`, fact-pack pin ffe20fec…014)
- recent_three_round_evidence:
  - `000 — baseline: wall 0.483530/0.481109 ms, 14.94 kernels/call, device 180.11/178.84 us/call, ratio ~0.372 (adapter≈base identity)`
  - `001 — preprocess-fusion-triton-stages ACCEPTED: paired wall 0.470655 -> 0.416933 ms (+11.41%); kernels/call 14.94 -> 6.97; device us/call 180.45 -> 105.31; retained vendor pair unchanged; device_ratio 0.383 -> 0.253`
  - `002 — compile-graph-default ACCEPTED: prescribed-pair basis wall 0.475034 -> 0.338824 ms (+28.67%); same-session accepted-pair r001->r002 +18.22% (bitwise-equal outputs pre-timing); kernels/call 6.97 -> 6.90; device us/call 105.31 -> 103.99 inside declared 90–130 flat band; stage kernel names byte-persisted; gatherTopK/bitonicSortKVInPlace at 1.97 counts/call each; cold compile ≈2812.8 ms absorbed outside timed medians per invocation`
- open_hypotheses: `H-003 (pending Designer): remaining host share ≈69% of wall after default-mode compression; unused reduce-overhead/CUDA-graph replay lever (attribution-collapsing, deferred by design); top-k vendor pair still ~85.5 of ~104 µs/call device behind tie-exactness audit gate`

## Current Bottleneck

- `After round-002 compile dispatch compression: wall 0.338824 ms with device only ~104 µs/call (ratio 0.307) — residual host time outside kernels remains the dominant term (~235 µs/call); within device, the two retained torch.topk vendor kernels (gatherTopK+bitonicSort ≈85.5 µs/call) dominate and are gated by the open tie-exactness audit for any future selection-side work.`

## Round Measurement Facts (authoritative, rolling)

- Round 001: correctness PASS incl. 4 tie suites; wall pairs `0.484525/0.481109`, `0.483530/0.482140`, `0.452363/0.451582`; kernel-mode via run_out first-attempt success.
- Round 002: correctness PASS through COMPILED route incl. bitwise-vs-r001 on all probe cases + T=41 staged-fallback selectivity + run_out poisoned-buffer ×2; wall authoritative `[0.475034, 0.472995, 0.479432]` / `[0.338824, 0.338136, 0.344416]`; kernel-mode canonical again first-attempt success.

## Open Checks

- `gpu_user_annotation double-record scope rejection (round-001 P1) did NOT recur on any round-002 scope with the canonical tool; keep the P1 salvage convention available if strict-overlap rc=2 reappears and count a named failed attempt whenever it fires.`
- `torch.compile cache state persists across processes under CoreX (probe cold sanity 2812.8 ms vs coder smoke 3544 ms) — report such numbers as observation-only sanity, never benchmark evidence.`

## Artifact Read Hashes (rolling ledger)

- ../base.py `12f33248…d0f58`; auto_bench.py `71fb3ad0…fe29`; baseline_adapter.py `ecce4dac…39fa`
- report_000.md `320b8b03…56fdd`; decision_001.md `93783baa…532b`; sketch_001.json `637917e0…6985`; triton_grouped_topk_r2_001.py `4ae64cad…8de3`
- decision_002.md `31c972fb…ff37`; sketch_002.json `0ccbec47…4cf3`; binding_statement_report.json(r2) `9315ba1b…a6b6`; triton_cuda.yaml `dc8fa4c0…b7ae`; triton_grouped_topk_r2_002.py `ad703266…ab12`
- verifier probes/results: log/probes/verifier_tie_runout_result_001.json @`68892c95…c0af`, verifier_tie_runout_check/result_002 files; paired-probe result JSON logged in report_002.md
