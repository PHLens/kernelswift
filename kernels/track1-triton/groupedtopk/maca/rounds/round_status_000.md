# Round Status 000

- phase: `verification-complete`
- result: `baseline`
- candidate: `baseline_adapter.py`
- accepted_reference: `base.py` (Phase 0 baseline establishment)
- measurement_fingerprint: `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`
- timing_order: `sequential complete accepted-reference block, then complete candidate block`
- status_template: `missing at skills/kernel-opt-loop/references/round-status-template.md; fields materialized from verifier.md`

## Artifact Hashes

| Artifact | Local SHA256 | Remote SHA256 | Verdict |
|---|---|---|---|
| `base.py` | `49ec0cf7a48679c23c5187eb7ba546fd41025019a9d6e9b59e1273c48a31dfbb` | `49ec0cf7a48679c23c5187eb7ba546fd41025019a9d6e9b59e1273c48a31dfbb` | match |
| `baseline_adapter.py` | `d92a1be10bf9ad036287ed0f769b195262132893fa78d88fa30f992fbe757827` | `d92a1be10bf9ad036287ed0f769b195262132893fa78d88fa30f992fbe757827` | match |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | match |
| `rounds/report_000.md` | `9b8374ee96d72fa8eed02415440eb778867d9ee0f3d0e8914608695a0c299f00` | not-applicable | written |
| `state/verifier_context.md` | `35bac2ef274b1f149dc4630f336e6f692b35e9985362ec35009319495d6dfabd` | not-applicable | written |

## Completed Commands

1. Local SHA256 verification: exit `0`.
2. Remote SHA256 verification over the existing SSH control connection: exit `0`.
3. Frozen correctness command (`warmup=5`, `repeat=10`, `--full-traceback`): exit `0`; `PASS accuracy`; summary `1 passed, 0 failed, 1 total`.
4. Frozen wall sample 1 (`warmup=200`, `repeat=500`): exit `0`; reference `0.233009 ms`; candidate `0.231739 ms`; evidence `log/wall_000_sample_1.log`.
5. Frozen wall sample 2 (`warmup=200`, `repeat=500`): exit `0`; reference `0.233050 ms`; candidate `0.232012 ms`; evidence `log/wall_000_sample_2.log`.
6. Frozen wall sample 3 (`warmup=200`, `repeat=500`): exit `0`; reference `0.227752 ms`; candidate `0.226243 ms`; evidence `log/wall_000_sample_3.log`.
7. Frozen forward profiler (`profile-mode=forward`, `profile-warmup=20`, `profile-iterations=100`): exit `0`; raw trace `log/round_000_forward_100iter.pt.trace.json`.
8. Initial raw-trace `baseline_base` summary: exit `1`; `overlapping scope events`; evidence preserved in `log/profile_processing_000.log`.
9. Initial raw-trace `candidate_baseline_adapter` summary: exit `1`; `overlapping scope events`; evidence preserved in `log/profile_processing_000.log`.
10. Approved exact duplicate-marker filter: exit `0`; deleted exactly 2 nested `gpu_user_annotation` scope markers; kernel events remained `3000`, `cuda_runtime` remained `5202`.
11. Derived-trace `baseline_base` summary with unmodified repository script: exit `0`.
12. Derived-trace `candidate_baseline_adapter` summary with unmodified repository script: exit `0`.
13. Normalized-AST equivalence check: exit `0`; `normalized_ast_equal=True`.
14. Post-measurement remote SHA256 verification: exit `0`; all three frozen files still match.

## Raw Samples

- correctness: `pass`; smoke timing `v0=0.254451 ms`, `v1=0.252472 ms`; evidence `log/correctness_000.log`
- reference wall samples ms: `[0.233009, 0.233050, 0.227752]`; median `0.233009`
- candidate wall samples ms: `[0.231739, 0.232012, 0.226243]`; median `0.231739`
- descriptive difference: candidate is `0.5450433245067758%` faster than base by independent-sample medians; Round 000 establishes the baseline and makes no optimization adoption decision.
- profile raw trace SHA256: `d5fa6827a1c1e8b92f210fd06d0ccfe17b5a14098c2bf9b218e9f91b8e287da3`
- profile derived trace SHA256: `d712820f7adae8a164ddd996f18a37726c0abdf0bb2367719b15678be419e7e3`
- `baseline_base`: `147.3481103515625 us/call`, `15.0 kernels/call`, device ratio `0.6323708970536009`
- `candidate_baseline_adapter`: `147.7526708984375 us/call`, `15.0 kernels/call`, device ratio `0.6375822407900158`
- canonical Round 000 wall baseline: `baseline_adapter.py = 0.231739 ms`
- profiler summaries: `log/round_000_baseline_base_summary.json`, `log/round_000_candidate_baseline_adapter_summary.json`

## Next Safe Action

Orchestrator validates `rounds/report_000.md`, updates canonical Phase 0 pointers/state, and releases measurement exclusivity. Verifier must not mutate those pointers.
