# Round Status 004

- phase: `complete-awaiting-orchestrator`
- result: `no-improvement`
- decision: `rounds/decision_004.md`
- candidate: `triton_grouped_topk_004.py`
- accepted_reference: `triton_grouped_topk_001.py`
- harness_reference_adapter: `reference_triton_grouped_topk_001.py`
- measurement_fingerprint: `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`
- timing_order: `sequential complete accepted-reference-adapter block, then complete candidate block`
- wall_gate: `failed: 2.6856032379199086% < 5.0%`
- profile: `not-run: formal wall threshold failed`
- stop_recommendation: `valid-no-improvement-limit`
- policy_note: `Rounds 002, 003, and 004 are three consecutive valid no-improvement results after Round 001 acceptance`
- dispatch_record_correction: `final coder_result_004.md SHA256 is c4ca2fdb07cfa49ba8ce2363f1e9238362d8ba2463aab16ee2ea00f2707a1551; candidate remained unchanged`

## Artifact Hashes

| Artifact | SHA256 | Remote/record verdict |
|---|---|---|
| `triton_grouped_topk_001.py` | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | local/remote match |
| `reference_triton_grouped_topk_001.py` | `70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9` | local/remote match; rename-only adapter |
| `triton_grouped_topk_004.py` | `50da7d1b6483822273b0dda404b33fb79fd96fbd95e3f0a9a48d8c8a3e315683` | local/remote match |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | local/remote match |
| `base.py` | `49ec0cf7a48679c23c5187eb7ba546fd41025019a9d6e9b59e1273c48a31dfbb` | local/remote match |
| `rounds/decision_004.md` | `5aab9a8df7ab5664cf6a2374d945b06a3a76d60a5fff9e4134a229a75bf0f587` | final record read |
| `rounds/coder_result_004.md` | `c4ca2fdb07cfa49ba8ce2363f1e9238362d8ba2463aab16ee2ea00f2707a1551` | corrected final record read |
| `rounds/report_001.md` | `f2866692d8d1c4519e9a2028c7b1d707fcb4f9f945fd856f78139f2dbe2aec4a` | accepted evidence |
| `rounds/report_002.md` | `a5ad9cfe8ead4e1e3cf06ef990ea0817537af4c088219f1eed9a551055426365` | prior evidence |
| `rounds/report_003.md` | `6c645acf858745585d4f668546609dc9d3dbc3f7c1b8110a013193f6c89c2fdd` | prior evidence |
| `rounds/report_004.md` | `1679384012c3fd29ef9c36522ca7277a7af830ec29737b7bb4da39625371dcd5` | final non-empty artifact |
| `state/verifier_context.md` | `46224f338df439b80d043268ed865006b983192729d6c11fd13d85da91ddc9cf` | final non-empty artifact |

## Completed Commands

1. Mandatory Verifier contracts, target profile, current project/team state, final corrected Round 004 decision/Coder result/candidate, canonical/adapter, prior reports, and role contexts read: exit `0`.
2. Local and remote frozen SHA256 verification: exit `0`; all executable artifacts matched.
3. Exact source-delta audit: exit `0`; exact authorized delta only; kernel/launch/fallback/config and grad guards frozen; tuple materializations `2 -> 0`, hidden eligibility metadata `4 -> 0`, tensor device-property reads `5 -> 1`, and two independent `torch.empty` unchanged.
4. Standard correctness (`warmup=5`, `repeat=10`, `--full-traceback`): exit `0`; elapsed `15.3491005 s`; `PASS accuracy`; `1 passed, 0 failed`; smoke-only `reference=0.083263 ms`, `candidate=0.073877 ms`.
5. Combined semantic-probe local wrapper precheck attempt: exit `1` before probe execution due nested command-wrapper quoting; exact evidence retained; no remote process or candidate execution.
6. Corrected AST-only local precheck: exit `0`; probe SHA256 `f39dcdc58d98ff37ddfc7b24e8e9776f0e2d8dbd4b4dd5647500e87022029dc8`.
7. Remote semantic-probe SHA256 verification: exit `0`; exact match.
8. Single targeted semantic/tie/fallback/storage gate: exit `0`; elapsed `15.0939439 s`; all `18` cases and overall verdict `PASS`.
9. Formal wall sample 1 (`warmup=200`, `repeat=500`): exit `0`; elapsed `15.4091149 s`; `reference=0.067650 ms`, `candidate=0.065375 ms`.
10. Formal wall sample 2: exit `0`; elapsed `15.5491880 s`; `reference=0.072364 ms`, `candidate=0.070434 ms`.
11. Formal wall sample 3: exit `0`; elapsed `15.3127396 s`; `reference=0.068439 ms`, `candidate=0.066601 ms`.
12. Cross-invocation median: reference `0.068439 ms`, candidate `0.066601 ms`, unrounded improvement `2.6856032379199086%`, speedup `1.027597183225477x`; `5%` gate failed.
13. Targeted profiler: not run, as required after the formal wall gate failed.

## Raw Samples and Evidence

- standard correctness: `pass`; smoke-only `reference=0.083263 ms`, `candidate=0.073877 ms`; `log/correctness_004.log`
- source audit: `pass`; `log/source_equivalence_004.py`, `log/source_equivalence_004.log`
- semantic/tie/fallback/storage gate: `18/18 pass`; `log/semantic_guard_004.py`, `log/semantic_guard_004.log`
- reference wall samples ms: `[0.067650, 0.072364, 0.068439]`
- candidate wall samples ms: `[0.065375, 0.070434, 0.066601]`
- formal wall logs: `log/wall_004_sample_1.log`, `log/wall_004_sample_2.log`, `log/wall_004_sample_3.log`
- profile: `not-run`; no trace created

## Hypothesis and Policy Verdict

- hypothesis_id: `H-004`
- verdict: `partially-confirmed: source/semantic mechanism passed; primary wall claim falsified; wall-gated profiler mechanism uncollected`
- adoption: `rejected; canonical remains triton_grouped_topk_001.py`
- recommendation: `valid-no-improvement-limit`

## Next Safe Action

Orchestrator validates `rounds/report_004.md`, this status, and `state/verifier_context.md`; records Round 004 as the third consecutive valid no-improvement; retains Round 001 canonical/report pointers; applies the configured stop policy; and releases measurement exclusivity. Do not run the Round 004 profiler.
